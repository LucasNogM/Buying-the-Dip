"""Input data loading, API downloads, and standardized CSV caching."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import StringIO
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import http.cookiejar
import pandas as pd

from buying_the_dip.config import (
    AppConfig,
    IndexSourceConfig,
    RETRYABLE_HTTP_STATUS_CODES,
)
from buying_the_dip.models import LoadedIndexData

LOGGER = logging.getLogger(__name__)

PROVIDER_LABELS = {
    "marketwatch": "MarketWatch download",
    "stooq": "Stooq historical data",
    "yahoo_finance": "Yahoo Finance API",
}


@dataclass(slots=True)
class DownloadedFrame:
    """One freshly downloaded frame plus its source label."""

    frame: pd.DataFrame
    source_label: str


class BaseHTTPAPIClient:
    """Common HTTP utilities for external market data providers."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._request_lock = Lock()
        self._next_request_at = 0.0

    def _build_headers(
        self,
        *,
        accept: str,
        referer: str | None = None,
        origin: str | None = None,
    ) -> dict[str, str]:
        """Build browser-like headers for a market data request."""
        headers = {
            "User-Agent": self.config.data_api.user_agent,
            "Accept": accept,
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        }
        if referer:
            headers["Referer"] = referer
        if origin:
            headers["Origin"] = origin
        return headers

    def _perform_json_request(
        self,
        request: Request,
        *,
        opener=None,
    ) -> dict[str, object]:
        """Execute one JSON request and parse its payload."""
        with self._open_request(request, opener=opener) as response:
            return json.load(response)

    def _perform_text_request(
        self,
        request: Request,
        *,
        opener=None,
    ) -> str:
        """Execute one text request and decode the response body."""
        with self._open_request(request, opener=opener) as response:
            content = response.read()
        return content.decode("utf-8", errors="replace")

    def _open_request(self, request: Request, *, opener=None):
        """Open one HTTP request while respecting the configured pacing."""
        self._wait_for_request_slot()
        if opener is None:
            return urlopen(request, timeout=self.config.data_api.timeout_seconds)
        return opener.open(request, timeout=self.config.data_api.timeout_seconds)

    def _wait_for_request_slot(self) -> None:
        """Throttle outgoing requests to reduce provider-side rate limits."""
        pause_seconds = self.config.data_api.request_pause_seconds
        if pause_seconds <= 0:
            return

        with self._request_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_at - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._next_request_at = time.monotonic() + pause_seconds

    @staticmethod
    def _should_retry_http_error(
        error: HTTPError,
        attempt_number: int,
        total_attempts: int,
    ) -> bool:
        """Return whether one HTTP error should trigger a retry."""
        return (
            error.code in RETRYABLE_HTTP_STATUS_CODES
            and attempt_number < total_attempts
        )

    def _resolve_retry_delay(
        self,
        error: HTTPError | None,
        attempt_number: int,
    ) -> float:
        """Compute the wait time before the next retry."""
        retry_after_seconds = self._parse_retry_after(error)
        if retry_after_seconds is not None:
            return retry_after_seconds

        base_seconds = self.config.data_api.retry_backoff_seconds
        jitter_seconds = self.config.data_api.retry_jitter_seconds
        exponential_component = base_seconds * (2 ** max(0, attempt_number - 1))
        random_component = (
            random.uniform(0.0, jitter_seconds) if jitter_seconds > 0 else 0.0
        )
        return exponential_component + random_component

    @staticmethod
    def _parse_retry_after(error: HTTPError | None) -> float | None:
        """Return the server-specified retry delay when available."""
        if error is None or error.headers is None:
            return None

        retry_after_value = error.headers.get("Retry-After")
        if not retry_after_value:
            return None

        retry_after_value = retry_after_value.strip()
        if retry_after_value.isdigit():
            return float(retry_after_value)

        try:
            retry_datetime = parsedate_to_datetime(retry_after_value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        return max(0.0, retry_datetime.timestamp() - time.time())

    @staticmethod
    def _csv_payload_to_frame(
        payload_text: str,
        symbol: str,
        provider_name: str,
    ) -> pd.DataFrame:
        """Convert one CSV payload into the internal two-column schema."""
        stripped_payload = payload_text.strip()
        if not stripped_payload:
            raise RuntimeError(
                f"{provider_name} returned an empty response for '{symbol}'."
            )

        lowered_payload = stripped_payload.lower()
        if "<html" in lowered_payload:
            raise RuntimeError(
                f"{provider_name} returned HTML instead of CSV for '{symbol}'."
            )
        if "too many requests" in lowered_payload:
            raise RuntimeError(
                f"{provider_name} reported rate limiting for '{symbol}'."
            )

        first_line = stripped_payload.splitlines()[0].strip().lower()
        if not first_line.startswith("date,"):
            raise RuntimeError(
                f"{provider_name} returned an unexpected payload for '{symbol}'."
            )

        frame = pd.read_csv(StringIO(stripped_payload))
        renamed_columns = {column: str(column).strip().lower() for column in frame.columns}
        frame = frame.rename(columns=renamed_columns)

        if {"date", "close"} - set(frame.columns):
            raise RuntimeError(
                f"{provider_name} CSV for '{symbol}' must contain 'Date' and 'Close'."
            )

        normalized = frame[["date", "close"]].copy()
        normalized.columns = ["date", "price"]
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
        normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
        normalized = normalized.dropna(subset=["date", "price"])
        normalized = normalized.sort_values("date").drop_duplicates("date", keep="last")
        return normalized.reset_index(drop=True)


class YahooFinanceAPIClient(BaseHTTPAPIClient):
    """Download daily historical market data from Yahoo Finance."""

    _BASE_URLS = (
        "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    )

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._cookie_jar = http.cookiejar.CookieJar()
        self._opener = build_opener(HTTPCookieProcessor(self._cookie_jar))

    def fetch_history(
        self,
        symbol: str,
        period: str | None = None,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """Download the longest available history for one Yahoo Finance symbol."""
        data_api = self.config.data_api
        resolved_period = period or data_api.period
        resolved_interval = interval or data_api.interval
        query_string = urlencode(
            {
                "range": resolved_period,
                "interval": resolved_interval,
                "includePrePost": "false",
                "events": "div,splits,capitalGains",
            }
        )
        encoded_symbol = quote(symbol, safe="")
        total_attempts = data_api.max_retries + 1

        for attempt_number in range(1, total_attempts + 1):
            request = self._build_request(
                symbol=encoded_symbol,
                query_string=query_string,
                attempt_number=attempt_number,
            )

            try:
                self._prime_session(symbol)
                payload = self._perform_json_request(request, opener=self._opener)
                return self._payload_to_frame(payload, symbol)
            except HTTPError as error:
                if self._should_retry_http_error(error, attempt_number, total_attempts):
                    wait_seconds = self._resolve_retry_delay(error, attempt_number)
                    LOGGER.warning(
                        "Yahoo Finance HTTP %s for '%s'. Retry %s/%s in %.2f s.",
                        error.code,
                        symbol,
                        attempt_number,
                        total_attempts,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    f"Yahoo Finance HTTP error for '{symbol}': "
                    f"{error.code} {error.reason}."
                ) from error
            except (URLError, TimeoutError) as error:
                if attempt_number < total_attempts:
                    wait_seconds = self._resolve_retry_delay(None, attempt_number)
                    LOGGER.warning(
                        "Yahoo Finance connection issue for '%s'. Retry %s/%s in %.2f s. %s",
                        symbol,
                        attempt_number,
                        total_attempts,
                        wait_seconds,
                        error,
                    )
                    time.sleep(wait_seconds)
                    continue
                if isinstance(error, URLError):
                    raise RuntimeError(
                        f"Yahoo Finance connection error for '{symbol}': "
                        f"{error.reason}."
                    ) from error
                raise RuntimeError(f"Yahoo Finance timeout for '{symbol}'.") from error

        raise RuntimeError(f"Yahoo Finance download exhausted all retries for '{symbol}'.")

    def _build_request(
        self,
        symbol: str,
        query_string: str,
        attempt_number: int,
    ) -> Request:
        """Build one HTTP request for the Yahoo Finance chart endpoint."""
        base_url = self._BASE_URLS[(attempt_number - 1) % len(self._BASE_URLS)]
        return Request(
            base_url.format(symbol=symbol) + f"?{query_string}",
            headers=self._build_headers(
                accept="application/json,text/plain,*/*",
                referer="https://finance.yahoo.com/",
                origin="https://finance.yahoo.com",
            ),
        )

    def _prime_session(self, symbol: str) -> None:
        """Warm up the Yahoo session to obtain browser-like cookies."""
        encoded_symbol = quote(symbol, safe="")
        warm_up_request = Request(
            f"https://finance.yahoo.com/quote/{encoded_symbol}/history",
            headers=self._build_headers(
                accept=(
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,*/*;q=0.8"
                ),
                referer="https://finance.yahoo.com/",
                origin="https://finance.yahoo.com",
            ),
        )

        try:
            with self._open_request(warm_up_request, opener=self._opener):
                return
        except Exception as error:  # pragma: no cover - network dependent.
            LOGGER.debug("Yahoo session warm-up failed for '%s': %s", symbol, error)

    @staticmethod
    def _payload_to_frame(payload: dict[str, object], symbol: str) -> pd.DataFrame:
        """Convert the Yahoo Finance JSON payload into a normalized frame."""
        chart = payload.get("chart", {})
        if not isinstance(chart, dict):
            raise RuntimeError(
                f"Yahoo Finance returned an unexpected payload for '{symbol}'."
            )

        if chart.get("error") is not None:
            chart_error = chart["error"]
            description = (
                chart_error.get("description", "unknown error")
                if isinstance(chart_error, dict)
                else str(chart_error)
            )
            raise RuntimeError(
                f"Yahoo Finance returned an error for '{symbol}': {description}."
            )

        result_items = chart.get("result") or []
        if not isinstance(result_items, list) or not result_items:
            raise RuntimeError(f"Yahoo Finance returned no data for '{symbol}'.")

        result = result_items[0]
        if not isinstance(result, dict):
            raise RuntimeError(
                f"Yahoo Finance returned an invalid result for '{symbol}'."
            )

        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_entries = indicators.get("quote") or []
        if not timestamps or not quote_entries:
            raise RuntimeError(
                f"Yahoo Finance returned an empty history for '{symbol}'."
            )

        quote_frame = quote_entries[0]
        price_values = quote_frame.get("close") or []
        adjusted_entries = indicators.get("adjclose") or []
        if adjusted_entries:
            adjusted_values = adjusted_entries[0].get("adjclose") or []
            if len(adjusted_values) == len(timestamps):
                price_values = adjusted_values

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True),
                "price": pd.to_numeric(price_values, errors="coerce"),
            }
        )
        timezone_name = (result.get("meta") or {}).get("exchangeTimezoneName")
        if timezone_name:
            try:
                frame["date"] = frame["date"].dt.tz_convert(timezone_name)
            except Exception:  # pragma: no cover - timezone metadata varies.
                pass
        frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
        frame = frame.dropna(subset=["date", "price"])
        frame = frame.sort_values("date").drop_duplicates("date", keep="last")
        return frame.reset_index(drop=True)


class StooqAPIClient(BaseHTTPAPIClient):
    """Download historical CSV files from Stooq."""

    def fetch_history(self, symbol: str) -> pd.DataFrame:
        """Download one Stooq historical CSV file."""
        total_attempts = self.config.data_api.max_retries + 1
        encoded_symbol = quote(symbol, safe="")
        request = Request(
            f"https://stooq.com/q/d/l/?i=d&s={encoded_symbol}",
            headers=self._build_headers(
                accept="text/csv,text/plain,*/*",
                referer="https://stooq.com/",
            ),
        )

        for attempt_number in range(1, total_attempts + 1):
            try:
                payload_text = self._perform_text_request(request)
                return self._csv_payload_to_frame(payload_text, symbol, "Stooq")
            except HTTPError as error:
                if self._should_retry_http_error(error, attempt_number, total_attempts):
                    wait_seconds = self._resolve_retry_delay(error, attempt_number)
                    LOGGER.warning(
                        "Stooq HTTP %s for '%s'. Retry %s/%s in %.2f s.",
                        error.code,
                        symbol,
                        attempt_number,
                        total_attempts,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    f"Stooq HTTP error for '{symbol}': {error.code} {error.reason}."
                ) from error
            except (URLError, TimeoutError) as error:
                if attempt_number < total_attempts:
                    wait_seconds = self._resolve_retry_delay(None, attempt_number)
                    LOGGER.warning(
                        "Stooq connection issue for '%s'. Retry %s/%s in %.2f s. %s",
                        symbol,
                        attempt_number,
                        total_attempts,
                        wait_seconds,
                        error,
                    )
                    time.sleep(wait_seconds)
                    continue
                if isinstance(error, URLError):
                    raise RuntimeError(
                        f"Stooq connection error for '{symbol}': {error.reason}."
                    ) from error
                raise RuntimeError(f"Stooq timeout for '{symbol}'.") from error

        raise RuntimeError(f"Stooq download exhausted all retries for '{symbol}'.")


class MarketWatchAPIClient(BaseHTTPAPIClient):
    """Download historical CSV data from MarketWatch."""

    def fetch_history(
        self,
        symbol: str,
        *,
        instrument_type: str = "index",
        country_code: str | None = None,
        start_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Download one MarketWatch historical CSV file."""
        total_attempts = self.config.data_api.max_retries + 1
        end_datetime = datetime.now(tz=timezone.utc).strftime("%m/%d/%Y 23:59:59")
        requested_start_date = (
            pd.Timestamp(start_date).normalize()
            if start_date is not None
            else pd.Timestamp("1950-01-01")
        )
        start_datetime = requested_start_date.strftime("%m/%d/%Y 00:00:00")
        query_params = {
            "csvdownload": "true",
            "daterange": "dall",
            "downloadpartial": "false",
            "enddate": end_datetime,
            "frequency": "p1d",
            "newdates": "false",
            "startdate": start_datetime,
        }
        if country_code:
            query_params["countrycode"] = country_code

        request = Request(
            (
                f"https://www.marketwatch.com/investing/{instrument_type}/"
                f"{quote(symbol, safe='')}"
                f"/downloaddatapartial?{urlencode(query_params)}"
            ),
            headers=self._build_headers(
                accept="text/csv,text/plain,*/*",
                referer=(
                    f"https://www.marketwatch.com/investing/{instrument_type}/"
                    f"{quote(symbol, safe='')}/download-data"
                ),
            ),
        )

        for attempt_number in range(1, total_attempts + 1):
            try:
                payload_text = self._perform_text_request(request)
                return self._csv_payload_to_frame(
                    payload_text,
                    symbol,
                    "MarketWatch",
                )
            except HTTPError as error:
                if self._should_retry_http_error(error, attempt_number, total_attempts):
                    wait_seconds = self._resolve_retry_delay(error, attempt_number)
                    LOGGER.warning(
                        "MarketWatch HTTP %s for '%s'. Retry %s/%s in %.2f s.",
                        error.code,
                        symbol,
                        attempt_number,
                        total_attempts,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise RuntimeError(
                    f"MarketWatch HTTP error for '{symbol}': "
                    f"{error.code} {error.reason}."
                ) from error
            except (URLError, TimeoutError) as error:
                if attempt_number < total_attempts:
                    wait_seconds = self._resolve_retry_delay(None, attempt_number)
                    LOGGER.warning(
                        "MarketWatch connection issue for '%s'. Retry %s/%s in %.2f s. %s",
                        symbol,
                        attempt_number,
                        total_attempts,
                        wait_seconds,
                        error,
                    )
                    time.sleep(wait_seconds)
                    continue
                if isinstance(error, URLError):
                    raise RuntimeError(
                        f"MarketWatch connection error for '{symbol}': "
                        f"{error.reason}."
                    ) from error
                raise RuntimeError(f"MarketWatch timeout for '{symbol}'.") from error

        raise RuntimeError(
            f"MarketWatch download exhausted all retries for '{symbol}'."
        )


class IndexDataLoader:
    """Load market index datasets from files or an API with CSV cache fallback."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.yahoo_finance_client = YahooFinanceAPIClient(config)
        self.stooq_client = StooqAPIClient(config)
        self.marketwatch_client = MarketWatchAPIClient(config)

    def load_many(self, sources: list[IndexSourceConfig]) -> list[LoadedIndexData]:
        """Load all configured market indices while preserving source order."""
        api_frames, api_errors = self._download_api_frames(sources)
        loaded_sources: list[LoadedIndexData] = []
        for source in sources:
            try:
                if source.source_mode == "file":
                    loaded_sources.append(self.load(source))
                    continue

                if source.name in api_frames:
                    downloaded_frame = api_frames[source.name]
                    full_history_frame = self._finalize_frame(
                        downloaded_frame.frame,
                        source.name,
                    )
                    cache_path = self._resolve_cache_path(source)
                    self._save_cache(full_history_frame, cache_path)
                    loaded_sources.append(
                        self._build_loaded_index_data(
                            source=source,
                            full_history_frame=full_history_frame,
                            source_path=str(cache_path),
                            source_label=downloaded_frame.source_label,
                        )
                    )
                    continue

                loaded_sources.append(
                    self._load_from_cache(source, api_errors.get(source.name))
                )
            except Exception as error:
                if self.config.data_api.continue_on_index_error:
                    LOGGER.exception(
                        "Skipping index '%s' because it could not be loaded.",
                        source.name,
                    )
                    continue
                raise error

        if not loaded_sources:
            raise RuntimeError("No index data could be loaded successfully.")
        return loaded_sources

    def load(self, source: IndexSourceConfig) -> LoadedIndexData:
        """Load one market index source and return standardized data."""
        if source.source_mode != "file":
            raise ValueError(
                "The single-source loader only supports file-based sources. Use "
                "'load_many' for API-backed inputs."
            )
        raw_frame = self._read_frame(source)
        frame = self._normalize_file_frame(raw_frame, source)
        full_history_frame = self._finalize_frame(frame, source.name)
        return self._build_loaded_index_data(
            source=source,
            full_history_frame=full_history_frame,
            source_path=source.path,
            source_label="Local file",
        )

    def _download_api_frames(
        self,
        sources: list[IndexSourceConfig],
    ) -> tuple[dict[str, DownloadedFrame], dict[str, Exception]]:
        """Download all API-backed sources with optional controlled concurrency."""
        api_sources = [
            source
            for source in sources
            if source.source_mode in {"api", "api_with_cache"}
        ]
        if not api_sources:
            return {}, {}

        frames: dict[str, DownloadedFrame] = {}
        errors: dict[str, Exception] = {}
        max_workers = min(self.config.data_api.max_workers, len(api_sources))
        if max_workers <= 1:
            for source in api_sources:
                try:
                    frames[source.name] = self._download_api_source(source)
                except Exception as error:  # pragma: no cover - network dependent.
                    errors[source.name] = error
            return frames, errors

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._download_api_source, source): source
                for source in api_sources
            }
            for future in as_completed(future_map):
                source = future_map[future]
                try:
                    frames[source.name] = future.result()
                except Exception as error:  # pragma: no cover - network dependent.
                    errors[source.name] = error
        return frames, errors

    def _download_api_source(self, source: IndexSourceConfig) -> DownloadedFrame:
        """Download one API-backed source using its provider fallback chain."""
        provider_errors: list[str] = []
        requested_start_date = self._resolve_source_start_date(source.analysis_start_date)
        best_downloaded_frame: DownloadedFrame | None = None
        best_start_date: pd.Timestamp | None = None

        for provider in self._resolve_provider_sequence(source):
            try:
                LOGGER.info(
                    "Downloading '%s' from provider '%s'.",
                    source.name,
                    provider,
                )
                frame = self._download_with_provider(source, provider)
                if len(frame) < 2:
                    raise RuntimeError(
                        f"Provider '{provider}' returned fewer than two rows for "
                        f"'{source.name}'."
                    )

                provider_start_date = pd.Timestamp(frame["date"].iloc[0]).normalize()
                downloaded_frame = DownloadedFrame(
                    frame=frame,
                    source_label=PROVIDER_LABELS.get(provider, provider),
                )
                if (
                    best_downloaded_frame is None
                    or best_start_date is None
                    or provider_start_date < best_start_date
                ):
                    best_downloaded_frame = downloaded_frame
                    best_start_date = provider_start_date

                if (
                    requested_start_date is None
                    or provider_start_date <= requested_start_date
                ):
                    return downloaded_frame

                LOGGER.info(
                    "Provider '%s' for '%s' starts on %s, later than the requested %s. "
                    "Trying the next provider for a longer history.",
                    provider,
                    source.name,
                    provider_start_date.strftime("%Y-%m-%d"),
                    requested_start_date.strftime("%Y-%m-%d"),
                )
            except Exception as error:
                provider_errors.append(f"{provider}: {error}")
                LOGGER.warning(
                    "Provider '%s' failed for '%s': %s",
                    provider,
                    source.name,
                    error,
                )

        if best_downloaded_frame is not None and best_start_date is not None:
            if requested_start_date is not None:
                LOGGER.warning(
                    "No provider reached the requested analysis_start_date %s for '%s'. "
                    "Using the earliest available history, which starts on %s.",
                    requested_start_date.strftime("%Y-%m-%d"),
                    source.name,
                    best_start_date.strftime("%Y-%m-%d"),
                )
            return best_downloaded_frame

        error_details = "; ".join(provider_errors)
        raise RuntimeError(
            f"All configured providers failed for '{source.name}'. {error_details}"
        )

    def _resolve_provider_sequence(self, source: IndexSourceConfig) -> list[str]:
        """Return the ordered provider list for one source."""
        if source.provider_priority:
            providers = source.provider_priority
        elif source.provider:
            providers = [source.provider]
        else:
            providers = self.config.data_api.provider_priority
        return [provider.strip().lower() for provider in providers if provider.strip()]

    @staticmethod
    def _resolve_provider_symbol(source: IndexSourceConfig, provider: str) -> str:
        """Return the correct provider-specific symbol for one source."""
        symbol = source.provider_symbols.get(provider) or source.symbol
        if not symbol or not symbol.strip():
            raise RuntimeError(
                f"No symbol is configured for provider '{provider}' on "
                f"source '{source.name}'."
            )
        return symbol.strip()

    def _download_with_provider(
        self,
        source: IndexSourceConfig,
        provider: str,
    ) -> pd.DataFrame:
        """Download one source from one explicit provider."""
        symbol = self._resolve_provider_symbol(source, provider)
        if provider == "stooq":
            return self.stooq_client.fetch_history(symbol)
        if provider == "marketwatch":
            provider_options = source.provider_options.get("marketwatch", {})
            instrument_type = str(provider_options.get("instrument_type", "index"))
            country_code = provider_options.get("country_code")
            if country_code is None:
                country_code = provider_options.get("countrycode")
            if country_code is not None:
                country_code = str(country_code)
            return self.marketwatch_client.fetch_history(
                symbol,
                instrument_type=instrument_type,
                country_code=country_code,
                start_date=source.analysis_start_date,
            )
        if provider == "yahoo_finance":
            return self.yahoo_finance_client.fetch_history(
                symbol=symbol,
                period=source.api_period,
                interval=source.api_interval,
            )
        raise ValueError(
            f"Unsupported provider '{provider}' for source '{source.name}'."
        )

    def _load_from_cache(
        self,
        source: IndexSourceConfig,
        download_error: Exception | None,
    ) -> LoadedIndexData:
        """Load a cached CSV when a fresh API download is unavailable."""
        cache_path = self._resolve_cache_path(source)
        if cache_path.exists():
            frame = pd.read_csv(cache_path)
            frame = self._normalize_cache_frame(frame, source.name)
            full_history_frame = self._finalize_frame(frame, source.name)
            return self._build_loaded_index_data(
                source=source,
                full_history_frame=full_history_frame,
                source_path=str(cache_path),
                source_label="CSV cache fallback",
            )

        if source.source_mode == "api_with_cache" and download_error is not None:
            raise RuntimeError(
                f"Unable to download '{source.name}' from any configured provider, "
                f"and no CSV cache was found at '{cache_path}'. Original error: "
                f"{download_error}."
            ) from download_error

        raise RuntimeError(
            f"No CSV cache is available for source '{source.name}' at '{cache_path}'."
        )

    def _build_loaded_index_data(
        self,
        *,
        source: IndexSourceConfig,
        full_history_frame: pd.DataFrame,
        source_path: str,
        source_label: str,
    ) -> LoadedIndexData:
        """Build one loaded index object with per-index date filters applied."""
        analysis_start_date = self._resolve_source_start_date(source.analysis_start_date)
        simulation_start_date = self._resolve_source_start_date(
            source.simulation_start_date,
        )
        if simulation_start_date is None:
            simulation_start_date = analysis_start_date

        analysis_frame = self._apply_analysis_start_date(
            frame=full_history_frame,
            index_name=source.name,
            analysis_start_date=analysis_start_date,
        )
        effective_simulation_start = self._resolve_effective_simulation_start(
            analysis_frame,
            simulation_start_date,
        )
        return LoadedIndexData(
            index_name=source.name,
            frame=analysis_frame,
            source_path=source_path,
            source_label=source_label,
            symbol=source.symbol,
            analysis_start_date=analysis_start_date,
            simulation_start_date=effective_simulation_start,
        )

    @staticmethod
    def _resolve_source_start_date(value: str | None) -> pd.Timestamp | None:
        """Parse one optional per-index start date from the config."""
        if value is None:
            return None
        return pd.Timestamp(value).normalize()

    def _apply_analysis_start_date(
        self,
        *,
        frame: pd.DataFrame,
        index_name: str,
        analysis_start_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Apply one configured analysis start date and recompute derived fields."""
        date_price_frame = frame[["date", "price"]].copy()
        if analysis_start_date is not None:
            date_price_frame = date_price_frame[
                date_price_frame["date"] >= analysis_start_date
            ]
        if len(date_price_frame) < 2:
            requested_start = (
                analysis_start_date.strftime("%Y-%m-%d")
                if analysis_start_date is not None
                else "the configured range"
            )
            raise RuntimeError(
                f"Index '{index_name}' does not contain enough rows on or after "
                f"{requested_start}."
            )
        return self._finalize_frame(
            date_price_frame.reset_index(drop=True),
            index_name,
        )

    @staticmethod
    def _resolve_effective_simulation_start(
        frame: pd.DataFrame,
        simulation_start_date: pd.Timestamp | None,
    ) -> pd.Timestamp | None:
        """Clamp the simulation start date to the available analysis history."""
        if simulation_start_date is None:
            return None
        first_available_date = pd.Timestamp(frame["date"].iloc[0]).normalize()
        return max(first_available_date, simulation_start_date)

    @staticmethod
    def _read_frame(source: IndexSourceConfig) -> pd.DataFrame:
        """Read a CSV, Excel, or Parquet file."""
        path = Path(source.path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(
                path,
                sep=source.delimiter,
                encoding=source.encoding,
            )
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, sheet_name=source.sheet_name)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file type for '{path}'.")

    @staticmethod
    def _normalize_file_frame(
        frame: pd.DataFrame,
        source: IndexSourceConfig,
    ) -> pd.DataFrame:
        """Normalize a user-provided file frame into the internal schema."""
        missing_columns = {
            source.date_column,
            source.price_column,
        } - set(frame.columns)
        if missing_columns:
            missing_display = ", ".join(sorted(missing_columns))
            raise KeyError(
                f"Source '{source.name}' is missing required column(s): "
                f"{missing_display}."
            )

        normalized = frame[[source.date_column, source.price_column]].copy()
        normalized.columns = ["date", "price"]
        normalized["date"] = pd.to_datetime(
            normalized["date"],
            format=source.date_format,
            errors="coerce",
        ).dt.normalize()
        normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
        normalized = normalized.dropna(subset=["date", "price"])
        normalized = normalized.sort_values("date").drop_duplicates(
            "date",
            keep="last",
        )
        return normalized.reset_index(drop=True)

    @staticmethod
    def _normalize_cache_frame(frame: pd.DataFrame, index_name: str) -> pd.DataFrame:
        """Normalize a cached CSV frame into the internal schema."""
        if {"date", "price"} - set(frame.columns):
            raise KeyError(
                f"Cached CSV for '{index_name}' must contain 'date' and 'price'."
            )
        normalized = frame[["date", "price"]].copy()
        normalized["date"] = pd.to_datetime(
            normalized["date"],
            errors="coerce",
        ).dt.normalize()
        normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
        normalized = normalized.dropna(subset=["date", "price"])
        normalized = normalized.sort_values("date").drop_duplicates(
            "date",
            keep="last",
        )
        return normalized.reset_index(drop=True)

    @staticmethod
    def _finalize_frame(frame: pd.DataFrame, index_name: str) -> pd.DataFrame:
        """Add the derived columns required by the analytical pipeline."""
        if len(frame) < 2:
            raise ValueError(
                f"Source '{index_name}' must contain at least two valid rows."
            )
        if (frame["price"] <= 0).any():
            raise ValueError(f"Source '{index_name}' contains non-positive prices.")

        final_frame = frame.copy()
        iso_calendar = final_frame["date"].dt.isocalendar()
        final_frame["index_name"] = index_name
        final_frame["year"] = final_frame["date"].dt.year
        final_frame["month"] = final_frame["date"].dt.month
        final_frame["day"] = final_frame["date"].dt.day
        final_frame["month_key"] = (
            final_frame["date"].dt.year * 100 + final_frame["date"].dt.month
        ).astype(int)
        final_frame["quarter_key"] = (
            final_frame["date"].dt.year * 10 + final_frame["date"].dt.quarter
        ).astype(int)
        final_frame["week_key"] = (
            iso_calendar["year"].astype(int) * 100 + iso_calendar["week"].astype(int)
        )
        final_frame["return_factor"] = (
            final_frame["price"].pct_change().add(1.0).fillna(1.0)
        )
        return final_frame

    def _resolve_cache_path(self, source: IndexSourceConfig) -> Path:
        """Return the resolved cache file path for one API-backed source."""
        if source.cache_path:
            return Path(source.cache_path)
        safe_name = _slugify(source.name)
        return (
            Path(self.config.project.output_directory)
            / "data_cache"
            / f"{safe_name}.csv"
        )

    @staticmethod
    def _save_cache(frame: pd.DataFrame, path: Path) -> None:
        """Persist a standardized CSV cache for future fallback usage."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_frame = frame[["date", "price"]].copy()
        cache_frame["date"] = cache_frame["date"].dt.strftime("%Y-%m-%d")
        cache_frame.to_csv(path, index=False)


def _slugify(value: str) -> str:
    """Convert one label into a filesystem-friendly slug."""
    allowed_characters = [
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    ]
    collapsed = "".join(allowed_characters)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_") or "index"
