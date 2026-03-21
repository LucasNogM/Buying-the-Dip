"""Configuration models and YAML loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ALLOWED_ANCHORS = {"start", "end"}
ALLOWED_CASH_MODES = {"periodic", "linear"}
ALLOWED_CALENDAR_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly"}
ALLOWED_EXPORT_FORMATS = {"csv", "xlsx"}
ALLOWED_SOURCE_MODES = {"file", "api", "api_with_cache"}
ALLOWED_DATA_PROVIDERS = {"marketwatch", "stooq", "yahoo_finance"}
ALLOWED_REPORT_THEMES = {"light", "dark"}
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(slots=True)
class ProjectConfig:
    """Project-level settings."""

    title: str = "Buying the Dip Analysis"
    output_directory: str = "./outputs"
    log_level: str = "INFO"


@dataclass(slots=True)
class DataAPIConfig:
    """HTTP API download settings."""

    provider: str = "stooq"
    provider_priority: list[str] = field(
        default_factory=lambda: ["stooq", "marketwatch", "yahoo_finance"]
    )
    period: str = "max"
    interval: str = "1d"
    timeout_seconds: int = 25
    max_workers: int = 1
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    retry_jitter_seconds: float = 0.5
    request_pause_seconds: float = 1.25
    continue_on_index_error: bool = False
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )


@dataclass(slots=True)
class IndexSourceConfig:
    """Single input index source definition."""

    name: str
    source_mode: str = "file"
    path: str = ""
    symbol: str | None = None
    cache_path: str | None = None
    provider: str | None = None
    provider_priority: list[str] = field(default_factory=list)
    provider_symbols: dict[str, str] = field(default_factory=dict)
    provider_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    api_period: str | None = None
    api_interval: str | None = None
    analysis_start_date: str | None = None
    simulation_start_date: str | None = None
    date_column: str = "Date"
    price_column: str = "Close"
    date_format: str | None = None
    delimiter: str = ","
    encoding: str = "utf-8"
    sheet_name: str | int | None = None


@dataclass(slots=True)
class BaseStrategyConfig:
    """Base strategy parameters."""

    cash_mode: str = "periodic"
    contribution_amount: float = 1.0
    cash_frequency: str = "monthly"


@dataclass(slots=True)
class DCAConfig(BaseStrategyConfig):
    """Dollar-cost averaging settings."""

    period: str | int = "monthly"
    anchor: str = "end"


@dataclass(slots=True)
class OriginalBTDConfig(BaseStrategyConfig):
    """Original buying-the-dip settings."""

    enabled: bool = True
    implementable_enabled: bool = True
    implementable_drawdown_threshold_pct: float = 0.05


@dataclass(slots=True)
class PeriodicBTDConfig(BaseStrategyConfig):
    """Periodic buying-the-dip settings."""

    enabled: bool = True
    period_days: int = 30
    anchor: str = "end"
    implementable_enabled: bool = True
    implementable_drawdown_threshold_pct: float = 0.05
    implementable_fallback_to_window_end: bool = True


@dataclass(slots=True)
class StrategiesConfig:
    """Bundle of strategy configurations."""

    dca: DCAConfig = field(default_factory=DCAConfig)
    btd_original: OriginalBTDConfig = field(default_factory=OriginalBTDConfig)
    btd_periodic: PeriodicBTDConfig = field(default_factory=PeriodicBTDConfig)


@dataclass(slots=True)
class SimulationConfig:
    """Simulation sweep configuration."""

    enabled: bool = True
    horizons_years: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    )
    contribution_intervals_days: list[int] = field(
        default_factory=lambda: [15, 30, 45, 60, 90, 120]
    )
    window_count: int = 5
    anchor: str = "end"
    cash_mode: str = "periodic"
    cash_frequency: str = "monthly"
    contribution_amount: float = 1.0
    aggregations: dict[str, list[str]] = field(
        default_factory=lambda: {
            "annual_alpha_theoretical": ["mean", "median"],
            "annual_alpha_implementable": ["mean", "median"],
        }
    )


@dataclass(slots=True)
class ReportConfig:
    """HTML report configuration."""

    output_directory: str = "./outputs/reports"
    index_file_name: str = "index.html"
    table_precision: int = 4
    figure_dpi: int = 180
    include_original_btd: bool = True
    include_periodic_btd: bool = True
    include_simulation: bool = True
    include_implementable_analysis: bool = True
    include_consistency_analysis: bool = True
    max_simulation_rows: int = 12
    theme: str = "dark"
    include_overview_chart: bool = True
    advantage_clip_percentile: float = 97.5
    advantage_min_limit_pct: float = 0.25


@dataclass(slots=True)
class ExportConfig:
    """Raw result export configuration."""

    enabled: bool = True
    format: str = "csv"
    include_input_data: bool = False


@dataclass(slots=True)
class AppConfig:
    """Root configuration object."""

    project: ProjectConfig
    data_api: DataAPIConfig
    indices: list[IndexSourceConfig]
    strategies: StrategiesConfig
    simulation: SimulationConfig
    report: ReportConfig
    exports: ExportConfig

    @property
    def output_directory(self) -> Path:
        """Return the resolved project output directory."""

        return Path(self.project.output_directory)


class ConfigError(ValueError):
    """Raised when the YAML configuration is invalid."""


def load_config(config_path: str | Path) -> AppConfig:
    """Load the application configuration from YAML."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        raw_config = yaml.safe_load(file_handle) or {}

    project = ProjectConfig(**raw_config.get("project", {}))
    data_api = DataAPIConfig(**raw_config.get("data_api", {}))
    indices = [IndexSourceConfig(**item) for item in raw_config.get("indices", [])]
    strategies_section = raw_config.get("strategies", {})
    strategies = StrategiesConfig(
        dca=DCAConfig(**strategies_section.get("dca", {})),
        btd_original=OriginalBTDConfig(**strategies_section.get("btd_original", {})),
        btd_periodic=PeriodicBTDConfig(**strategies_section.get("btd_periodic", {})),
    )
    simulation = SimulationConfig(
        **_normalize_simulation_config(raw_config.get("simulation", {}))
    )
    report = ReportConfig(**raw_config.get("report", {}))
    exports = ExportConfig(**raw_config.get("exports", {}))

    config = AppConfig(
        project=project,
        data_api=data_api,
        indices=indices,
        strategies=strategies,
        simulation=simulation,
        report=report,
        exports=exports,
    )
    _validate_config(config)
    return config


def _normalize_simulation_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize simulation range inputs into explicit lists."""

    normalized = dict(raw_config)
    if "horizons_years" in normalized:
        normalized["horizons_years"] = _expand_numeric_sequence(
            normalized["horizons_years"]
        )
    if "contribution_intervals_days" in normalized:
        normalized["contribution_intervals_days"] = _expand_numeric_sequence(
            normalized["contribution_intervals_days"]
        )
    return normalized


def _expand_numeric_sequence(value: Any) -> list[int]:
    """Expand a numeric list or range-like mapping into a list of integers."""

    if isinstance(value, list):
        return [int(item) for item in value]

    if not isinstance(value, dict):
        raise ConfigError(
            "Numeric sequences must be provided as a list or as a mapping with "
            "start, stop, and step."
        )

    start = int(value["start"])
    stop = int(value["stop"])
    step = int(value["step"])
    inclusive = bool(value.get("inclusive", False))

    if step <= 0:
        raise ConfigError("Range step must be a positive integer.")

    upper_bound = stop + (1 if inclusive else 0)
    values = list(range(start, upper_bound, step))
    if not values:
        raise ConfigError("Expanded numeric sequence cannot be empty.")
    return values


def _validate_config(config: AppConfig) -> None:
    """Validate the loaded application configuration."""

    if not config.indices:
        raise ConfigError("At least one index source must be defined in 'indices'.")

    _validate_data_api_config(config.data_api)

    for source in config.indices:
        if not source.name.strip():
            raise ConfigError("Each index source requires a non-empty name.")
        _validate_source_config(source)

    _validate_strategy_config(config.strategies.dca)
    _validate_anchor(config.strategies.dca.anchor, "strategies.dca.anchor")
    _validate_period(config.strategies.dca.period, "strategies.dca.period")

    _validate_original_btd_config(config.strategies.btd_original)
    _validate_periodic_btd_config(config.strategies.btd_periodic)

    _validate_simulation_config(config.simulation)
    _validate_report_config(config.report)
    _validate_export_config(config.exports)


def _validate_source_config(source: IndexSourceConfig) -> None:
    """Validate one index source definition."""

    source.source_mode = source.source_mode.strip().lower()
    if source.source_mode not in ALLOWED_SOURCE_MODES:
        raise ConfigError(
            f"Index source '{source.name}' uses unsupported source_mode "
            f"'{source.source_mode}'."
        )

    analysis_start = _parse_optional_date(
        source.analysis_start_date,
        f"indices[{source.name}].analysis_start_date",
    )
    simulation_start = _parse_optional_date(
        source.simulation_start_date,
        f"indices[{source.name}].simulation_start_date",
    )
    if (
        analysis_start is not None
        and simulation_start is not None
        and simulation_start < analysis_start
    ):
        raise ConfigError(
            f"Index source '{source.name}' cannot use a simulation_start_date "
            "earlier than analysis_start_date."
        )

    if source.source_mode == "file":
        if not source.path.strip():
            raise ConfigError(
                f"Index source '{source.name}' requires a non-empty path when "
                "source_mode is 'file'."
            )
        return

    if not source.symbol or not source.symbol.strip():
        raise ConfigError(
            f"Index source '{source.name}' requires a non-empty symbol when using "
            "an API source mode."
        )

    resolved_providers = _resolve_provider_sequence(source)
    for provider in resolved_providers:
        if provider not in ALLOWED_DATA_PROVIDERS:
            raise ConfigError(
                f"Index source '{source.name}' uses unsupported provider "
                f"'{provider}'."
            )
    if source.cache_path and not source.cache_path.lower().endswith(".csv"):
        raise ConfigError(
            f"Index source '{source.name}' cache_path must end with '.csv'."
        )


def _resolve_provider_sequence(source: IndexSourceConfig) -> list[str]:
    """Return the ordered provider list for one source."""

    if source.provider_priority:
        return [provider.strip().lower() for provider in source.provider_priority]
    if source.provider:
        return [source.provider.strip().lower()]
    return ["yahoo_finance"]


def _validate_data_api_config(config: DataAPIConfig) -> None:
    """Validate the shared data API settings."""

    config.provider = config.provider.strip().lower()
    if config.provider not in ALLOWED_DATA_PROVIDERS:
        raise ConfigError(f"Unsupported data_api.provider '{config.provider}'.")

    config.provider_priority = [
        provider.strip().lower()
        for provider in config.provider_priority
        if provider and provider.strip()
    ]
    if not config.provider_priority:
        config.provider_priority = [config.provider]
    for provider in config.provider_priority:
        if provider not in ALLOWED_DATA_PROVIDERS:
            raise ConfigError(
                f"Unsupported provider in data_api.provider_priority: '{provider}'."
            )

    if config.timeout_seconds <= 0:
        raise ConfigError("'data_api.timeout_seconds' must be positive.")
    if config.max_workers <= 0:
        raise ConfigError("'data_api.max_workers' must be positive.")
    if config.max_retries < 0:
        raise ConfigError("'data_api.max_retries' cannot be negative.")
    if config.retry_backoff_seconds < 0:
        raise ConfigError("'data_api.retry_backoff_seconds' cannot be negative.")
    if config.retry_jitter_seconds < 0:
        raise ConfigError("'data_api.retry_jitter_seconds' cannot be negative.")
    if config.request_pause_seconds < 0:
        raise ConfigError("'data_api.request_pause_seconds' cannot be negative.")
    if not config.period.strip():
        raise ConfigError("'data_api.period' must not be empty.")
    if not config.interval.strip():
        raise ConfigError("'data_api.interval' must not be empty.")


def _validate_strategy_config(config: BaseStrategyConfig) -> None:
    """Validate the shared strategy configuration attributes."""

    if config.cash_mode not in ALLOWED_CASH_MODES:
        raise ConfigError(
            f"Unsupported cash mode '{config.cash_mode}'. Allowed values: "
            f"{sorted(ALLOWED_CASH_MODES)}."
        )
    if config.contribution_amount <= 0:
        raise ConfigError("Contribution amount must be positive.")
    if config.cash_frequency not in ALLOWED_CALENDAR_FREQUENCIES:
        raise ConfigError(
            f"Unsupported cash frequency '{config.cash_frequency}'. Allowed values: "
            f"{sorted(ALLOWED_CALENDAR_FREQUENCIES)}."
        )


def _validate_original_btd_config(config: OriginalBTDConfig) -> None:
    """Validate the original BtD settings."""

    _validate_strategy_config(config)
    _validate_drawdown_threshold(
        config.implementable_drawdown_threshold_pct,
        "strategies.btd_original.implementable_drawdown_threshold_pct",
    )


def _validate_periodic_btd_config(config: PeriodicBTDConfig) -> None:
    """Validate the periodic BtD settings."""

    _validate_strategy_config(config)
    if config.period_days <= 0:
        raise ConfigError(
            "'strategies.btd_periodic.period_days' must be a positive integer."
        )
    _validate_anchor(config.anchor, "strategies.btd_periodic.anchor")
    _validate_drawdown_threshold(
        config.implementable_drawdown_threshold_pct,
        "strategies.btd_periodic.implementable_drawdown_threshold_pct",
    )


def _validate_drawdown_threshold(value: float, field_name: str) -> None:
    """Validate an implementable BtD drawdown threshold."""

    if not 0.0 < float(value) < 1.0:
        raise ConfigError(
            f"'{field_name}' must be strictly between 0 and 1."
        )


def _parse_optional_date(
    value: str | None,
    field_name: str,
) -> pd.Timestamp | None:
    """Parse one optional ISO-style date string for config validation."""

    if value is None:
        return None
    if not str(value).strip():
        raise ConfigError(f"'{field_name}' must not be empty when provided.")
    try:
        return pd.Timestamp(value).normalize()
    except Exception as error:  # pragma: no cover
        raise ConfigError(
            f"'{field_name}' must be a valid date string, for example '1950-01-01'."
        ) from error


def _validate_anchor(value: str, field_name: str) -> None:
    """Validate a strategy or simulation anchor value."""

    if value not in ALLOWED_ANCHORS:
        raise ConfigError(
            f"Unsupported anchor '{value}' for '{field_name}'. Allowed values: "
            f"{sorted(ALLOWED_ANCHORS)}."
        )


def _validate_period(value: str | int, field_name: str) -> None:
    """Validate an investment period value."""

    if isinstance(value, int):
        if value <= 0:
            raise ConfigError(f"'{field_name}' must be a positive integer.")
        return
    if value not in ALLOWED_CALENDAR_FREQUENCIES:
        raise ConfigError(
            f"Unsupported period '{value}' for '{field_name}'. Allowed values: "
            f"positive integers or {sorted(ALLOWED_CALENDAR_FREQUENCIES)}."
        )


def _validate_simulation_config(config: SimulationConfig) -> None:
    """Validate the simulation sweep configuration."""

    if config.window_count <= 0:
        raise ConfigError("'simulation.window_count' must be positive.")
    if not config.horizons_years:
        raise ConfigError("'simulation.horizons_years' cannot be empty.")
    if not config.contribution_intervals_days:
        raise ConfigError("'simulation.contribution_intervals_days' cannot be empty.")
    if any(value <= 0 for value in config.horizons_years):
        raise ConfigError("All simulation horizons must be positive.")
    if any(value <= 0 for value in config.contribution_intervals_days):
        raise ConfigError("All simulation contribution intervals must be positive.")
    _validate_anchor(config.anchor, "simulation.anchor")
    if config.cash_mode not in ALLOWED_CASH_MODES:
        raise ConfigError(
            f"Unsupported simulation cash mode '{config.cash_mode}'. Allowed values: "
            f"{sorted(ALLOWED_CASH_MODES)}."
        )
    if config.cash_frequency not in ALLOWED_CALENDAR_FREQUENCIES:
        raise ConfigError(
            f"Unsupported simulation cash frequency '{config.cash_frequency}'."
        )
    if config.contribution_amount <= 0:
        raise ConfigError("Simulation contribution amount must be positive.")


def _validate_report_config(config: ReportConfig) -> None:
    """Validate the report settings."""

    if config.table_precision < 0:
        raise ConfigError("'report.table_precision' cannot be negative.")
    if config.figure_dpi <= 0:
        raise ConfigError("'report.figure_dpi' must be positive.")
    if config.max_simulation_rows <= 0:
        raise ConfigError("'report.max_simulation_rows' must be positive.")
    config.theme = config.theme.strip().lower()
    if config.theme not in ALLOWED_REPORT_THEMES:
        raise ConfigError(
            f"Unsupported report theme '{config.theme}'. Allowed values: "
            f"{sorted(ALLOWED_REPORT_THEMES)}."
        )
    if not 50.0 <= config.advantage_clip_percentile <= 100.0:
        raise ConfigError(
            "'report.advantage_clip_percentile' must be between 50 and 100."
        )
    if config.advantage_min_limit_pct <= 0:
        raise ConfigError("'report.advantage_min_limit_pct' must be positive.")


def _validate_export_config(config: ExportConfig) -> None:
    """Validate the export settings."""

    if config.format not in ALLOWED_EXPORT_FORMATS:
        raise ConfigError(
            f"Unsupported export format '{config.format}'. Allowed values: "
            f"{sorted(ALLOWED_EXPORT_FORMATS)}."
        )
