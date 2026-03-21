"""Analysis orchestration, comparisons, simulations, and exports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from buying_the_dip.config import AppConfig
from buying_the_dip.models import (
    ComparisonResult,
    IndexAnalysisResult,
    LoadedIndexData,
    SimulationOutput,
    StrategyOutput,
)
from buying_the_dip.portfolio import PortfolioEngine
from buying_the_dip.strategies import (
    BuyTheDipOriginalStrategy,
    BuyTheDipPeriodicStrategy,
    DCAStrategy,
)

LOGGER = logging.getLogger(__name__)


class StrategyComparator:
    """Compare two strategy runs."""

    @staticmethod
    def compare(
        strategy_a: StrategyOutput,
        strategy_b: StrategyOutput,
        label: str,
    ) -> ComparisonResult:
        """Return a full comparison between two strategy outputs."""

        comparison_frame = pd.DataFrame(
            {
                "date": strategy_a.frame["date"],
                "year": strategy_a.frame["year"],
                "investment_signal_a": strategy_a.frame["investment_signal"],
                "investment_signal_b": strategy_b.frame["investment_signal"],
                "wealth_a": strategy_a.frame["wealth"],
                "wealth_b": strategy_b.frame["wealth"],
                "wealth_return_a": strategy_a.frame["wealth_return"],
                "wealth_return_b": strategy_b.frame["wealth_return"],
            }
        )
        wealth_b = comparison_frame["wealth_b"].to_numpy(dtype=float)
        relative_advantage = np.divide(
            comparison_frame["wealth_a"].to_numpy(dtype=float),
            wealth_b,
            out=np.full(len(comparison_frame), np.nan, dtype=float),
            where=wealth_b != 0,
        ) - 1.0
        comparison_frame["relative_advantage"] = relative_advantage

        series = comparison_frame["relative_advantage"]
        analysis_years = max(
            (
                comparison_frame["date"].iloc[-1] - comparison_frame["date"].iloc[0]
            ).days
            / 365.25,
            np.finfo(float).eps,
        )
        annual_return_a = _annualize_return(strategy_a.final_return, analysis_years)
        annual_return_b = _annualize_return(strategy_b.final_return, analysis_years)
        annual_alpha = annual_return_a - annual_return_b

        return ComparisonResult(
            label=label,
            strategy_a_name=strategy_a.name,
            strategy_b_name=strategy_b.name,
            frame=comparison_frame,
            final_relative_advantage=float(series.iloc[-1]),
            mean_relative_advantage=float(series.mean(skipna=True)),
            median_relative_advantage=float(series.median(skipna=True)),
            std_relative_advantage=float(series.std(skipna=True)),
            percent_time_strategy_a_better=float((series > 0).mean(skipna=True)),
            annual_return_strategy_a=float(annual_return_a),
            annual_return_strategy_b=float(annual_return_b),
            annual_alpha=float(annual_alpha),
        )


class SimulationRunner:
    """Run periodic buy-the-dip versus DCA simulations."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(
        self,
        data: pd.DataFrame,
        simulation_start_date: pd.Timestamp | None = None,
    ) -> SimulationOutput | None:
        """Run the configured simulation sweep for a single index."""

        simulation_config = self.config.simulation
        if not simulation_config.enabled:
            return None

        date_values = data["date"].to_numpy(dtype="datetime64[ns]")
        first_date = pd.Timestamp(data["date"].iloc[0])
        last_date = pd.Timestamp(data["date"].iloc[-1])
        if simulation_start_date is None:
            eligible_first_date = first_date
        else:
            eligible_first_date = max(first_date, simulation_start_date.normalize())
        available_days = int((last_date - eligible_first_date).days)

        records: list[dict[str, object]] = []
        for horizon_years in simulation_config.horizons_years:
            horizon_days = int(round(horizon_years * 365.25))
            if horizon_days <= 0 or horizon_days > available_days:
                continue

            max_start_offset = max(available_days - horizon_days, 0)
            if simulation_config.window_count == 1:
                start_offsets = np.array([0], dtype=int)
            else:
                start_offsets = np.linspace(
                    0,
                    max_start_offset,
                    simulation_config.window_count,
                    dtype=int,
                )
                start_offsets = np.unique(start_offsets)

            for contribution_interval_days in (
                simulation_config.contribution_intervals_days
            ):
                for window_number, start_offset in enumerate(start_offsets):
                    start_date = eligible_first_date + pd.Timedelta(days=int(start_offset))
                    end_date = start_date + pd.Timedelta(days=horizon_days)
                    if end_date > last_date:
                        continue

                    start_position = int(
                        np.searchsorted(date_values, start_date.to_datetime64(), side="left")
                    )
                    end_position = int(
                        np.searchsorted(date_values, end_date.to_datetime64(), side="right")
                    )
                    if end_position - start_position < 2:
                        continue

                    window_frame = data.iloc[start_position:end_position].reset_index(
                        drop=True
                    )
                    dca_output, btd_output = self._run_periodic_pair(
                        window_frame,
                        contribution_interval_days,
                    )
                    comparison = StrategyComparator.compare(
                        strategy_a=btd_output,
                        strategy_b=dca_output,
                        label=(
                            f"{btd_output.name} vs. {dca_output.name} "
                            f"({contribution_interval_days} days)"
                        ),
                    )
                    records.append(
                        {
                            "horizon_years": horizon_years,
                            "contribution_interval_days": contribution_interval_days,
                            "window_stride_days": int(start_offset),
                            "window_number": window_number,
                            "start_date": start_date,
                            "end_date": end_date,
                            "final_relative_advantage": (
                                comparison.final_relative_advantage
                            ),
                            "mean_relative_advantage": (
                                comparison.mean_relative_advantage
                            ),
                            "median_relative_advantage": (
                                comparison.median_relative_advantage
                            ),
                            "std_relative_advantage": (
                                comparison.std_relative_advantage
                            ),
                            "percent_time_btd_better": (
                                comparison.percent_time_strategy_a_better
                            ),
                            "annual_return_btd": (
                                comparison.annual_return_strategy_a
                            ),
                            "annual_return_dca": (
                                comparison.annual_return_strategy_b
                            ),
                            "annual_alpha": comparison.annual_alpha,
                        }
                    )

        if not records:
            LOGGER.warning(
                "Simulation produced no valid windows for index '%s'.",
                data["index_name"].iloc[0],
            )
            empty_frame = pd.DataFrame()
            return SimulationOutput(
                raw_results=empty_frame,
                aggregated_results=empty_frame,
            )

        raw_results = pd.DataFrame.from_records(records)
        aggregated_results = self._aggregate_simulation_results(raw_results)
        return SimulationOutput(
            raw_results=raw_results,
            aggregated_results=aggregated_results,
        )

    def _run_periodic_pair(
        self,
        data: pd.DataFrame,
        contribution_interval_days: int,
    ) -> tuple[StrategyOutput, StrategyOutput]:
        """Run DCA and periodic BtD for one simulation window."""

        simulation_config = self.config.simulation

        dca_strategy = DCAStrategy(
            period=contribution_interval_days,
            anchor=simulation_config.anchor,
        )
        btd_strategy = BuyTheDipPeriodicStrategy(
            period_days=contribution_interval_days,
            anchor=simulation_config.anchor,
        )
        engine = PortfolioEngine(
            cash_mode=simulation_config.cash_mode,
            contribution_amount=simulation_config.contribution_amount,
            cash_frequency=simulation_config.cash_frequency,
        )

        dca_output = engine.run(
            data=data,
            strategy_name=dca_strategy.name,
            signal=dca_strategy.generate_signal(data),
        )
        btd_output = engine.run(
            data=data,
            strategy_name=btd_strategy.name,
            signal=btd_strategy.generate_signal(data),
        )
        return dca_output, btd_output

    def _aggregate_simulation_results(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the raw simulation output according to the config."""

        aggregation_map = self.config.simulation.aggregations
        aggregated = data.groupby(
            ["horizon_years", "contribution_interval_days"],
            as_index=False,
        ).agg(aggregation_map)
        aggregated.columns = [
            "_".join(str(part) for part in column if part)
            if isinstance(column, tuple)
            else str(column)
            for column in aggregated.columns
        ]
        return aggregated


class IndexAnalyzer:
    """Run all enabled analyses for a single index."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.simulation_runner = SimulationRunner(config)

    def analyze(
        self,
        loaded_data: LoadedIndexData,
    ) -> IndexAnalysisResult:
        """Run the configured strategies, comparisons, and simulation."""

        LOGGER.info("Running analysis for %s", loaded_data.index_name)

        data = loaded_data.frame
        dca_output = self._run_dca(data)
        btd_original = self._run_original_btd(data)
        btd_periodic = self._run_periodic_btd(data)

        original_vs_dca = None
        if btd_original is not None:
            original_vs_dca = StrategyComparator.compare(
                strategy_a=btd_original,
                strategy_b=dca_output,
                label=f"{btd_original.name} vs. {dca_output.name}",
            )

        periodic_vs_dca = None
        if btd_periodic is not None:
            periodic_vs_dca = StrategyComparator.compare(
                strategy_a=btd_periodic,
                strategy_b=dca_output,
                label=f"{btd_periodic.name} vs. {dca_output.name}",
            )

        simulation = self.simulation_runner.run(data, loaded_data.simulation_start_date)
        return IndexAnalysisResult(
            index_name=loaded_data.index_name,
            source_path=loaded_data.source_path,
            source_label=loaded_data.source_label,
            symbol=loaded_data.symbol,
            input_data=data,
            dca=dca_output,
            btd_original=btd_original,
            btd_periodic=btd_periodic,
            original_vs_dca=original_vs_dca,
            periodic_vs_dca=periodic_vs_dca,
            simulation=simulation,
        )

    def _run_dca(self, data: pd.DataFrame) -> StrategyOutput:
        strategy = DCAStrategy(
            period=self.config.strategies.dca.period,
            anchor=self.config.strategies.dca.anchor,
        )
        engine = PortfolioEngine(
            cash_mode=self.config.strategies.dca.cash_mode,
            contribution_amount=self.config.strategies.dca.contribution_amount,
            cash_frequency=self.config.strategies.dca.cash_frequency,
        )
        return engine.run(
            data=data,
            strategy_name=strategy.name,
            signal=strategy.generate_signal(data),
        )

    def _run_original_btd(self, data: pd.DataFrame) -> StrategyOutput | None:
        strategy_config = self.config.strategies.btd_original
        if not strategy_config.enabled:
            return None

        strategy = BuyTheDipOriginalStrategy()
        engine = PortfolioEngine(
            cash_mode=strategy_config.cash_mode,
            contribution_amount=strategy_config.contribution_amount,
            cash_frequency=strategy_config.cash_frequency,
        )
        return engine.run(
            data=data,
            strategy_name=strategy.name,
            signal=strategy.generate_signal(data),
        )

    def _run_periodic_btd(self, data: pd.DataFrame) -> StrategyOutput | None:
        strategy_config = self.config.strategies.btd_periodic
        if not strategy_config.enabled:
            return None

        strategy = BuyTheDipPeriodicStrategy(
            period_days=strategy_config.period_days,
            anchor=strategy_config.anchor,
        )
        engine = PortfolioEngine(
            cash_mode=strategy_config.cash_mode,
            contribution_amount=strategy_config.contribution_amount,
            cash_frequency=strategy_config.cash_frequency,
        )
        return engine.run(
            data=data,
            strategy_name=strategy.name,
            signal=strategy.generate_signal(data),
        )


class ResultExporter:
    """Export raw outputs to CSV or XLSX files."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def export_all(self, results: Iterable[IndexAnalysisResult]) -> None:
        """Export all analysis results if enabled in the config."""

        if not self.config.exports.enabled:
            return

        output_directory = self.config.output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        for result in results:
            safe_name = _slugify(result.index_name)
            if self.config.exports.format == "csv":
                self._export_csv_bundle(output_directory / safe_name, result)
            else:
                self._export_xlsx_bundle(output_directory / f"{safe_name}.xlsx", result)

    def _export_csv_bundle(
        self,
        directory: Path,
        result: IndexAnalysisResult,
    ) -> None:
        """Export all result frames to individual CSV files."""

        directory.mkdir(parents=True, exist_ok=True)
        frames = self._collect_frames(result)
        for name, frame in frames.items():
            frame.to_csv(directory / f"{name}.csv", index=False)

    def _export_xlsx_bundle(
        self,
        file_path: Path,
        result: IndexAnalysisResult,
    ) -> None:
        """Export all result frames into a single workbook."""

        frames = self._collect_frames(result)
        with pd.ExcelWriter(file_path) as writer:
            for name, frame in frames.items():
                frame.to_excel(writer, sheet_name=name[:31], index=False)

    def _collect_frames(
        self,
        result: IndexAnalysisResult,
    ) -> dict[str, pd.DataFrame]:
        """Collect all dataframes that should be exported."""

        frames: dict[str, pd.DataFrame] = {}
        if self.config.exports.include_input_data:
            frames["input_data"] = result.input_data

        frames["dca"] = result.dca.frame
        if result.btd_original is not None:
            frames["btd_original"] = result.btd_original.frame
        if result.btd_periodic is not None:
            frames["btd_periodic"] = result.btd_periodic.frame
        if result.original_vs_dca is not None:
            frames["original_vs_dca"] = result.original_vs_dca.frame
        if result.periodic_vs_dca is not None:
            frames["periodic_vs_dca"] = result.periodic_vs_dca.frame
        if result.simulation is not None and not result.simulation.raw_results.empty:
            frames["simulation_raw"] = result.simulation.raw_results
            frames["simulation_aggregated"] = result.simulation.aggregated_results
        return frames


class AnalysisPipeline:
    """End-to-end multi-index analysis pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.index_analyzer = IndexAnalyzer(config)
        self.exporter = ResultExporter(config)

    def run(
        self,
        loaded_sources: Iterable[LoadedIndexData],
    ) -> list[IndexAnalysisResult]:
        """Run the full pipeline for all configured sources."""

        results = [self.index_analyzer.analyze(loaded_data) for loaded_data in loaded_sources]
        self.exporter.export_all(results)
        return results


def _slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""

    safe_text = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    )
    return "_".join(part for part in safe_text.split("_") if part)


def _annualize_return(total_return: float, years: float) -> float:
    """Annualize a total return over a given number of years."""

    if total_return <= -1.0:
        return float("nan")
    return (1.0 + total_return) ** (1.0 / years) - 1.0
