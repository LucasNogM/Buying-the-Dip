"""Shared dataclasses for loaded data, strategy outputs, and analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class StrategySignal:
    """Investment signal and metadata produced by a strategy."""

    investment_signal: np.ndarray
    metadata: dict[str, np.ndarray | pd.Series] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedIndexData:
    """Materialized input data for one configured index source."""

    index_name: str
    frame: pd.DataFrame
    source_path: str
    source_label: str
    symbol: str | None = None
    analysis_start_date: pd.Timestamp | None = None
    simulation_start_date: pd.Timestamp | None = None


@dataclass(slots=True)
class StrategyOutput:
    """Materialized backtest output for a strategy."""

    name: str
    frame: pd.DataFrame
    final_wealth: float
    final_invested_wealth: float
    final_return: float
    total_contribution: float
    contribution_count: int


@dataclass(slots=True)
class ComparisonResult:
    """Comparison statistics between two strategy runs."""

    label: str
    strategy_a_name: str
    strategy_b_name: str
    frame: pd.DataFrame
    final_relative_advantage: float
    mean_relative_advantage: float
    median_relative_advantage: float
    std_relative_advantage: float
    percent_time_strategy_a_better: float
    annual_return_strategy_a: float
    annual_return_strategy_b: float
    annual_alpha: float


@dataclass(slots=True)
class SimulationOutput:
    """Raw and aggregated simulation outputs."""

    raw_results: pd.DataFrame
    aggregated_results: pd.DataFrame


@dataclass(slots=True)
class IndexAnalysisResult:
    """All outputs generated for a single market index."""

    index_name: str
    source_path: str
    source_label: str
    symbol: str | None
    input_data: pd.DataFrame
    dca: StrategyOutput
    btd_original: StrategyOutput | None
    btd_periodic: StrategyOutput | None
    original_vs_dca: ComparisonResult | None
    periodic_vs_dca: ComparisonResult | None
    simulation: SimulationOutput | None
