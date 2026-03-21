"""Investment strategy definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from buying_the_dip.models import StrategySignal
from buying_the_dip.scheduling import ScheduleBuilder


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.schedule_builder = ScheduleBuilder()

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate the investment signal for the strategy."""


class DCAStrategy(BaseStrategy):
    """Dollar-cost averaging strategy."""

    def __init__(self, period: str | int, anchor: str) -> None:
        super().__init__(name="DCA")
        self.period = period
        self.anchor = anchor

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate the DCA investment signal."""

        investment_signal = self.schedule_builder.build_investment_signal(
            data=data,
            period=self.period,
            anchor=self.anchor,
        )
        return StrategySignal(investment_signal=investment_signal)


class BuyTheDipOriginalStrategy(BaseStrategy):
    """Theoretical original BtD strategy based on ex-post segment minima."""

    def __init__(self) -> None:
        super().__init__(name="BtD Original (Theoretical)")

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate the original buy-the-dip signal."""

        price = data["price"]
        all_time_high = price.cummax()
        previous_high = all_time_high.shift(1)
        all_time_high_flag = price.gt(previous_high)
        all_time_high_flag.iloc[0] = True

        segment_label = np.cumsum(all_time_high_flag.to_numpy(dtype=bool), dtype=int) - 1
        dip_positions = data.groupby(segment_label, sort=False)["price"].idxmin()

        investment_signal = np.zeros(len(data), dtype=bool)
        investment_signal[dip_positions.to_numpy(dtype=int)] = True
        investment_signal &= ~all_time_high_flag.to_numpy(dtype=bool)

        metadata = {
            "all_time_high": all_time_high.to_numpy(dtype=float),
            "all_time_high_flag": all_time_high_flag.to_numpy(dtype=bool),
        }
        return StrategySignal(
            investment_signal=investment_signal,
            metadata=metadata,
        )


class BuyTheDipOriginalImplementableStrategy(BaseStrategy):
    """Implementable original BtD strategy using an ATH drawdown trigger."""

    def __init__(self, drawdown_threshold_pct: float) -> None:
        super().__init__(name="BtD Original (Implementable)")
        self.drawdown_threshold_pct = float(drawdown_threshold_pct)

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate a live-executable original BtD signal."""

        price = data["price"].to_numpy(dtype=float)
        price_series = data["price"]
        all_time_high = price_series.cummax()
        previous_high = all_time_high.shift(1)
        all_time_high_flag = price_series.gt(previous_high)
        all_time_high_flag.iloc[0] = True

        all_time_high_array = all_time_high.to_numpy(dtype=float)
        all_time_high_flag_array = all_time_high_flag.to_numpy(dtype=bool)
        segment_label = np.cumsum(all_time_high_flag_array, dtype=int) - 1

        investment_signal = np.zeros(len(data), dtype=bool)
        segment_reference_high = np.full(len(data), np.nan, dtype=float)
        drawdown_from_reference_high = np.full(len(data), np.nan, dtype=float)
        threshold_hit = np.zeros(len(data), dtype=bool)

        unique_segments = np.unique(segment_label)
        for segment in unique_segments:
            positions = np.flatnonzero(segment_label == segment)
            if positions.size == 0:
                continue

            reference_high = float(price[positions[0]])
            segment_reference_high[positions] = reference_high
            segment_drawdown = price[positions] / reference_high - 1.0
            drawdown_from_reference_high[positions] = segment_drawdown

            candidate_positions = positions[
                (segment_drawdown <= -self.drawdown_threshold_pct)
                & (~all_time_high_flag_array[positions])
            ]
            if candidate_positions.size == 0:
                continue

            chosen_position = int(candidate_positions[0])
            investment_signal[chosen_position] = True
            threshold_hit[chosen_position] = True

        metadata = {
            "all_time_high": all_time_high_array,
            "all_time_high_flag": all_time_high_flag_array,
            "segment_reference_high": segment_reference_high,
            "drawdown_from_reference_high": drawdown_from_reference_high,
            "threshold_hit": threshold_hit,
        }
        return StrategySignal(
            investment_signal=investment_signal,
            metadata=metadata,
        )


class BuyTheDipPeriodicStrategy(BaseStrategy):
    """Theoretical periodic BtD strategy based on ex-post window minima."""

    def __init__(self, period_days: int, anchor: str) -> None:
        super().__init__(name="BtD Periodic (Theoretical)")
        self.period_days = period_days
        self.anchor = anchor

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate the theoretical periodic buy-the-dip signal."""

        window_label = self.schedule_builder.build_window_labels(
            data=data,
            period=self.period_days,
            anchor=self.anchor,
        )
        dip_positions = data.groupby(window_label, sort=False)["price"].idxmin()

        investment_signal = np.zeros(len(data), dtype=bool)
        investment_signal[dip_positions.to_numpy(dtype=int)] = True

        metadata = {"dip_window": window_label}
        return StrategySignal(
            investment_signal=investment_signal,
            metadata=metadata,
        )


class BuyTheDipPeriodicImplementableStrategy(BaseStrategy):
    """Implementable periodic BtD using an in-window drawdown trigger."""

    def __init__(
        self,
        period_days: int,
        anchor: str,
        drawdown_threshold_pct: float,
        fallback_to_window_end: bool,
    ) -> None:
        super().__init__(name="BtD Periodic (Implementable)")
        self.period_days = period_days
        self.anchor = anchor
        self.drawdown_threshold_pct = float(drawdown_threshold_pct)
        self.fallback_to_window_end = bool(fallback_to_window_end)

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate a live-executable periodic BtD signal."""

        window_label = self.schedule_builder.build_window_labels(
            data=data,
            period=self.period_days,
            anchor=self.anchor,
        )
        price = data["price"].to_numpy(dtype=float)

        investment_signal = np.zeros(len(data), dtype=bool)
        running_window_peak = np.full(len(data), np.nan, dtype=float)
        drawdown_from_window_peak = np.full(len(data), np.nan, dtype=float)
        threshold_hit = np.zeros(len(data), dtype=bool)
        fallback_execution = np.zeros(len(data), dtype=bool)

        for window in np.unique(window_label):
            positions = np.flatnonzero(window_label == window)
            if positions.size == 0:
                continue

            segment_prices = price[positions]
            segment_running_peak = np.maximum.accumulate(segment_prices)
            segment_drawdown = segment_prices / segment_running_peak - 1.0

            running_window_peak[positions] = segment_running_peak
            drawdown_from_window_peak[positions] = segment_drawdown

            candidate_positions = positions[
                segment_drawdown <= -self.drawdown_threshold_pct
            ]
            if candidate_positions.size > 0:
                chosen_position = int(candidate_positions[0])
                investment_signal[chosen_position] = True
                threshold_hit[chosen_position] = True
                continue

            if self.fallback_to_window_end:
                chosen_position = int(positions[-1])
                investment_signal[chosen_position] = True
                fallback_execution[chosen_position] = True

        metadata = {
            "dip_window": window_label,
            "running_window_peak": running_window_peak,
            "drawdown_from_window_peak": drawdown_from_window_peak,
            "threshold_hit": threshold_hit,
            "fallback_execution": fallback_execution,
        }
        return StrategySignal(
            investment_signal=investment_signal,
            metadata=metadata,
        )
