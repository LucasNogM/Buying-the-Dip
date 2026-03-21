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
    """Original buying-the-dip strategy based on all-time highs."""

    def __init__(self) -> None:
        super().__init__(name="BtD Original")

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


class BuyTheDipPeriodicStrategy(BaseStrategy):
    """Periodic buying-the-dip strategy based on fixed day windows."""

    def __init__(self, period_days: int, anchor: str) -> None:
        super().__init__(name="BtD Periodic")
        self.period_days = period_days
        self.anchor = anchor

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """Generate the periodic buy-the-dip signal."""

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
