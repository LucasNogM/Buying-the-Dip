"""Scheduling helpers for investment signals and grouping windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Anchor = Literal["start", "end"]


@dataclass(slots=True)
class ScheduleBuilder:
    """Build investment schedules and window labels."""

    def build_investment_signal(
        self,
        data: pd.DataFrame,
        period: str | int,
        anchor: Anchor,
    ) -> np.ndarray:
        """Return a boolean investment signal for the requested schedule."""

        if isinstance(period, int):
            positions = self._integer_schedule_positions(data, period, anchor)
            signal = np.zeros(len(data), dtype=bool)
            signal[positions] = True
            return signal

        positions = self._calendar_schedule_positions(data, period, anchor)
        signal = np.zeros(len(data), dtype=bool)
        signal[positions] = True
        return signal

    def build_window_labels(
        self,
        data: pd.DataFrame,
        period: str | int,
        anchor: Anchor,
    ) -> np.ndarray:
        """Return a window label per row for the requested period definition."""

        row_positions = np.arange(len(data))

        if isinstance(period, int):
            if anchor == "end":
                end_positions = self._integer_schedule_positions(data, period, anchor)
                return np.searchsorted(end_positions, row_positions, side="left")

            start_positions = self._integer_schedule_positions(data, period, anchor)
            return np.searchsorted(start_positions, row_positions, side="right") - 1

        return self._calendar_window_labels(data, period)

    @staticmethod
    def build_post_investment_reset_groups(investment_signal: np.ndarray) -> np.ndarray:
        """Return group labels that reset immediately after each investment day."""

        if investment_signal.size == 0:
            return np.array([], dtype=int)

        groups = np.empty(investment_signal.size, dtype=int)
        groups[0] = 0
        if investment_signal.size > 1:
            groups[1:] = np.cumsum(investment_signal[:-1], dtype=int)
        return groups

    def _integer_schedule_positions(
        self,
        data: pd.DataFrame,
        period_days: int,
        anchor: Anchor,
    ) -> np.ndarray:
        """Map an integer-day schedule into row positions."""

        dates = data["date"].to_numpy(dtype="datetime64[ns]")
        first_date = pd.Timestamp(data["date"].iloc[0])
        last_date = pd.Timestamp(data["date"].iloc[-1])

        schedule_start = first_date
        if anchor == "end":
            schedule_start = first_date + pd.Timedelta(days=period_days - 1)

        schedule_dates = pd.date_range(
            start=schedule_start,
            end=last_date,
            freq=f"{period_days}D",
        )
        scheduled_values = schedule_dates.to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(dates, scheduled_values, side="left")
        positions = positions[positions < len(dates)]
        return np.unique(positions)

    def _calendar_schedule_positions(
        self,
        data: pd.DataFrame,
        period: str,
        anchor: Anchor,
    ) -> np.ndarray:
        """Resolve calendar-based schedule positions."""

        group_key = self._calendar_group_key(data, period)
        grouped = data.groupby(group_key, sort=False)
        selection = grouped.head(1) if anchor == "start" else grouped.tail(1)
        return selection.index.to_numpy(dtype=int)

    def _calendar_window_labels(
        self,
        data: pd.DataFrame,
        period: str,
    ) -> np.ndarray:
        """Return one label per calendar group."""

        group_key = self._calendar_group_key(data, period)
        return pd.factorize(group_key, sort=False)[0]

    @staticmethod
    def _calendar_group_key(data: pd.DataFrame, period: str) -> pd.Series:
        """Return the grouping key for a calendar frequency."""

        key_map = {
            "daily": "date",
            "weekly": "week_key",
            "monthly": "month_key",
            "quarterly": "quarter_key",
        }
        key_column = key_map[period]
        return data[key_column]
