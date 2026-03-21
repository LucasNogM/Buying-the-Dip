"""Cash allocation and portfolio backtesting logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from buying_the_dip.models import StrategyOutput, StrategySignal
from buying_the_dip.scheduling import ScheduleBuilder


class CashAllocator:
    """Allocate cash according to the configured contribution mode."""

    def __init__(
        self,
        mode: str,
        contribution_amount: float,
        cash_frequency: str = "monthly",
    ) -> None:
        self.mode = mode
        self.contribution_amount = float(contribution_amount)
        self.cash_frequency = cash_frequency
        self.schedule_builder = ScheduleBuilder()

    def allocate(
        self,
        data: pd.DataFrame,
        investment_signal: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cash inflows, cash balance, and invested contributions."""

        if self.mode == "periodic":
            cash_inflow = investment_signal.astype(float) * self.contribution_amount
            cash_balance = cash_inflow.copy()
            contribution = cash_balance * investment_signal.astype(float)
            return cash_inflow, cash_balance, contribution

        cash_inflow = self._linear_cash_inflow(data)
        reset_groups = self.schedule_builder.build_post_investment_reset_groups(
            investment_signal
        )
        cash_balance = (
            pd.Series(cash_inflow)
            .groupby(reset_groups, sort=False)
            .cumsum()
            .to_numpy(dtype=float)
        )
        contribution = cash_balance * investment_signal.astype(float)
        return cash_inflow, cash_balance, contribution

    def _linear_cash_inflow(self, data: pd.DataFrame) -> np.ndarray:
        """Return the per-row cash accrual for linear contribution mode."""

        accrual_labels = self.schedule_builder.build_window_labels(
            data,
            self.cash_frequency,
            anchor="start",
        )
        group_sizes = pd.Series(accrual_labels).groupby(accrual_labels).transform("size")
        return (self.contribution_amount / group_sizes).to_numpy(dtype=float)


class PortfolioEngine:
    """Backtest a strategy signal over a market index time series."""

    def __init__(
        self,
        cash_mode: str,
        contribution_amount: float,
        cash_frequency: str = "monthly",
    ) -> None:
        self.cash_allocator = CashAllocator(
            mode=cash_mode,
            contribution_amount=contribution_amount,
            cash_frequency=cash_frequency,
        )

    def run(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        signal: StrategySignal,
    ) -> StrategyOutput:
        """Materialize a strategy run into a backtest output."""

        investment_signal = signal.investment_signal.astype(bool)
        cash_inflow, cash_balance, contribution = self.cash_allocator.allocate(
            data=data,
            investment_signal=investment_signal,
        )
        invested_wealth = self._compute_invested_wealth(
            return_factor=data["return_factor"].to_numpy(dtype=float),
            contribution=contribution,
        )
        visible_wealth = invested_wealth + np.where(
            investment_signal,
            0.0,
            cash_balance,
        )
        total_contribution = np.cumsum(contribution, dtype=float)
        denominator = total_contribution + cash_balance - contribution
        wealth_return = np.zeros(len(data), dtype=float)
        valid_mask = denominator > 0
        wealth_return[valid_mask] = visible_wealth[valid_mask] / denominator[valid_mask] - 1.0

        frame = data.copy()
        frame["investment_signal"] = investment_signal
        frame["cash_inflow"] = cash_inflow
        frame["cash_balance"] = cash_balance
        frame["contribution"] = contribution
        frame["invested_wealth"] = invested_wealth
        frame["wealth"] = visible_wealth
        frame["total_contribution"] = total_contribution
        frame["wealth_return"] = wealth_return

        for column_name, values in signal.metadata.items():
            frame[column_name] = values

        return StrategyOutput(
            name=strategy_name,
            frame=frame,
            final_wealth=float(visible_wealth[-1]),
            final_invested_wealth=float(invested_wealth[-1]),
            final_return=float(wealth_return[-1]),
            total_contribution=float(total_contribution[-1]),
            contribution_count=int(investment_signal.sum()),
        )

    @staticmethod
    def _compute_invested_wealth(
        return_factor: np.ndarray,
        contribution: np.ndarray,
    ) -> np.ndarray:
        """Compute the wealth invested in the market using a vectorized formula."""

        if contribution.size == 0:
            return np.array([], dtype=float)

        growth_curve = np.cumprod(return_factor, dtype=float)
        scaled_contribution = np.divide(
            contribution,
            growth_curve,
            out=np.zeros_like(contribution, dtype=float),
            where=growth_curve != 0,
        )
        return growth_curve * np.cumsum(scaled_contribution, dtype=float)
