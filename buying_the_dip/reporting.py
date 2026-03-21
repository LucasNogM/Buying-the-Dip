"""HTML report generation for multi-index analyses."""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from buying_the_dip.config import AppConfig
from buying_the_dip.models import (
    ComparisonResult,
    IndexAnalysisResult,
    SimulationOutput,
    StrategyOutput,
)


class HtmlReportBuilder:
    """Build a report hub plus one dedicated HTML page per index."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.figure_facecolor = "#020617"
        self.axes_facecolor = "#0f172a"
        self.text_color = "#e2e8f0"
        self.muted_text_color = "#94a3b8"
        self.border_color = "#334155"
        self.grid_color = "#475569"

    def build(self, results: list[IndexAnalysisResult]) -> Path:
        """Generate the main report page and one HTML file per index."""
        output_directory = Path(self.config.report.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        generated_at = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        page_names = {
            result.index_name: f"{self._anchor_id(result.index_name)}.html"
            for result in results
        }

        home_file_name = self.config.report.index_file_name
        for result in results:
            page_html = self._build_index_page(result, generated_at, home_file_name)
            (output_directory / page_names[result.index_name]).write_text(
                page_html,
                encoding="utf-8",
            )

        home_html = self._build_main_page(results, page_names, generated_at)
        home_path = output_directory / home_file_name
        home_path.write_text(home_html, encoding="utf-8")
        return home_path

    def _build_main_page(
        self,
        results: list[IndexAnalysisResult],
        page_names: dict[str, str],
        generated_at: str,
    ) -> str:
        """Render the main hub page with navigation, comparison, and methodology."""
        overview_cards = self._build_overview_cards(results)
        overview_chart_html = ""
        if self.config.report.include_overview_chart:
            overview_chart = self._plot_overview_indices(results)
            overview_chart_html = (
                '<div class="figure-grid">'
                + self._figure_card(
                    "Normalized price path by index",
                    overview_chart,
                    wide=True,
                )
                + '</div>'
            )

        index_reports_table = self._frame_to_html(
            self._build_index_reports_table(results, page_names),
            escape=False,
        )
        comparison_table = self._frame_to_html(
            self._build_main_comparison_table(results),
            escape=False,
            classes=["table", "comparison-table"],
        )
        methodology_html = self._build_methodology_section()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self._project_title())}</title>
    <style>
        {self._build_css()}
    </style>
</head>
<body>
    <div class="container">
        <section class="panel hero">
            <h1>{html.escape(self._project_title())}</h1>
            <p class="muted">Generated at {generated_at}.</p>
            <div class="metrics-grid">{overview_cards}</div>
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>Index reports</h2>
            </div>
            <p class="muted">
                Quick navigation table with one link per index and a minimal index summary.
            </p>
            {index_reports_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>General comparison</h2>
            </div>
            <p class="muted">
                Side-by-side annualized-return comparison for DCA, original BtD, and periodic BtD.
                The final row aggregates only alpha values across indices so the two BtD variants
                can be compared directly at the portfolio-selection level.
            </p>
            {overview_chart_html}
            {comparison_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>Methodology</h2>
            </div>
            {methodology_html}
        </section>
    </div>
    {self._build_modal_markup()}
    <script>
        {self._build_javascript()}
    </script>
</body>
</html>
"""

    def _build_index_page(
        self,
        result: IndexAnalysisResult,
        generated_at: str,
        home_file_name: str,
    ) -> str:
        """Render one standalone HTML page for a single index."""
        data_start = result.input_data["date"].iloc[0].strftime("%Y-%m-%d")
        data_end = result.input_data["date"].iloc[-1].strftime("%Y-%m-%d")
        cards = self._build_metric_cards(result)
        strategy_table = self._frame_to_html(self._build_strategy_summary(result))
        comparison_table = self._frame_to_html(self._build_comparison_summary(result))
        figure_html = self._build_strategy_figures(result)
        simulation_html = self._build_simulation_section(result)
        symbol_html = html.escape(result.symbol) if result.symbol else "N/A"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(result.index_name)} - {html.escape(self._project_title())}</title>
    <style>
        {self._build_css()}
    </style>
</head>
<body>
    <div class="container">
        <section class="panel hero">
            <a class="button-link" href="{html.escape(home_file_name)}">&larr; Back to main report</a>
            <h1>{html.escape(result.index_name)}</h1>
            <p class="muted">Generated at {generated_at}.</p>
            <p class="muted">
                Data source: {html.escape(self._display_source_label(result.source_label))}<br>
                Symbol: {symbol_html}<br>
                Date range: {data_start} to {data_end}
            </p>
            <div class="metrics-grid">{cards}</div>
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>Strategy comparison</h2>
            </div>
            <p class="muted">
                Comparative view of the three approaches, including annualized return.
            </p>
            {strategy_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>BtD alpha versus DCA</h2>
            </div>
            <p class="muted">
                Alpha is reported for both BtD variants relative to DCA.
            </p>
            {comparison_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>Main charts</h2>
            </div>
            <p class="muted">Click any chart to expand it.</p>
            <div class="figure-grid">{figure_html}</div>
        </section>

        {simulation_html}

        <section class="panel footer-panel">
            <a class="button-link" href="{html.escape(home_file_name)}">&larr; Back to main report</a>
        </section>
    </div>
    {self._build_modal_markup()}
    <script>
        {self._build_javascript()}
    </script>
</body>
</html>
"""

    def _project_title(self) -> str:
        """Return a cleaned project title for report presentation."""
        title = self.config.project.title.replace(" (Offline Validation)", "").strip()
        return title or self.config.project.title.strip()

    def _build_overview_cards(self, results: list[IndexAnalysisResult]) -> str:
        """Build high-level comparative cards for the main page."""
        dca_returns = [
            self._annualized_return(result.dca.final_return, result.input_data)
            for result in results
        ]
        original_returns = [
            self._annualized_return(result.btd_original.final_return, result.input_data)
            for result in results
            if result.btd_original is not None
        ]
        periodic_returns = [
            self._annualized_return(result.btd_periodic.final_return, result.input_data)
            for result in results
            if result.btd_periodic is not None
        ]
        cards = [
            ("Indices", str(len(results))),
            (
                "Average DCA annualized return",
                self._format_percent(float(np.nanmean(dca_returns))),
            ),
            (
                "Average original BtD annualized return",
                (
                    self._format_percent(float(np.nanmean(original_returns)))
                    if original_returns
                    else "N/A"
                ),
            ),
            (
                "Average periodic BtD annualized return",
                (
                    self._format_percent(float(np.nanmean(periodic_returns)))
                    if periodic_returns
                    else "N/A"
                ),
            ),
        ]
        return "".join(
            f'<div class="metric-card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div></div>'
            for label, value in cards
        )

    def _build_index_reports_table(
        self,
        results: list[IndexAnalysisResult],
        page_names: dict[str, str],
    ) -> pd.DataFrame:
        """Build the navigation table for the main page."""
        rows: list[dict[str, object]] = []
        for result in results:
            latest_row = result.input_data.iloc[-1]
            rows.append(
                {
                    "Index": result.index_name,
                    "Symbol": result.symbol or "N/A",
                    "Earliest date": result.input_data["date"].iloc[0].strftime("%Y-%m-%d"),
                    "Latest date": latest_row["date"].strftime("%Y-%m-%d"),
                    "Latest price": self._format_number(float(latest_row["price"])),
                    "Index report": (
                        f'<a class="inline-link" '
                        f'href="{html.escape(page_names[result.index_name])}">'
                        "Open"
                        "</a>"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_main_comparison_table(
        self,
        results: list[IndexAnalysisResult],
    ) -> pd.DataFrame:
        """Build the cross-index strategy comparison table for the hub page."""
        rows: list[dict[str, object]] = []
        original_alphas: list[float] = []
        periodic_alphas: list[float] = []

        for result in results:
            original_alpha = (
                result.original_vs_dca.annual_alpha
                if result.original_vs_dca is not None
                else None
            )
            periodic_alpha = (
                result.periodic_vs_dca.annual_alpha
                if result.periodic_vs_dca is not None
                else None
            )
            if original_alpha is not None:
                original_alphas.append(float(original_alpha))
            if periodic_alpha is not None:
                periodic_alphas.append(float(periodic_alpha))

            rows.append(
                {
                    "Index": result.index_name,
                    "DCA annualized return": self._format_percent(
                        self._annualized_return(result.dca.final_return, result.input_data)
                    ),
                    "Original BtD annualized return": self._format_percent_or_na(
                        self._strategy_annualized_return(result, "original")
                    ),
                    "Original BtD alpha vs DCA": self._format_percent_or_na(
                        original_alpha
                    ),
                    "Periodic BtD annualized return": self._format_percent_or_na(
                        self._strategy_annualized_return(result, "periodic")
                    ),
                    "Periodic BtD alpha vs DCA": self._format_percent_or_na(
                        periodic_alpha
                    ),
                }
            )

        rows.append(
            {
                "Index": "<strong>Cross-index mean alpha</strong>",
                "DCA annualized return": "&mdash;",
                "Original BtD annualized return": "&mdash;",
                "Original BtD alpha vs DCA": (
                    f"<strong>{self._format_percent(float(np.nanmean(original_alphas)))}</strong>"
                    if original_alphas
                    else "<strong>N/A</strong>"
                ),
                "Periodic BtD annualized return": "&mdash;",
                "Periodic BtD alpha vs DCA": (
                    f"<strong>{self._format_percent(float(np.nanmean(periodic_alphas)))}</strong>"
                    if periodic_alphas
                    else "<strong>N/A</strong>"
                ),
            }
        )
        return pd.DataFrame(rows)

    def _build_methodology_section(self) -> str:
        """Render a direct explanation of the analytical methodology."""
        dca_period = self.config.strategies.dca.period
        dca_anchor = self.config.strategies.dca.anchor
        periodic_window = self.config.strategies.btd_periodic.period_days
        simulation_intervals = ", ".join(
            str(value) for value in self.config.simulation.contribution_intervals_days
        )
        simulation_horizons = ", ".join(
            str(value) for value in self.config.simulation.horizons_years
        )
        return f"""
<div class="methodology-grid">
    <article class="method-card">
        <h3>1. Data and preprocessing</h3>
        <p>
            Each index is downloaded with the API-first loader using the maximum
            available daily history. When Yahoo Finance provides adjusted close,
            that series is preferred; otherwise the regular close is used. Dates
            are normalized to one row per day, duplicates are removed, and only
            positive prices are kept. The downloaded history is then saved to CSV
            so it can be reused as a local fallback in a later run.
        </p>
    </article>

    <article class="method-card">
        <h3>2. Main strategies</h3>
        <p>
            <strong>DCA</strong> contributes on a fixed schedule defined in the
            configuration. In the current setup, the main backtest uses
            <strong>{html.escape(str(dca_period))}</strong> contributions anchored at
            <strong>{html.escape(str(dca_anchor))}</strong>.
        </p>
        <p>
            <strong>Original BtD</strong> splits the series into segments delimited by
            new all-time highs and allocates at the lowest close inside each completed
            segment.
        </p>
        <p>
            <strong>Periodic BtD</strong> splits time into fixed windows and allocates at
            the lowest close inside each window. In the main backtest, the default
            window is <strong>{periodic_window} days</strong>.
        </p>
        <div class="method-example">
            Example: if a 30-day window has closes 100, 98, 95, 97, and 99,
            periodic BtD allocates on the 95 close, while DCA allocates on its
            scheduled day regardless of where the minimum occurred.
        </div>
    </article>

    <article class="method-card">
        <h3>3. Wealth, return, and annualization</h3>
        <p>
            Portfolio wealth is the sum of the invested component that compounds with
            the index and the idle cash that has not yet been deployed. Final return is
            the visible portfolio wealth divided by the capital made available over time,
            minus one.
        </p>
        <div class="formula-box">
            Annualized return = (1 + total return)^(1 / years) - 1
        </div>
        <p>
            This makes all three methods directly comparable even when the full history
            spans many decades.
        </p>
    </article>

    <article class="method-card">
        <h3>4. Alpha and relative advantage</h3>
        <p>
            Alpha is measured against DCA, which is the baseline strategy.
            Positive alpha means the BtD approach delivered a higher annualized return
            than DCA for the same index and full sample.
        </p>
        <div class="formula-box">
            Alpha = annualized return of BtD - annualized return of DCA
        </div>
        <p>
            The relative-advantage chart uses the running ratio of portfolio wealth:
            wealth of BtD divided by wealth of DCA, minus one.
        </p>
    </article>

    <article class="method-card">
        <h3>5. Simulation grid</h3>
        <p>
            The simulation section compares <strong>periodic BtD</strong> and
            <strong>periodic DCA</strong> across multiple contribution intervals and
            investment horizons. In this configuration, the tested intervals are
            <strong>{html.escape(simulation_intervals)} days</strong> and the tested
            horizons are <strong>{html.escape(simulation_horizons)} years</strong>.
        </p>
        <p>
            <strong>Period</strong> is the spacing between contributions, expressed in
            days. Example: a 30-day period means new capital is added every 30 days.
            <strong>Horizon</strong> is the total length of each simulated backtest
            window, expressed in years. Example: a 5-year horizon means the strategy is
            evaluated from a chosen start date through the next five years.
        </p>
        <p>
            For each horizon, the code samples multiple evenly spaced start points over
            the available history, runs both strategies inside each window, and then
            aggregates the resulting annual alpha. The heatmap shows mean annual alpha:
            values above zero favor periodic BtD; values below zero favor periodic DCA.
        </p>
    </article>

    <article class="method-card method-card-warning">
        <h3>6. Important interpretation note</h3>
        <p>
            Both BtD variants are built with <strong>ex-post minima</strong>. In other
            words, the code identifies the lowest close only after the full segment or
            window is known. This makes the BtD results a hindsight benchmark for dip
            timing, not a directly live-executable rule as written.
        </p>
        <p>
            The reports also exclude transaction costs, taxes, slippage, spreads, and
            execution constraints. For that reason, the BtD results should be read as a
            clean analytical comparison, not as a complete implementation-ready trading
            system.
        </p>
    </article>
</div>
"""

    def _build_index_summary_card(self, result: IndexAnalysisResult, href: str) -> str:
        """Build one summary card with three-strategy comparisons."""
        return f"""
<article class="summary-card">
    <h3>{html.escape(result.index_name)}</h3>
    <p><strong>DCA annualized return:</strong> {html.escape(self._format_percent(self._annualized_return(result.dca.final_return, result.input_data)))}</p>
    <p><strong>Original BtD annualized return:</strong> {html.escape(self._format_percent_or_na(self._strategy_annualized_return(result, 'original')))}</p>
    <p><strong>Original BtD alpha vs DCA:</strong> {html.escape(self._format_percent_or_na(result.original_vs_dca.annual_alpha if result.original_vs_dca is not None else None))}</p>
    <p><strong>Periodic BtD annualized return:</strong> {html.escape(self._format_percent_or_na(self._strategy_annualized_return(result, 'periodic')))}</p>
    <p><strong>Periodic BtD alpha vs DCA:</strong> {html.escape(self._format_percent_or_na(result.periodic_vs_dca.annual_alpha if result.periodic_vs_dca is not None else None))}</p>
    <a class="button-link" href="{html.escape(href)}">Open report</a>
</article>
"""

    def _build_metric_cards(self, result: IndexAnalysisResult) -> str:
        """Build the comparative card grid for one index page."""
        cards = [
            (
                "DCA annualized return",
                self._format_percent(
                    self._annualized_return(result.dca.final_return, result.input_data)
                ),
            ),
            (
                "Original BtD annualized return",
                self._format_percent_or_na(
                    self._strategy_annualized_return(result, "original")
                ),
            ),
            (
                "Original BtD alpha vs DCA",
                self._format_percent_or_na(
                    result.original_vs_dca.annual_alpha
                    if result.original_vs_dca is not None
                    else None
                ),
            ),
            (
                "Periodic BtD annualized return",
                self._format_percent_or_na(
                    self._strategy_annualized_return(result, "periodic")
                ),
            ),
            (
                "Periodic BtD alpha vs DCA",
                self._format_percent_or_na(
                    result.periodic_vs_dca.annual_alpha
                    if result.periodic_vs_dca is not None
                    else None
                ),
            ),
        ]
        return "".join(
            f'<div class="metric-card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div></div>'
            for label, value in cards
        )

    def _build_strategy_summary(self, result: IndexAnalysisResult) -> pd.DataFrame:
        """Return the comparative strategy table for one index."""
        rows = [self._strategy_summary_row(result.dca, result.input_data)]
        if result.btd_original is not None:
            rows.append(self._strategy_summary_row(result.btd_original, result.input_data))
        if result.btd_periodic is not None:
            rows.append(self._strategy_summary_row(result.btd_periodic, result.input_data))
        return pd.DataFrame(rows)

    def _build_comparison_summary(self, result: IndexAnalysisResult) -> pd.DataFrame:
        """Return the BtD-versus-DCA alpha table for one index."""
        rows: list[dict[str, str]] = []
        if result.original_vs_dca is not None:
            rows.append(self._comparison_summary_row(result.original_vs_dca))
        if result.periodic_vs_dca is not None:
            rows.append(self._comparison_summary_row(result.periodic_vs_dca))
        if not rows:
            rows.append({"Comparison": "No comparison enabled."})
        return pd.DataFrame(rows)

    def _build_simulation_section(self, result: IndexAnalysisResult) -> str:
        """Build the optional simulation section for one index page."""
        if (
            not self.config.report.include_simulation
            or result.simulation is None
            or result.simulation.aggregated_results.empty
        ):
            return ""

        simulation_table = self._frame_to_html(self._build_simulation_table(result))
        figures = "\n".join(
            [
                self._figure_card(
                    "Mean annual alpha by interval and horizon",
                    self._plot_simulation_heatmap(result.simulation),
                    wide=True,
                ),
                self._figure_card(
                    "Mean annual alpha as the horizon increases",
                    self._plot_simulation_lines(result.simulation),
                    wide=True,
                ),
            ]
        )
        return f"""
<section class="panel">
    <div class="section-title-row">
        <h2>Simulation</h2>
    </div>
    <p class="muted">
        Periodic DCA and periodic BtD are simulated over the configured horizons and intervals.
    </p>
    <div class="figure-grid">{figures}</div>
    {simulation_table}
</section>
"""

    def _build_simulation_table(self, result: IndexAnalysisResult) -> pd.DataFrame:
        """Return the top simulation combinations for one index."""
        simulation_frame = result.simulation.aggregated_results.copy()
        sort_column = (
            "annual_alpha_mean"
            if "annual_alpha_mean" in simulation_frame.columns
            else simulation_frame.columns[-1]
        )
        simulation_frame = simulation_frame.sort_values(
            by=sort_column,
            ascending=False,
        ).head(self.config.report.max_simulation_rows)
        formatted = simulation_frame.copy()
        for column in formatted.columns:
            if column.endswith(("_mean", "_median")) or "alpha" in column:
                formatted[column] = formatted[column].map(self._format_percent)
            elif "advantage" in column or "return" in column:
                formatted[column] = formatted[column].map(self._format_percent)
            elif "horizon" in column or "interval" in column:
                formatted[column] = formatted[column].map(lambda value: f"{int(value)}")
        return formatted

    def _build_strategy_figures(self, result: IndexAnalysisResult) -> str:
        """Build all non-simulation figures for one index page."""
        figures: list[str] = []
        if result.original_vs_dca is not None and self.config.report.include_original_btd:
            figures.extend(
                [
                    self._figure_card(
                        "Price and contributions: DCA vs. original BtD",
                        self._plot_price_and_signals(
                            result.input_data,
                            [result.dca, result.btd_original],
                            include_all_time_high=True,
                        ),
                    ),
                    self._figure_card(
                        "Wealth: DCA vs. original BtD",
                        self._plot_wealth_comparison([result.dca, result.btd_original]),
                    ),
                    self._figure_card(
                        "Relative advantage: original BtD vs. DCA",
                        self._plot_relative_advantage(result.original_vs_dca),
                    ),
                ]
            )
        if result.periodic_vs_dca is not None and self.config.report.include_periodic_btd:
            figures.extend(
                [
                    self._figure_card(
                        "Price and contributions: DCA vs. periodic BtD",
                        self._plot_price_and_signals(
                            result.input_data,
                            [result.dca, result.btd_periodic],
                            include_all_time_high=False,
                        ),
                    ),
                    self._figure_card(
                        "Wealth: DCA vs. periodic BtD",
                        self._plot_wealth_comparison([result.dca, result.btd_periodic]),
                    ),
                    self._figure_card(
                        "Relative advantage: periodic BtD vs. DCA",
                        self._plot_relative_advantage(result.periodic_vs_dca),
                    ),
                ]
            )
        if not figures:
            return "<p>No figures enabled for this index.</p>"
        return "\n".join(figures)

    def _strategy_summary_row(
        self,
        output: StrategyOutput,
        input_data: pd.DataFrame,
    ) -> dict[str, str]:
        """Return one formatted strategy summary row."""
        annualized_return = self._annualized_return(output.final_return, input_data)
        return {
            "Strategy": output.name,
            "Annualized return": self._format_percent(annualized_return),
            "Final wealth": self._format_number(output.final_wealth),
            "Final return": self._format_percent(output.final_return),
            "Final invested wealth": self._format_number(output.final_invested_wealth),
            "Total contribution": self._format_number(output.total_contribution),
            "Contribution count": f"{output.contribution_count:,}",
        }

    def _comparison_summary_row(self, comparison: ComparisonResult) -> dict[str, str]:
        """Return one formatted comparison summary row."""
        return {
            "Comparison": comparison.label,
            f"{comparison.strategy_a_name} annualized return": self._format_percent(
                comparison.annual_return_strategy_a
            ),
            f"{comparison.strategy_b_name} annualized return": self._format_percent(
                comparison.annual_return_strategy_b
            ),
            "Annual alpha": self._format_percent(comparison.annual_alpha),
            "Final relative advantage": self._format_percent(
                comparison.final_relative_advantage
            ),
            "Mean relative advantage": self._format_percent(
                comparison.mean_relative_advantage
            ),
            "Median relative advantage": self._format_percent(
                comparison.median_relative_advantage
            ),
            "Volatility of advantage": self._format_percent(
                comparison.std_relative_advantage
            ),
            "Time BtD is ahead": self._format_percent(
                comparison.percent_time_strategy_a_better
            ),
        }

    def _plot_overview_indices(self, results: list[IndexAnalysisResult]) -> str:
        """Plot all configured indices rebased to a common starting value."""
        figure, axis = self._create_figure(figsize=(12, 4.8))
        for result in results:
            rebased = result.input_data["price"].div(result.input_data["price"].iloc[0])
            axis.plot(
                result.input_data["date"],
                rebased * 100.0,
                linewidth=1.7,
                label=result.index_name,
            )
        self._apply_date_axis(axis)
        axis.set_title("Normalized price path by index")
        axis.set_xlabel("Date")
        axis.set_ylabel("Rebased level (start = 100)")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_large_number))
        self._apply_axis_style(axis)
        self._style_legend(axis, ncols=2)
        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _plot_price_and_signals(
        self,
        data: pd.DataFrame,
        outputs: list[StrategyOutput | None],
        include_all_time_high: bool,
    ) -> str:
        """Plot the price series and strategy contribution markers."""
        valid_outputs = [output for output in outputs if output is not None]
        figure, axis = self._create_figure(figsize=(12, 4.5))
        axis.plot(data["date"], data["price"], label="Price", linewidth=1.4)
        marker_cycle = ["o", "s", "^", "D"]

        for position, output in enumerate(valid_outputs):
            signal_frame = output.frame.loc[output.frame["investment_signal"]]
            axis.scatter(
                signal_frame["date"],
                signal_frame["price"],
                label=output.name,
                marker=marker_cycle[position % len(marker_cycle)],
                s=28,
                alpha=0.9,
            )

        if include_all_time_high and valid_outputs:
            reference_frame = valid_outputs[-1].frame
            if "all_time_high_flag" in reference_frame.columns:
                high_frame = reference_frame.loc[reference_frame["all_time_high_flag"]]
                axis.scatter(
                    high_frame["date"],
                    high_frame["price"],
                    label="All-time high",
                    marker="x",
                    s=24,
                    alpha=0.8,
                )

        axis.set_title("Price with investment signals")
        axis.set_xlabel("Date")
        axis.set_ylabel("Index price")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_large_number))
        self._apply_date_axis(axis)
        self._apply_axis_style(axis)
        self._style_legend(axis)
        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _plot_wealth_comparison(
        self,
        outputs: list[StrategyOutput | None],
    ) -> str:
        """Plot the wealth evolution for multiple strategy outputs."""
        valid_outputs = [output for output in outputs if output is not None]
        figure, axis = self._create_figure(figsize=(12, 4.5))
        for output in valid_outputs:
            axis.plot(output.frame["date"], output.frame["wealth"], label=output.name)

        axis.set_title("Portfolio wealth over time")
        axis.set_xlabel("Date")
        axis.set_ylabel("Wealth")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_large_number))
        self._apply_date_axis(axis)
        self._apply_axis_style(axis)
        self._style_legend(axis)
        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _plot_relative_advantage(self, comparison: ComparisonResult) -> str:
        """Plot the relative advantage of strategy A versus strategy B."""
        values_pct = comparison.frame["relative_advantage"].to_numpy(dtype=float) * 100.0
        figure, axis = self._create_figure(figsize=(12, 4.0))
        axis.plot(comparison.frame["date"], values_pct, linewidth=1.3)
        axis.axhline(0.0, linestyle="--", linewidth=1.0, color=self.muted_text_color)
        clipped = self._apply_advantage_scale(axis, values_pct)
        axis.set_title(comparison.label)
        axis.set_xlabel("Date")
        axis.set_ylabel("Relative advantage")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_percentage_axis))
        self._apply_date_axis(axis)
        self._apply_axis_style(axis)
        if clipped:
            axis.text(
                0.99,
                0.02,
                "Robust y-axis applied",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color=self.text_color,
                fontsize=8,
            )
        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _plot_simulation_heatmap(self, simulation: SimulationOutput) -> str:
        """Plot the mean annual alpha heatmap."""
        aggregated = simulation.aggregated_results
        value_column = (
            "annual_alpha_mean"
            if "annual_alpha_mean" in aggregated.columns
            else aggregated.columns[-1]
        )
        pivot = aggregated.pivot(
            index="contribution_interval_days",
            columns="horizon_years",
            values=value_column,
        )
        figure, axis = self._create_figure(figsize=(10, 5))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        axis.set_title("Mean annual alpha")
        axis.set_xlabel("Investment horizon (years)")
        axis.set_ylabel("Contribution interval (days)")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns.astype(int))
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels(pivot.index.astype(int))
        self._apply_axis_style(axis)
        color_bar = figure.colorbar(image, ax=axis)
        color_bar.ax.yaxis.set_major_formatter(FuncFormatter(_format_percent_axis_decimal))
        color_bar.ax.tick_params(colors=self.muted_text_color)
        color_bar.ax.yaxis.label.set_color(self.text_color)
        color_bar.outline.set_edgecolor(self.border_color)

        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    self._format_percent(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=self.text_color,
                )

        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _plot_simulation_lines(self, simulation: SimulationOutput) -> str:
        """Plot mean annual alpha versus horizon by contribution interval."""
        aggregated = simulation.aggregated_results
        value_column = (
            "annual_alpha_mean"
            if "annual_alpha_mean" in aggregated.columns
            else aggregated.columns[-1]
        )
        figure, axis = self._create_figure(figsize=(12, 4.5))
        for interval_days, frame in aggregated.groupby(
            "contribution_interval_days",
            sort=True,
        ):
            ordered = frame.sort_values("horizon_years")
            axis.plot(
                ordered["horizon_years"],
                ordered[value_column],
                label=f"{int(interval_days)}",
            )
        horizon_values = aggregated["horizon_years"].dropna().astype(int)
        if not horizon_values.empty:
            min_horizon = int(horizon_values.min())
            max_horizon = int(horizon_values.max())
            axis.set_xlim(min_horizon, max_horizon)
            axis.set_xticks(np.arange(min_horizon, max_horizon + 1, 1))
        axis.set_title("Mean annual alpha by horizon")
        axis.set_xlabel("Investment horizon (years)")
        axis.set_ylabel("Mean annual alpha")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_percent_axis_decimal))
        self._apply_axis_style(axis)
        self._style_legend(axis, ncols=2, title="Investment period (days)")
        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _figure_card(self, title: str, image_base64: str, wide: bool = False) -> str:
        """Wrap a base64 image into an HTML figure card."""
        wide_class = " wide-card" if wide else ""
        escaped_title = html.escape(title)
        return f"""
<div class="figure-card{wide_class}">
    <h4>{escaped_title}</h4>
    <img
        class="expandable-image"
        src="data:image/png;base64,{image_base64}"
        alt="{escaped_title}"
        data-title="{escaped_title}"
    >
</div>
"""

    @staticmethod
    def _frame_to_html(
        frame: pd.DataFrame,
        escape: bool = True,
        classes: list[str] | tuple[str, ...] | str = "table",
    ) -> str:
        """Convert a dataframe into a styled HTML table."""
        if frame.empty:
            return "<p>No data available.</p>"
        return frame.to_html(index=False, classes=classes, escape=escape)

    def _figure_to_base64(self, figure: plt.Figure) -> str:
        """Encode a Matplotlib figure into a base64 PNG string."""
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=self.config.report.figure_dpi,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
        plt.close(figure)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _create_figure(self, figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
        """Create a styled Matplotlib figure and axis."""
        figure, axis = plt.subplots(figsize=figsize, facecolor=self.figure_facecolor)
        axis.set_facecolor(self.axes_facecolor)
        return figure, axis

    def _apply_date_axis(self, axis: plt.Axes) -> None:
        """Apply a concise date formatter to time-series charts."""
        locator = mdates.AutoDateLocator()
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def _apply_axis_style(self, axis: plt.Axes) -> None:
        """Apply the shared dark-theme styling to an axis."""
        axis.tick_params(colors=self.muted_text_color)
        axis.xaxis.label.set_color(self.text_color)
        axis.yaxis.label.set_color(self.text_color)
        axis.title.set_color(self.text_color)
        for spine in axis.spines.values():
            spine.set_color(self.border_color)
        axis.grid(True, alpha=0.22, color=self.grid_color)

    def _style_legend(
        self,
        axis: plt.Axes,
        ncols: int = 1,
        title: str | None = None,
    ) -> None:
        """Render a themed legend when the axis has labeled artists."""
        handles, labels = axis.get_legend_handles_labels()
        if not handles:
            return
        legend = axis.legend(
            handles,
            labels,
            loc="best",
            ncols=ncols,
            title=title,
            frameon=True,
            facecolor=self.axes_facecolor,
            edgecolor=self.border_color,
        )
        for text in legend.get_texts():
            text.set_color(self.text_color)
        legend_title = legend.get_title()
        if legend_title is not None:
            legend_title.set_color(self.text_color)

    def _apply_advantage_scale(self, axis: plt.Axes, values_pct: np.ndarray) -> bool:
        """Use a robust y-axis when a few extreme points flatten the series."""
        finite_values = values_pct[np.isfinite(values_pct)]
        if finite_values.size == 0:
            return False

        absolute_values = np.abs(finite_values)
        full_limit = float(np.nanmax(absolute_values))
        robust_limit = float(
            np.nanpercentile(
                absolute_values,
                self.config.report.advantage_clip_percentile,
            )
        )
        middle_limit = float(np.nanpercentile(absolute_values, 90.0))
        robust_limit = max(
            robust_limit * 1.08,
            middle_limit * 1.20,
            self.config.report.advantage_min_limit_pct,
        )

        if full_limit <= 0:
            axis_limit = self.config.report.advantage_min_limit_pct
            axis.set_ylim(-axis_limit, axis_limit)
            return False

        if full_limit > robust_limit * 1.25:
            axis.set_ylim(-robust_limit, robust_limit)
            return True

        axis_limit = max(full_limit * 1.08, self.config.report.advantage_min_limit_pct)
        axis.set_ylim(-axis_limit, axis_limit)
        return False

    def _build_modal_markup(self) -> str:
        """Return the HTML markup used to expand charts on click."""
        return """
<div id="image-modal" class="image-modal" aria-hidden="true">
    <div class="image-modal-backdrop" data-close-modal="true"></div>
    <div class="image-modal-content" role="dialog" aria-modal="true" aria-label="Expanded chart">
        <button type="button" class="image-modal-close" data-close-modal="true">&times;</button>
        <div class="image-modal-title" id="image-modal-title"></div>
        <img id="image-modal-image" class="image-modal-image" alt="Expanded chart">
    </div>
</div>
"""

    @staticmethod
    def _build_javascript() -> str:
        """Return the client-side JavaScript for the chart modal."""
        return """
(function () {
    const modal = document.getElementById('image-modal');
    const modalImage = document.getElementById('image-modal-image');
    const modalTitle = document.getElementById('image-modal-title');
    if (!modal || !modalImage || !modalTitle) {
        return;
    }

    const openModal = function (sourceImage) {
        modalImage.src = sourceImage.src;
        modalImage.alt = sourceImage.alt || 'Expanded chart';
        modalTitle.textContent = sourceImage.dataset.title || sourceImage.alt || '';
        modal.classList.add('is-visible');
        modal.setAttribute('aria-hidden', 'false');
        document.body.classList.add('modal-open');
    };

    const closeModal = function () {
        modal.classList.remove('is-visible');
        modal.setAttribute('aria-hidden', 'true');
        modalImage.src = '';
        document.body.classList.remove('modal-open');
    };

    document.querySelectorAll('.expandable-image').forEach(function (imageElement) {
        imageElement.addEventListener('click', function () {
            openModal(imageElement);
        });
    });

    modal.querySelectorAll('[data-close-modal="true"]').forEach(function (element) {
        element.addEventListener('click', closeModal);
    });

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && modal.classList.contains('is-visible')) {
            closeModal();
        }
    });
})();
"""

    @staticmethod
    def _anchor_id(index_name: str) -> str:
        """Return a deterministic HTML identifier and file stem."""
        normalized = "".join(
            character.lower() if character.isalnum() else "-"
            for character in index_name.strip()
        )
        return "-".join(part for part in normalized.split("-") if part)

    def _format_number(self, value: float) -> str:
        """Format a numeric value with the configured precision."""
        if pd.isna(value):
            return "N/A"
        return f"{value:,.{self.config.report.table_precision}f}"

    @staticmethod
    def _format_percent(value: float) -> str:
        """Format a decimal value as a percentage string."""
        if pd.isna(value):
            return "N/A"
        return f"{value * 100:.2f}%"

    def _format_percent_or_na(self, value: float | None) -> str:
        """Format a percentage-like decimal while accepting missing values."""
        if value is None or pd.isna(value):
            return "N/A"
        return self._format_percent(float(value))

    @staticmethod
    def _display_source_label(source_label: str) -> str:
        """Return a cleaner source label for report presentation."""
        mapping = {
            "CSV cache fallback": "CSV cache",
        }
        return mapping.get(source_label, source_label)

    def _strategy_annualized_return(
        self,
        result: IndexAnalysisResult,
        strategy_kind: str,
    ) -> float | None:
        """Return the annualized return for one optional strategy."""
        strategy = None
        if strategy_kind == "original":
            strategy = result.btd_original
        elif strategy_kind == "periodic":
            strategy = result.btd_periodic

        if strategy is None:
            return None
        return self._annualized_return(strategy.final_return, result.input_data)

    @staticmethod
    def _annualized_return(total_return: float, input_data: pd.DataFrame) -> float:
        """Annualize a total return using the input data date span."""
        years = max(
            (input_data["date"].iloc[-1] - input_data["date"].iloc[0]).days / 365.25,
            np.finfo(float).eps,
        )
        if total_return <= -1.0:
            return float("nan")
        return (1.0 + total_return) ** (1.0 / years) - 1.0

    def _build_css(self) -> str:
        """Return the shared dark-theme stylesheet for all report pages."""
        return """
:root {
    color-scheme: dark;
    --bg: #020617;
    --panel: #0f172a;
    --panel-soft: #111c32;
    --border: #243147;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --heading: #f8fafc;
    --accent: #60a5fa;
    --accent-strong: #3b82f6;
    --shadow: 0 16px 40px rgba(2, 6, 23, 0.45);
}
* {
    box-sizing: border-box;
}
body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 0;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
}
body.modal-open {
    overflow: hidden;
}
a {
    color: var(--accent);
}
.container {
    width: min(1360px, calc(100vw - 36px));
    margin: 0 auto;
    padding: 18px 0 28px;
}
.panel {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(10, 15, 28, 0.98));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}
.hero h1 {
    margin: 0 0 10px 0;
    font-size: 32px;
    color: var(--heading);
}
.muted {
    color: var(--muted);
}
.button-link,
.inline-link {
    text-decoration: none;
    color: #eff6ff;
    background: linear-gradient(180deg, var(--accent), var(--accent-strong));
    border-radius: 999px;
    padding: 10px 16px;
    display: inline-block;
    font-weight: 600;
}
.inline-link {
    padding: 6px 12px;
    font-size: 13px;
}
.section-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 8px;
}
h2,
h3,
h4 {
    margin: 0;
    color: var(--heading);
}
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin: 18px 0 6px 0;
}
.metric-card {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(17, 28, 50, 0.9));
}
.metric-card .label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
}
.metric-card .value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 8px;
    color: var(--heading);
}
.summary-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 14px;
    margin-top: 18px;
}
.summary-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    background: linear-gradient(180deg, rgba(17, 28, 50, 0.92), rgba(10, 15, 28, 0.92));
}
.summary-card h3 {
    margin-bottom: 10px;
}
.summary-card p {
    margin: 8px 0;
}
.methodology-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    margin-top: 16px;
}
.method-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(8, 13, 24, 0.96));
}
.method-card p {
    margin: 10px 0;
}
.method-card-warning {
    border-color: rgba(96, 165, 250, 0.45);
}
.method-example,
.formula-box {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: rgba(15, 23, 42, 0.7);
}
.formula-box {
    font-family: "Courier New", Courier, monospace;
    color: #bfdbfe;
}
.figure-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 16px;
    margin-top: 16px;
}
.figure-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(8, 13, 24, 0.96));
}
.figure-card h4 {
    margin-bottom: 10px;
}
.figure-card img {
    width: 100%;
    height: auto;
    border-radius: 10px;
    display: block;
    cursor: zoom-in;
}
.wide-card {
    grid-column: 1 / -1;
}
.table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
    font-size: 14px;
    border: 1px solid var(--border);
}
.table th,
.table td {
    border: 1px solid var(--border);
    padding: 8px 10px;
    text-align: left;
    vertical-align: top;
}
.table thead th {
    background: rgba(30, 41, 59, 0.95);
    color: var(--heading);
}
.table tbody td {
    background: rgba(10, 15, 28, 0.92);
}
.comparison-table tbody tr:last-child td {
    background: rgba(59, 130, 246, 0.14);
    font-weight: 700;
}
.footer-panel {
    text-align: left;
}
.image-modal {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}
.image-modal.is-visible {
    display: flex;
}
.image-modal-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(2, 6, 23, 0.82);
}
.image-modal-content {
    position: relative;
    width: min(96vw, 1600px);
    max-height: 92vh;
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.99), rgba(8, 13, 24, 0.99));
    box-shadow: var(--shadow);
}
.image-modal-title {
    padding-right: 48px;
    margin-bottom: 12px;
    color: var(--heading);
    font-weight: 700;
}
.image-modal-close {
    position: absolute;
    top: 10px;
    right: 12px;
    width: 36px;
    height: 36px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.92);
    color: var(--heading);
    font-size: 24px;
    line-height: 1;
    cursor: pointer;
}
.image-modal-image {
    width: 100%;
    max-height: calc(92vh - 90px);
    object-fit: contain;
    display: block;
    border-radius: 10px;
}
@media (max-width: 900px) {
    .container {
        width: min(100vw - 20px, 100%);
    }
    .figure-grid {
        grid-template-columns: 1fr;
    }
    .image-modal-content {
        width: min(98vw, 98vw);
        padding: 14px;
    }
}
"""


def _format_large_number(value: float, _position: int) -> str:
    """Format large numbers for axes."""
    return f"{value:,.0f}"



def _format_percent_axis_decimal(value: float, _position: int) -> str:
    """Format decimal axes as percentages."""
    return f"{value * 100:.0f}%"



def _format_percentage_axis(value: float, _position: int) -> str:
    """Format percentage axes that already use percentage-point values."""
    return f"{value:.1f}%"
