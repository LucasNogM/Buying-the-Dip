"""Enhanced HTML reporting with implementability and consistency analyses."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from buying_the_dip.models import ComparisonResult, IndexAnalysisResult, SimulationOutput
from buying_the_dip.reporting import (
    HtmlReportBuilder as BaseHtmlReportBuilder,
    _format_large_number,
    _format_percent_axis_decimal,
)


class HtmlReportBuilder(BaseHtmlReportBuilder):
    """Extended report builder with implementability and consistency views."""

    def _build_main_page(
        self,
        results: list[IndexAnalysisResult],
        page_names: dict[str, str],
        generated_at: str,
    ) -> str:
        """Render the main hub page with extra cross-index analyses."""

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
                + "</div>"
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
        implementability_table = self._frame_to_html(
            self._build_main_implementability_table(results),
            escape=False,
            classes=["table", "comparison-table"],
        )
        consistency_table = self._frame_to_html(
            self._build_main_consistency_table(results),
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
                <h2>BtD theoretical vs implementable</h2>
            </div>
            <p class="muted">
                Full-sample comparison between hindsight BtD and live-executable BtD rules.
                The implementation gap is the implementable annualized return minus the
                corresponding theoretical annualized return.
            </p>
            {implementability_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>Consistency of advantage</h2>
            </div>
            <p class="muted">
                Simulation-based robustness summary for periodic BtD versus DCA. Win rate is the
                share of simulation windows with positive annual alpha, and the alpha-distribution
                columns report p10 / median / p90.
            </p>
            {consistency_table}
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
        implementability_table = self._frame_to_html(
            self._build_theoretical_vs_implementable_summary(result)
        )
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
                Comparative view of DCA, theoretical BtD, and implementable BtD variants.
            </p>
            {strategy_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>BtD alpha versus DCA</h2>
            </div>
            <p class="muted">
                Alpha is reported for both theoretical and implementable BtD variants relative to DCA.
            </p>
            {comparison_table}
        </section>

        <section class="panel">
            <div class="section-title-row">
                <h2>BtD theoretical vs implementable</h2>
            </div>
            <p class="muted">
                Comparison of hindsight BtD and live-executable BtD for the same index.
            </p>
            {implementability_table}
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

    def _build_methodology_section(self) -> str:
        """Render a direct explanation of the analytical methodology."""

        dca_period = self.config.strategies.dca.period
        dca_anchor = self.config.strategies.dca.anchor
        periodic_window = self.config.strategies.btd_periodic.period_days
        original_threshold = (
            self.config.strategies.btd_original.implementable_drawdown_threshold_pct
        )
        periodic_threshold = (
            self.config.strategies.btd_periodic.implementable_drawdown_threshold_pct
        )
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
            Each index is downloaded with the API-first loader using the maximum available
            daily history that the selected providers expose. The raw history is cached as CSV,
            cleaned, sorted by date, deduplicated, and filtered to positive prices before the
            analysis begins.
        </p>
    </article>

    <article class="method-card">
        <h3>2. Baseline strategy</h3>
        <p>
            <strong>DCA</strong> contributes on a fixed schedule defined in the configuration.
            In the current setup, the main backtest uses <strong>{html.escape(str(dca_period))}</strong>
            contributions anchored at <strong>{html.escape(str(dca_anchor))}</strong>.
        </p>
    </article>

    <article class="method-card">
        <h3>3. Theoretical BtD</h3>
        <p>
            <strong>Original BtD (theoretical)</strong> waits for the full segment between two
            all-time highs to be known and then allocates at that segment's lowest close.
            <strong>Periodic BtD (theoretical)</strong> does the same inside fixed windows; in the
            main backtest the default window is <strong>{periodic_window} days</strong>.
        </p>
        <div class="method-example">
            These rules are intentionally optimistic because they pick the minimum close only after
            the whole segment or window is already complete.
        </div>
    </article>

    <article class="method-card">
        <h3>4. Implementable BtD</h3>
        <p>
            <strong>Original BtD (implementable)</strong> buys on the <em>first</em> day after a new
            all-time high where the drawdown versus that high reaches
            <strong>{original_threshold * 100:.1f}%</strong> or more.
        </p>
        <p>
            <strong>Periodic BtD (implementable)</strong> buys on the <em>first</em> day inside each
            window where the drawdown from the running in-window peak reaches
            <strong>{periodic_threshold * 100:.1f}%</strong> or more. If that threshold is never hit,
            the strategy falls back to the window end when
            <code>implementable_fallback_to_window_end: true</code>.
        </p>
    </article>

    <article class="method-card">
        <h3>5. Wealth, return, annualization, and alpha</h3>
        <p>
            Portfolio wealth is the sum of the invested component that compounds with the index
            and any idle cash that has not yet been deployed. Final return is visible wealth divided
            by capital made available over time, minus one.
        </p>
        <div class="formula-box">
            Annualized return = (1 + total return)^(1 / years) - 1
        </div>
        <div class="formula-box">
            Alpha = annualized return of strategy A - annualized return of DCA
        </div>
    </article>

    <article class="method-card">
        <h3>6. BtD theoretical vs implementable</h3>
        <p>
            The theoretical-versus-implementable section measures the <strong>implementation gap</strong>.
            In the reports, that gap is shown as implementable annualized return minus theoretical
            annualized return, so negative values mean that the live-executable rule gave back part
            of the hindsight advantage.
        </p>
    </article>

    <article class="method-card">
        <h3>7. Simulation grid and consistency of advantage</h3>
        <p>
            The simulation section compares periodic DCA, theoretical periodic BtD, and implementable
            periodic BtD across multiple contribution intervals and investment horizons. In this
            configuration, the tested intervals are <strong>{html.escape(simulation_intervals)} days</strong>
            and the tested horizons are <strong>{html.escape(simulation_horizons)} years</strong>.
        </p>
        <p>
            <strong>Period</strong> is the spacing between contributions, expressed in days.
            Example: a 30-day period means new capital is added every 30 days.
            <strong>Horizon</strong> is the total length of each simulated backtest window,
            expressed in years. Example: a 5-year horizon means the strategy is evaluated from a
            chosen start date through the next five years.
        </p>
        <p>
            <strong>Win rate</strong> is the share of simulated windows where annual alpha is positive.
            The alpha-distribution summaries report <strong>p10 / median / p90</strong>, which makes it
            easier to see whether the apparent advantage is broad-based or concentrated in a few
            unusually strong windows.
        </p>
    </article>
</div>
"""

    def _build_metric_cards(self, result: IndexAnalysisResult) -> str:
        """Build the comparative card grid for one index page."""

        periodic_consistency = self._overall_consistency_metrics(
            result.simulation,
            "annual_alpha_implementable",
        )
        cards = [
            (
                "DCA annualized return",
                self._format_percent(
                    self._annualized_return(result.dca.final_return, result.input_data)
                ),
            ),
            (
                "Original theoretical alpha vs DCA",
                self._format_percent_or_na(
                    result.original_vs_dca.annual_alpha
                    if result.original_vs_dca is not None
                    else None
                ),
            ),
            (
                "Original implementable alpha vs DCA",
                self._format_percent_or_na(
                    result.original_implementable_vs_dca.annual_alpha
                    if result.original_implementable_vs_dca is not None
                    else None
                ),
            ),
            (
                "Periodic theoretical alpha vs DCA",
                self._format_percent_or_na(
                    result.periodic_vs_dca.annual_alpha
                    if result.periodic_vs_dca is not None
                    else None
                ),
            ),
            (
                "Periodic implementable alpha vs DCA",
                self._format_percent_or_na(
                    result.periodic_implementable_vs_dca.annual_alpha
                    if result.periodic_implementable_vs_dca is not None
                    else None
                ),
            ),
            (
                "Implementable periodic win rate",
                self._format_percent_or_na(periodic_consistency.get("win_rate")),
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
        for output in (
            result.btd_original,
            result.btd_original_implementable,
            result.btd_periodic,
            result.btd_periodic_implementable,
        ):
            if output is not None:
                rows.append(self._strategy_summary_row(output, result.input_data))
        return pd.DataFrame(rows)

    def _build_comparison_summary(self, result: IndexAnalysisResult) -> pd.DataFrame:
        """Return the BtD-versus-DCA alpha table for one index."""

        rows: list[dict[str, str]] = []
        for comparison in (
            result.original_vs_dca,
            result.original_implementable_vs_dca,
            result.periodic_vs_dca,
            result.periodic_implementable_vs_dca,
        ):
            if comparison is not None:
                rows.append(self._comparison_summary_row(comparison))
        if not rows:
            rows.append({"Comparison": "No comparison enabled."})
        return pd.DataFrame(rows)

    def _build_main_implementability_table(
        self,
        results: list[IndexAnalysisResult],
    ) -> pd.DataFrame:
        """Build the cross-index implementability table."""

        rows: list[dict[str, str]] = []
        for result in results:
            rows.append(
                {
                    "Index": result.index_name,
                    "Original theoretical annualized return": self._format_percent_or_na(
                        self._strategy_annualized_return(result, "original")
                    ),
                    "Original implementable annualized return": self._format_percent_or_na(
                        self._strategy_annualized_return(result, "original_implementable")
                    ),
                    "Original implementation gap": self._format_percent_or_na(
                        self._annualized_gap(
                            result,
                            result.btd_original_implementable,
                            result.btd_original,
                        )
                    ),
                    "Periodic theoretical annualized return": self._format_percent_or_na(
                        self._strategy_annualized_return(result, "periodic")
                    ),
                    "Periodic implementable annualized return": self._format_percent_or_na(
                        self._strategy_annualized_return(result, "periodic_implementable")
                    ),
                    "Periodic implementation gap": self._format_percent_or_na(
                        self._annualized_gap(
                            result,
                            result.btd_periodic_implementable,
                            result.btd_periodic,
                        )
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_main_consistency_table(
        self,
        results: list[IndexAnalysisResult],
    ) -> pd.DataFrame:
        """Build the cross-index simulation-consistency table."""

        rows: list[dict[str, str]] = []
        for result in results:
            theoretical = self._overall_consistency_metrics(
                result.simulation,
                "annual_alpha_theoretical",
            )
            implementable = self._overall_consistency_metrics(
                result.simulation,
                "annual_alpha_implementable",
            )
            rows.append(
                {
                    "Index": result.index_name,
                    "Simulation windows": self._format_count(theoretical.get("count")),
                    "Theoretical win rate": self._format_percent_or_na(
                        theoretical.get("win_rate")
                    ),
                    "Theoretical alpha distribution": self._format_distribution_text(
                        theoretical
                    ),
                    "Implementable win rate": self._format_percent_or_na(
                        implementable.get("win_rate")
                    ),
                    "Implementable alpha distribution": self._format_distribution_text(
                        implementable
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_theoretical_vs_implementable_summary(
        self,
        result: IndexAnalysisResult,
    ) -> pd.DataFrame:
        """Return the per-index implementation-gap summary table."""

        rows: list[dict[str, str]] = []
        rows.append(
            self._implementation_gap_row(
                variant_label="Original BtD",
                result=result,
                theoretical_output=result.btd_original,
                implementable_output=result.btd_original_implementable,
                theoretical_vs_dca=result.original_vs_dca,
                implementable_vs_dca=result.original_implementable_vs_dca,
            )
        )
        rows.append(
            self._implementation_gap_row(
                variant_label="Periodic BtD",
                result=result,
                theoretical_output=result.btd_periodic,
                implementable_output=result.btd_periodic_implementable,
                theoretical_vs_dca=result.periodic_vs_dca,
                implementable_vs_dca=result.periodic_implementable_vs_dca,
            )
        )
        return pd.DataFrame(rows)

    def _implementation_gap_row(
        self,
        variant_label: str,
        result: IndexAnalysisResult,
        theoretical_output,
        implementable_output,
        theoretical_vs_dca: ComparisonResult | None,
        implementable_vs_dca: ComparisonResult | None,
    ) -> dict[str, str]:
        """Build one theoretical-versus-implementable summary row."""

        theoretical_return = (
            self._annualized_return(theoretical_output.final_return, result.input_data)
            if theoretical_output is not None
            else None
        )
        implementable_return = (
            self._annualized_return(implementable_output.final_return, result.input_data)
            if implementable_output is not None
            else None
        )
        return {
            "Variant": variant_label,
            "Theoretical annualized return": self._format_percent_or_na(
                theoretical_return
            ),
            "Implementable annualized return": self._format_percent_or_na(
                implementable_return
            ),
            "Implementation gap": self._format_percent_or_na(
                self._difference(implementable_return, theoretical_return)
            ),
            "Theoretical alpha vs DCA": self._format_percent_or_na(
                theoretical_vs_dca.annual_alpha if theoretical_vs_dca else None
            ),
            "Implementable alpha vs DCA": self._format_percent_or_na(
                implementable_vs_dca.annual_alpha if implementable_vs_dca else None
            ),
            "Alpha gap": self._format_percent_or_na(
                self._difference(
                    implementable_vs_dca.annual_alpha
                    if implementable_vs_dca is not None
                    else None,
                    theoretical_vs_dca.annual_alpha
                    if theoretical_vs_dca is not None
                    else None,
                )
            ),
            "Theoretical contributions": self._format_count(
                theoretical_output.contribution_count if theoretical_output else None
            ),
            "Implementable contributions": self._format_count(
                implementable_output.contribution_count if implementable_output else None
            ),
        }

    def _build_simulation_section(self, result: IndexAnalysisResult) -> str:
        """Build the optional simulation section for one index page."""

        if (
            not self.config.report.include_simulation
            or result.simulation is None
            or result.simulation.aggregated_results.empty
        ):
            return ""

        summary_table = self._frame_to_html(self._build_simulation_table(result))
        consistency_table = self._frame_to_html(
            self._build_simulation_consistency_table(result)
        )
        figures: list[str] = [
            self._figure_card(
                "Mean annual alpha by interval and horizon (theoretical)",
                self._plot_simulation_metric_heatmap(
                    result.simulation,
                    "annual_alpha_theoretical_mean",
                    "Mean annual alpha (theoretical)",
                ),
                wide=True,
            ),
            self._figure_card(
                "Mean annual alpha as the horizon increases",
                self._plot_simulation_metric_lines(
                    result.simulation,
                    "annual_alpha_theoretical_mean",
                    "Mean annual alpha by horizon (theoretical)",
                ),
                wide=True,
            ),
            self._figure_card(
                "Win rate by interval and horizon (theoretical)",
                self._plot_simulation_metric_heatmap(
                    result.simulation,
                    "theoretical_win_rate",
                    "Win rate (theoretical)",
                ),
                wide=True,
            ),
        ]

        if result.btd_periodic_implementable is not None:
            figures.extend(
                [
                    self._figure_card(
                        "Mean annual alpha by interval and horizon (implementable)",
                        self._plot_simulation_metric_heatmap(
                            result.simulation,
                            "annual_alpha_implementable_mean",
                            "Mean annual alpha (implementable)",
                        ),
                        wide=True,
                    ),
                    self._figure_card(
                        "Mean annual alpha as the horizon increases (implementable)",
                        self._plot_simulation_metric_lines(
                            result.simulation,
                            "annual_alpha_implementable_mean",
                            "Mean annual alpha by horizon (implementable)",
                        ),
                        wide=True,
                    ),
                    self._figure_card(
                        "Win rate by interval and horizon (implementable)",
                        self._plot_simulation_metric_heatmap(
                            result.simulation,
                            "implementable_win_rate",
                            "Win rate (implementable)",
                        ),
                        wide=True,
                    ),
                ]
            )

        figures_html = "\n".join(figures)
        return f"""
<section class="panel">
    <div class="section-title-row">
        <h2>Simulation</h2>
    </div>
    <p class="muted">
        Periodic DCA, theoretical periodic BtD, and implementable periodic BtD are simulated over
        the configured horizons and intervals. The summary table shows the strongest combinations,
        while the consistency table aggregates win rate and alpha distribution by investment period.
    </p>
    <div class="figure-grid">{figures_html}</div>
    <h3>Top simulation combinations</h3>
    {summary_table}
    <h3>Consistency of advantage by investment period</h3>
    {consistency_table}
</section>
"""

    def _build_simulation_table(self, result: IndexAnalysisResult) -> pd.DataFrame:
        """Return the top simulation combinations for one index."""

        simulation_frame = result.simulation.aggregated_results.copy()
        sort_column = "annual_alpha_theoretical_mean"
        if (
            "annual_alpha_implementable_mean" in simulation_frame.columns
            and simulation_frame["annual_alpha_implementable_mean"].notna().any()
        ):
            sort_column = "annual_alpha_implementable_mean"
        simulation_frame = simulation_frame.sort_values(
            by=sort_column,
            ascending=False,
        ).head(self.config.report.max_simulation_rows)

        rows: list[dict[str, str]] = []
        for _, row in simulation_frame.iterrows():
            rows.append(
                {
                    "Horizon (years)": f"{int(row['horizon_years'])}",
                    "Investment period (days)": f"{int(row['contribution_interval_days'])}",
                    "Windows": self._format_count(row.get("window_count")),
                    "Theoretical mean alpha": self._format_percent_or_na(
                        row.get("annual_alpha_theoretical_mean")
                    ),
                    "Theoretical win rate": self._format_percent_or_na(
                        row.get("theoretical_win_rate")
                    ),
                    "Implementable mean alpha": self._format_percent_or_na(
                        row.get("annual_alpha_implementable_mean")
                    ),
                    "Implementable win rate": self._format_percent_or_na(
                        row.get("implementable_win_rate")
                    ),
                    "Implementation alpha gap": self._format_percent_or_na(
                        row.get("implementable_minus_theoretical_annual_alpha_mean")
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _build_simulation_consistency_table(
        self,
        result: IndexAnalysisResult,
    ) -> pd.DataFrame:
        """Return a compact simulation-consistency summary by interval."""

        raw_results = result.simulation.raw_results
        rows: list[dict[str, str]] = []
        for interval_days, frame in raw_results.groupby(
            "contribution_interval_days",
            sort=True,
        ):
            theoretical = self._alpha_series_metrics(frame["annual_alpha_theoretical"])
            implementable = self._alpha_series_metrics(frame["annual_alpha_implementable"])
            rows.append(
                {
                    "Investment period (days)": f"{int(interval_days)}",
                    "Simulation windows": self._format_count(theoretical.get("count")),
                    "Theoretical win rate": self._format_percent_or_na(
                        theoretical.get("win_rate")
                    ),
                    "Theoretical alpha distribution": self._format_distribution_text(
                        theoretical
                    ),
                    "Implementable win rate": self._format_percent_or_na(
                        implementable.get("win_rate")
                    ),
                    "Implementable alpha distribution": self._format_distribution_text(
                        implementable
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _plot_simulation_metric_heatmap(
        self,
        simulation: SimulationOutput,
        value_column: str,
        title: str,
    ) -> str:
        """Plot a simulation metric heatmap."""

        source_frame = simulation.aggregated_results
        if value_column.endswith("win_rate"):
            source_frame = simulation.consistency_results

        pivot = source_frame.pivot(
            index="contribution_interval_days",
            columns="horizon_years",
            values=value_column,
        ).sort_index(axis=0).sort_index(axis=1)

        figure, axis = self._create_figure(figsize=(10, 5.2))
        image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        axis.set_title(title)
        axis.set_xlabel("Investment horizon (years)")
        axis.set_ylabel("Investment period (days)")
        axis.set_xticks(np.arange(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns.astype(int))
        axis.set_yticks(np.arange(len(pivot.index)))
        axis.set_yticklabels(pivot.index.astype(int))
        self._apply_axis_style(axis)

        color_bar = figure.colorbar(image, ax=axis)
        color_bar.ax.yaxis.set_major_formatter(
            FuncFormatter(_format_percent_axis_decimal)
        )
        color_bar.ax.tick_params(colors=self.muted_text_color)
        color_bar.ax.yaxis.label.set_color(self.text_color)
        color_bar.outline.set_edgecolor(self.border_color)

        for row_index in range(pivot.shape[0]):
            for column_index in range(pivot.shape[1]):
                value = pivot.iloc[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    self._format_percent_or_na(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=self.text_color,
                )

        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _plot_simulation_metric_lines(
        self,
        simulation: SimulationOutput,
        value_column: str,
        title: str,
    ) -> str:
        """Plot a simulation metric across horizons for each investment period."""

        source_frame = simulation.aggregated_results
        if value_column.endswith("win_rate"):
            source_frame = simulation.consistency_results

        figure, axis = self._create_figure(figsize=(12, 4.6))
        for interval_days, frame in source_frame.groupby(
            "contribution_interval_days",
            sort=True,
        ):
            ordered = frame.sort_values("horizon_years")
            axis.plot(
                ordered["horizon_years"],
                ordered[value_column],
                label=f"{int(interval_days)}",
            )

        horizon_values = source_frame["horizon_years"].dropna().astype(int)
        if not horizon_values.empty:
            min_horizon = int(horizon_values.min())
            max_horizon = int(horizon_values.max())
            axis.set_xlim(min_horizon, max_horizon)
            axis.set_xticks(np.arange(min_horizon, max_horizon + 1, 1))

        axis.set_title(title)
        axis.set_xlabel("Investment horizon (years)")
        axis.set_ylabel("Mean annual alpha")
        axis.yaxis.set_major_formatter(FuncFormatter(_format_percent_axis_decimal))
        self._apply_axis_style(axis)
        self._style_legend(axis, ncols=2, title="Investment period (days)")
        figure.tight_layout()
        return self._figure_to_base64(figure)

    def _strategy_annualized_return(
        self,
        result: IndexAnalysisResult,
        strategy_kind: str,
    ) -> float | None:
        """Return the annualized return for one optional strategy."""

        strategy = None
        if strategy_kind == "original":
            strategy = result.btd_original
        elif strategy_kind == "original_implementable":
            strategy = result.btd_original_implementable
        elif strategy_kind == "periodic":
            strategy = result.btd_periodic
        elif strategy_kind == "periodic_implementable":
            strategy = result.btd_periodic_implementable

        if strategy is None:
            return None
        return self._annualized_return(strategy.final_return, result.input_data)

    @staticmethod
    def _difference(value_a: float | None, value_b: float | None) -> float | None:
        """Return a simple difference if both values are present."""

        if value_a is None or value_b is None:
            return None
        if pd.isna(value_a) or pd.isna(value_b):
            return None
        return float(value_a) - float(value_b)

    def _annualized_gap(
        self,
        result: IndexAnalysisResult,
        output_a,
        output_b,
    ) -> float | None:
        """Return the gap in annualized return between two strategy outputs."""

        if output_a is None or output_b is None:
            return None
        return self._difference(
            self._annualized_return(output_a.final_return, result.input_data),
            self._annualized_return(output_b.final_return, result.input_data),
        )

    def _overall_consistency_metrics(
        self,
        simulation: SimulationOutput | None,
        alpha_column: str,
    ) -> dict[str, float | int | None]:
        """Aggregate consistency metrics across every simulation window."""

        if simulation is None or simulation.raw_results.empty or alpha_column not in simulation.raw_results:
            return {}
        return self._alpha_series_metrics(simulation.raw_results[alpha_column])

    @staticmethod
    def _alpha_series_metrics(series: pd.Series) -> dict[str, float | int | None]:
        """Return win-rate and percentile metrics for one alpha series."""

        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {}
        return {
            "count": int(numeric.size),
            "win_rate": float((numeric > 0).mean()),
            "p10": float(numeric.quantile(0.10)),
            "median": float(numeric.median()),
            "p90": float(numeric.quantile(0.90)),
        }

    def _format_distribution_text(self, metrics: dict[str, float | int | None]) -> str:
        """Format p10 / median / p90 into a compact text cell."""

        if not metrics:
            return "N/A"
        return " / ".join(
            [
                self._format_percent_or_na(metrics.get("p10")),
                self._format_percent_or_na(metrics.get("median")),
                self._format_percent_or_na(metrics.get("p90")),
            ]
        )

    @staticmethod
    def _format_count(value: object) -> str:
        """Format an integer count or return N/A."""

        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{int(value):,}"
