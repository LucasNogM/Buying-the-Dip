# Buying the Dip Analysis

A Python project to compare DCA with both **theoretical** and **implementable** Buying the Dip (BtD) approaches across multiple market indices in a single run:

- **DCA** (Dollar-Cost Averaging)
- **BtD Original (Theoretical)**: buys the local minimum inside each completed all-time-high reset segment
- **BtD Original (Implementable)**: buys the first drawdown after an all-time high once a configurable threshold is reached
- **BtD Periodic (Theoretical)**: buys the local minimum inside each fixed investment window
- **BtD Periodic (Implementable)**: buys the first in-window drawdown once a configurable threshold is reached

The codebase is fully refactored into classes, uses YAML configuration, downloads data from an API first, caches it as CSV, falls back to local cache when needed, exports raw analytical outputs, and generates dark-theme HTML reports:

- one **main report hub** covering all configured indices;
- one **dedicated HTML page per index**;
- bidirectional navigation between the hub and each index page;
- charts that can be expanded by clicking them in the index reports.

---

## Contents

- [What this project does](#what-this-project-does)
- [Strategy definitions](#strategy-definitions)
- [Methodology at a glance](#methodology-at-a-glance)
- [Important interpretation caveat](#important-interpretation-caveat)
- [BtD theoretical vs implementable](#btd-theoretical-vs-implementable)
- [Consistency of advantage](#consistency-of-advantage)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration guide](#configuration-guide)
- [Data source modes](#data-source-modes)
- [Reports](#reports)
- [Raw exports](#raw-exports)
- [Performance notes](#performance-notes)
- [Extending the project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Practical limitations](#practical-limitations)

---

## What this project does

The pipeline loads one or more market indices, standardizes each time series into a common internal format, runs the configured strategies, compares their results, optionally runs a simulation sweep across multiple contribution intervals and horizons, exports raw result tables, and finally builds HTML reports.

The default setup is already configured for these indices:

- Ibovespa
- S&P 500
- Dow Jones
- Nasdaq Composite
- STOXX Europe 600
- Nikkei 225
- Shanghai Composite

The default setup is now **provider-first with fallback chaining**. The pipeline first tries the provider order configured in `data_api.provider_priority` and each index can override that order with `provider_priority`. When `analysis_start_date` is configured for an index, the loader keeps trying later providers if the earlier provider returns a shorter history than requested, and then keeps the earliest successful history available. After a successful download, the normalized history is saved locally as CSV so later runs can reuse it if the API is temporarily unavailable.

---

## Strategy definitions

### 1. DCA

DCA invests on a fixed schedule.

Examples:

- monthly at the end of each month;
- weekly;
- every 30 days;
- every 60 days.

The schedule is configurable in `strategies.dca.period` and `strategies.dca.anchor`.

### 2. BtD Original (Theoretical)

The theoretical original Buying the Dip implementation works like this:

1. compute the running all-time high;
2. split the price series into segments delimited by new highs;
3. in each completed segment, find the **lowest close**;
4. allocate the available capital at that lowest close.

### 3. BtD Original (Implementable)

The implementable original BtD rule uses only information available at the decision date:

1. compute the most recent all-time high;
2. wait for the first close that is below that high by at least a configurable drawdown threshold;
3. allocate on that first threshold hit.

### 4. BtD Periodic (Theoretical)

The theoretical periodic Buying the Dip implementation works like this:

1. split time into fixed windows such as 15, 30, 45, 60, 90, or 120 days;
2. inside each window, find the **lowest close**;
3. allocate the available capital at that lowest close.

### 5. BtD Periodic (Implementable)

The implementable periodic BtD rule works like this:

1. split time into fixed windows such as 15, 30, 45, 60, 90, or 120 days;
2. track the running peak inside each window;
3. allocate on the first day where the drawdown from that running peak reaches the configured threshold;
4. optionally fall back to the end of the window if the threshold is never hit.

This project treats DCA as the **baseline** and measures all BtD variants relative to it.

---

## Methodology at a glance

### Data normalization

Every input index is converted to the same internal schema:

- `date`
- `price`

Then the loader:

- parses dates;
- sorts rows ascending by date;
- removes duplicated dates;
- removes missing or non-positive prices;
- computes return helpers used by the backtest engine.

### Cash handling

The portfolio engine supports two cash behaviors:

- **periodic**: capital only appears on investment dates;
- **linear**: capital accrues gradually between investment dates and remains idle until deployed.

### Wealth and return

For each strategy, the engine computes:

- contribution dates;
- contributed capital;
- invested wealth;
- visible wealth (invested wealth plus idle cash not yet deployed);
- total return;
- annualized return.

The annualized return is used because the indices have long and unequal histories.

```text
annualized return = (1 + total return)^(1 / years) - 1
```

### Alpha

Alpha is always computed against DCA:

```text
alpha = annualized return of BtD - annualized return of DCA
```

Interpretation:

- positive alpha: the BtD approach outperformed DCA for that index;
- zero alpha: they were equivalent on an annualized basis;
- negative alpha: DCA outperformed the BtD approach.

### Simulation grid

The simulation section compares **periodic DCA** and **periodic BtD** across multiple settings.

In the current default configuration, the sweep uses:

- **investment periods** of `15, 30, 45, 60, 90, 120` days;
- **investment horizons** of `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20` years.

Definitions:

- **Investment period**: the spacing between contributions. Example: a 30-day period means new capital is made available every 30 days.
- **Investment horizon**: the total length of one simulation window. Example: a 5-year horizon means the strategy is evaluated from a chosen start date through the next five years.

For each combination, the code samples multiple windows across the full history and aggregates results such as mean annual alpha, win rate, and alpha percentiles.

---

## Important interpretation caveat

The project now distinguishes between **theoretical** BtD and **implementable** BtD.

- **Theoretical BtD** uses ex-post minima. It identifies the minimum price **after the full segment or full window is known**.
- **Implementable BtD** uses only information available at the decision date and therefore behaves like a live-executable rule.

This makes the theoretical outputs useful as analytical timing benchmarks, while the implementable outputs are a more realistic reference for deployable rules.

In addition, the project does **not** model:

- transaction costs;
- taxes;
- spreads;
- slippage;
- liquidity constraints;
- order execution delays.

So the BtD results should be interpreted as a clean comparative research exercise rather than a production trading system.

---

## BtD theoretical vs implementable

The reports include a dedicated comparison between the hindsight and live-executable versions of BtD.

The key metric is the **implementation gap**:

```text
implementation gap = implementable annualized return - theoretical annualized return
```

Interpretation:

- negative implementation gap: the live-executable rule gave back part of the hindsight advantage;
- zero implementation gap: the implementable rule matched the theoretical rule;
- positive implementation gap: the implementable rule outperformed the hindsight benchmark, which is rare but possible in some samples.

---

## Consistency of advantage

The simulation module no longer reports only average alpha. It also measures whether the advantage is **consistent** across many start dates, horizons, and investment periods.

The reports now include:

- **win rate**: the share of simulation windows where annual alpha is positive;
- **alpha distribution**: p10 / median / p90 for annual alpha;
- side-by-side consistency summaries for the theoretical and implementable periodic BtD rules.

This is useful because a strategy can have a positive average alpha but still rely on a small number of unusually favorable windows.

---

## Project structure

```text
buying_the_dip_delivery/
├── buying_the_dip/
│   ├── __init__.py
│   ├── analytics.py
│   ├── config.py
│   ├── data_loader.py
│   ├── models.py
│   ├── portfolio.py
│   ├── reporting.py
│   ├── scheduling.py
│   └── strategies.py
├── config.yaml
├── config.offline.yaml
├── main.py
├── requirements.txt
├── sample_cache/
└── README.md
```

### Module responsibilities

- `config.py`: dataclasses, validation, YAML loading.
- `data_loader.py`: API downloads, local file loading, CSV cache fallback.
- `scheduling.py`: contribution schedules and fixed-window labels.
- `strategies.py`: DCA plus theoretical and implementable BtD signal generation.
- `portfolio.py`: capital allocation and wealth backtesting logic.
- `analytics.py`: orchestration, comparisons, simulation sweep, raw exports.
- `reporting.py`: base HTML reporting implementation.
- `reporting_enhanced.py`: implementability and consistency report extensions.
- `main.py`: CLI entry point.

---

## Requirements

The repository uses these core Python dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `PyYAML`
- `openpyxl`

Notes:

- `openpyxl` is required for XLSX export.
- CSV mode works without Excel support.
- Local Parquet input can be used through pandas, but may require an additional parquet engine in your environment.

---

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. API-first run

```bash
python main.py --config config.yaml
```

This will:

1. download the configured indices from the configured provider chain when possible;
2. save normalized histories into the configured CSV caches;
3. run all enabled analyses;
4. export raw result tables;
5. generate the HTML report hub and one HTML page per index.

### 2. Offline validation run

```bash
python main.py --config config.offline.yaml
```

The offline configuration is useful when you want to validate the full report-generation pipeline without depending on network availability. It uses the bundled CSV cache files in `sample_cache/` as fallback inputs. Those offline cache files are synthetic but follow the same date policy as the default YAML: 1950 for the long-history indices, later starts when appropriate, and post-Real data for Ibovespa.

### 3. Open the reports

After the run, open:

```text
<report.output_directory>/index.html
```

That page is the report hub and links to the dedicated page for each index.

---

## Configuration guide

All runtime behavior is controlled through YAML.

### Project settings

```yaml
project:
  title: "Buying the Dip Analysis"
  output_directory: "./outputs/raw"
  log_level: "INFO"
```

- `title`: report title.
- `output_directory`: raw export location.
- `log_level`: logging verbosity.

### API settings

```yaml
data_api:
  provider: "stooq"
  provider_priority: ["stooq", "marketwatch", "yahoo_finance"]
  period: "max"
  interval: "1d"
  timeout_seconds: 25
  max_workers: 1
  max_retries: 4
  retry_backoff_seconds: 2.0
  retry_jitter_seconds: 0.5
  request_pause_seconds: 1.25
  continue_on_index_error: false
```

- `provider`: default provider when no per-index override is given.
- `provider_priority`: ordered fallback chain used when an index does not define its own provider order. If `analysis_start_date` is set for an index, the loader can continue through this chain to find a longer history.
- `period`: the history range requested from the API.
- `interval`: the API candle interval.
- `timeout_seconds`: request timeout.
- `max_workers`: concurrent download workers. Keep this at `1` for remote market data providers unless you know the endpoint is stable in your environment.
- `max_retries`: number of retries after the initial request.
- `retry_backoff_seconds`: exponential retry base delay.
- `retry_jitter_seconds`: random delay component added to retries.
- `request_pause_seconds`: pause inserted between outbound API requests to reduce rate limiting.
- `continue_on_index_error`: if `true`, the run skips failed indices instead of stopping the whole execution.

### Index list

Each entry in `indices:` defines one index source.

#### API with cache fallback


#### Provider-specific symbols

Some providers use different symbols for the same index.

```yaml
- name: "Ibovespa"
  source_mode: "api_with_cache"
  symbol: "^BVSP"
  provider_priority: ["stooq", "yahoo_finance"]
  provider_symbols:
    stooq: "^BVP"
    yahoo_finance: "^BVSP"
  cache_path: "./data_cache/ibovespa.csv"
  analysis_start_date: "1994-07-01"
```

You can also pass provider-specific options, for example for MarketWatch:

```yaml
- name: "STOXX Europe 600"
  source_mode: "api_with_cache"
  symbol: "^STOXX"
  provider_priority: ["marketwatch", "yahoo_finance"]
  provider_symbols:
    marketwatch: "sxxp"
    yahoo_finance: "^STOXX"
  provider_options:
    marketwatch:
      instrument_type: "index"
      country_code: "XX"
  cache_path: "./data_cache/stoxx_europe_600.csv"
```


```yaml
- name: "S&P 500"
  source_mode: "api_with_cache"
  symbol: "^GSPC"
  cache_path: "./data_cache/sp500.csv"
  analysis_start_date: "1950-01-01"
```

#### Per-index analysis and simulation start dates

Use `analysis_start_date` to define the first date considered by the main backtests, comparisons, charts, and report metrics for one index.

Use `simulation_start_date` only when you want the simulation windows to start later than the main analysis window for that same index. If you omit it, the simulation automatically starts from `analysis_start_date`.

```yaml
- name: "Ibovespa"
  source_mode: "api_with_cache"
  symbol: "^BVSP"
  provider_priority: ["stooq", "yahoo_finance"]
  provider_symbols:
    stooq: "^BVP"
    yahoo_finance: "^BVSP"
  cache_path: "./data_cache/ibovespa.csv"
  analysis_start_date: "1994-07-01"

- name: "S&P 500"
  source_mode: "api_with_cache"
  symbol: "^GSPC"
  cache_path: "./data_cache/sp500.csv"
  analysis_start_date: "1950-01-01"
```

Recommended default policy:

- use `1950-01-01` for broad long-history indices;
- keep `Ibovespa` at `1994-07-01` or later to avoid mixing pre-Real currency regimes;
- only add `simulation_start_date` when you intentionally want a narrower simulation window than the main analysis window.

#### API only

```yaml
- name: "Nasdaq Composite"
  source_mode: "api"
  symbol: "^IXIC"
```

#### Local file

```yaml
- name: "Custom Index"
  source_mode: "file"
  path: "./data/custom_index.csv"
  date_column: "Date"
  price_column: "Close"
  date_format: "%Y-%m-%d"
```

Supported local file formats:

- CSV
- Excel (`.xlsx`, `.xls`)
- Parquet

### Strategy settings

```yaml
strategies:
  dca:
    period: "monthly"
    anchor: "end"
    cash_mode: "linear"
    contribution_amount: 1.0
    cash_frequency: "monthly"

  btd_original:
    enabled: true
    cash_mode: "linear"
    contribution_amount: 1.0
    cash_frequency: "monthly"

  btd_periodic:
    enabled: true
    period_days: 30
    anchor: "end"
    cash_mode: "periodic"
    contribution_amount: 1.0
    cash_frequency: "monthly"
```

#### DCA period

`strategies.dca.period` accepts either:

- a calendar frequency: `daily`, `weekly`, `monthly`, `quarterly`;
- or an integer number of days.

#### Anchor

Available anchors:

- `start`
- `end`

This controls whether calendar or fixed-day schedules are aligned to the start or end of the period/window.

#### Cash mode

Available cash modes:

- `periodic`
- `linear`

### Simulation settings

```yaml
simulation:
  enabled: true
  horizons_years: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
  contribution_intervals_days: [15, 30, 45, 60, 90, 120]
  window_count: 5
  anchor: "end"
  cash_mode: "periodic"
  contribution_amount: 1.0
  cash_frequency: "monthly"
  aggregations:
    annual_alpha: ["mean", "median"]
    final_relative_advantage: ["mean", "median"]
```

You can also define `horizons_years` and `contribution_intervals_days` using a range-like mapping:

```yaml
horizons_years:
  start: 1
  stop: 10
  step: 1
  inclusive: true
```

### Report settings

```yaml
report:
  output_directory: "./outputs/reports"
  index_file_name: "index.html"
  table_precision: 4
  figure_dpi: 180
  include_original_btd: true
  include_periodic_btd: true
  include_simulation: true
  max_simulation_rows: 12
  theme: "dark"
  include_overview_chart: true
  advantage_clip_percentile: 97.5
  advantage_min_limit_pct: 0.25
```

Notable options:

- `include_simulation`: include or remove the simulation section from index pages.
- `include_overview_chart`: include a normalized cross-index price chart on the hub page.
- `advantage_clip_percentile` and `advantage_min_limit_pct`: control robust scaling for the relative-advantage chart when extreme outliers would flatten the series.

### Raw export settings

```yaml
exports:
  enabled: true
  format: "csv"
  include_input_data: false
```

Supported export formats:

- `csv`
- `xlsx`

If `format: xlsx`, the project writes one workbook per index.

---

## Data source modes

### `file`

Loads a local file directly.

Use this when:

- you already have your own cleaned history;
- you want to compare proprietary data;
- you do not want to call an API.

### `api`

Downloads data from one explicitly configured provider only.

Use this when:

- you always want fresh data;
- you do not care about a local fallback cache.

### `api_with_cache`

Attempts the API first and falls back to the local CSV cache if the request fails.

This is the recommended mode for most use cases because it balances convenience and resilience.

---

## Reports

### Main report hub

The main HTML page contains:

- the report title and high-level KPIs;
- the **Index reports** table with symbol, earliest date, latest date, latest price, and a link to the dedicated page;
- a **General comparison** table comparing the three strategies for every index;
- a final aggregate row comparing **cross-index mean alpha** for the two BtD variants;
- an optional normalized price chart covering all configured indices;
- a concise but complete **Methodology** section.

### Dedicated index pages

Each index page contains:

- a link back to the main report;
- source and date-range metadata;
- comparative annualized-return and alpha KPI cards;
- a strategy comparison table;
- a BtD-versus-DCA alpha comparison table;
- expandable charts;
- a simulation heatmap and simulation line chart when enabled;
- a simulation summary table.

### Expandable charts

The per-index reports include client-side JavaScript so that clicking a chart opens it in a modal overlay for easier reading.

---

## Raw exports

When raw exports are enabled, the pipeline writes one bundle per index.

### CSV mode

Each index gets its own directory, typically similar to:

```text
outputs/raw/ibovespa/
```

Inside that directory, the project may write files such as:

- `dca.csv`
- `btd_original.csv`
- `btd_periodic.csv`
- `original_vs_dca.csv`
- `periodic_vs_dca.csv`
- `simulation_raw.csv`
- `simulation_aggregated.csv`

### XLSX mode

Each index gets one workbook containing the same tables as separate sheets.

---

## Performance notes

The refactor improves performance compared with a notebook-style implementation in several ways:

- concurrent API downloads for multiple indices;
- vectorized wealth accumulation in NumPy;
- efficient simulation window slicing with `searchsorted`;
- list-based record collection before DataFrame construction;
- CSV caching to avoid repeated API requests;
- reusable classes that avoid redundant ad hoc transformations.

---

## Extending the project

Common extension paths include:

- adding a new data provider in `data_loader.py`;
- adding a new investment strategy in `strategies.py`;
- introducing transaction-cost modeling in `portfolio.py` or `analytics.py`;
- adding more report sections in `reporting.py`;
- exporting additional derived metrics in `analytics.py`.

A typical new-strategy workflow is:

1. create a strategy class that returns an investment signal;
2. run it through `PortfolioEngine`;
3. compare it with the existing strategies;
4. add the desired tables and charts to the HTML layer.

---

## Troubleshooting

### The API is unavailable

If your index source uses `api_with_cache`, the pipeline automatically tries the CSV cache.

If no cache exists yet, run once with network access so the cache can be created.

### Yahoo Finance returns HTTP 429

This means the endpoint throttled the requests. The default configuration is already tuned to reduce that risk:

- `data_api.max_workers: 1`
- `data_api.max_retries: 4`
- `data_api.request_pause_seconds: 1.25`

If you still get `429 Too Many Requests` on the first run:

1. wait a bit and rerun;
2. do not increase `max_workers`;
3. increase `request_pause_seconds`;
4. keep the generated CSV cache so later executions can fall back locally;
5. prefer a provider chain with more than one source, because the loader now keeps the earliest successful history if a later provider is throttled.

### A report page shows no simulation section

Check:

- `simulation.enabled`
- `report.include_simulation`
- whether the available history is long enough for the configured horizons

Very short series may not support long simulation windows.

### My local file fails to load

Check:

- `source_mode: file`
- `path`
- `date_column`
- `price_column`
- `delimiter`
- `encoding`
- `date_format`

Also confirm the file really contains one date column and one price column.

### Parquet input does not work

Install a parquet engine compatible with pandas in your environment.

### The relative-advantage chart looks flat

The reporting layer already applies a robust y-axis when a few outliers dominate the scale. You can further tune:

- `report.advantage_clip_percentile`
- `report.advantage_min_limit_pct`

---

## Practical limitations

This repository is intentionally focused on analytical comparison rather than production execution.

Keep these limitations in mind:

1. **BtD uses hindsight minima**.
2. **No trading frictions are modeled**.
3. **Index histories can differ in length and composition**.
4. **The quality and continuity of the API data source matter**.
5. **The simulation grid measures sensitivity, not certainty**.

---

## Suggested repository usage

For a GitHub repository, a practical workflow is:

1. keep `config.yaml` as the main API-first setup;
2. keep `config.offline.yaml` plus `sample_cache/` for reproducible offline validation;
3. commit the source code and configuration;
4. decide whether generated outputs should be committed or ignored;
5. document any new provider or strategy extensions directly in this README.
