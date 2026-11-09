# PJM Power Market Analysis & Trading Research

> [!IMPORTANT]
> **Data disclaimer.** No PJM data, or outputs derived from it, are published in this repository:
> no raw or processed data, CSV files, charts, or notebook outputs. PJM Data Miner data is licensed for
> internal use only. Everything can be regenerated locally with your own free PJM API key. See
> [Data](#data) and [Reproduction](#reproduction).

Quantitative research on the PJM wholesale electricity market. It asks **what market conditions are
associated with large differences between day-ahead (DA) and real-time (RT) prices at PJM Western Hub**,
and whether information available *before* the day-ahead auction can be turned into a trading signal.

The goal is a disciplined research workflow, not a production strategy:

> market question → data pipeline → exploratory analysis → features → models → signal → backtest → robustness

**Status:** stages 1–6 complete. The strategy was frozen and committed **before** the test period
(2025 H2) was opened, then evaluated on it once. Robustness checks (stage 7) and the single run on the
2026 holdout are next.

## Key findings so far

* **RT usually clears slightly below DA, but rare spikes cancel that out on average.** In 2023 – mid-2025,
  RT was below DA in about two-thirds of hours (median spread ≈ −$2.6/MWh), yet the mean spread is not
  distinguishable from zero. A small number of spike hours carry about a third of all positive spread dollars.
* **Spread risk is predictable before the auction; spread direction is not.** A model using only
  pre-auction information ranks hours by risk well: on held-out data the riskiest decile averaged about
  14× the absolute spread of the calmest. Direction models are barely better than a coin flip (ROC-AUC ≈ 0.53).
* **Load forecast surprises explain the spread's direction, but only after the fact.** When load came
  in well above the day-ahead forecast, RT cleared above DA most of the time; when it came in well below,
  rarely. The relationship is monotonic and stable across years.
* **Net of trading costs, the spread looks close to efficiently priced.** Out of sample, the average spread
  was smaller than an assumed $0.50/MWh cost. Filtering trades by predicted risk did not create an edge.
  Only a strict "short when the model is most confident RT < DA" rule was marginally positive, and it was
  not consistent across periods. That rule was frozen as the strategy, with expectations written down in advance.
* **On the untouched test period, the frozen strategy made money, but mostly because the market moved
  its way.** In 2025 H2, RT ran persistently below DA (average −$2.7/MWh, against about −$0.3 before), so
  every short-side strategy profited and always-long lost heavily. The frozen strategy earned more per MWh than
  simply shorting every hour, with a smaller drawdown, but less in total. Its average daily P&L was not
  statistically different from zero, and most of it came from five days. Conclusion: no robust edge beyond the
  market's own drift; the model's contribution was better hour selection.

## Why power markets

Electricity has to be balanced in real time, is hard to store, depends heavily on weather and load, and is
priced at thousands of locations through separate day-ahead and real-time settlements. In PJM, a
**virtual trader** can effectively take the DA/RT spread: a *DEC* bid buys in the day-ahead market and sells
back in real time (earning RT − DA), and an *INC* offer does the reverse (earning DA − RT). This project
asks whether that spread is predictable from public information available before the day-ahead bid
deadline.

## Research question

```
spread_t = RT_LMP_t − DA_LMP_t        (hourly, PJM Western Hub)
```

A positive spread means real-time cleared above day-ahead. The project relates the spread to load
forecast error, demand level, time of day, season and recent volatility, then tests look-ahead-free
trading signals on held-out data.

## Data

| Dataset | PJM Data Miner feed | Scope |
|---|---|---|
| Day-ahead hourly LMP | `da_hrl_lmps` | Western Hub (pnode 51288); all 12 PJM hubs downloaded |
| Real-time hourly LMP (settlement-verified) | `rt_hrl_lmps` | Western Hub; all 12 PJM hubs downloaded |
| Historical load forecast | `load_frcstd_hist` | RTO, all forecast vintages |
| Hourly metered load | `hrl_load_metered` | RTO |

| Period | Dates | Used for |
|---|---|---|
| Lookback | Dec 2022 | History for rolling features only |
| Train | 2023 – 2024 | Fitting models and every threshold |
| Validation | 2025 H1 | Comparing models; part of signal selection |
| Test | 2025 H2 | **Opened once**, after the strategy was frozen |
| Final holdout | Jan – Oct 2026 | Kept separate; run once at the very end |

**Data is not included in this repository.** PJM Data Miner data is for internal use and may not be
redistributed without a PJM licence. See [Reproduction](#reproduction) to download it yourself.

## How the project works, stage by stage

### 1. Data pipeline: `src/data/`
* **API client** (`pjm_client.py`, `download.py`): pages through the PJM Data Miner API, throttles to the
  non-member limit of 6 requests/minute, retries failures, and caches each year so nothing is downloaded
  twice. It also handles a PJM quirk: data older than about two years sits in an archive with different query
  rules, and the archive boundary moves with the clock.
* **Cleaning** (`clean.py`): joins the four feeds on UTC hour, derives market time in Eastern time, and
  handles daylight saving explicitly (23- and 25-hour days). It picks the load-forecast vintage that was
  actually available at decision time and writes one validated hourly table.
* **Validation and audit** (`notebooks/01_data_validation.ipynb`): checks for duplicates, missing hours and
  DST day lengths, confirms LMP components add up, and checks for hidden hour shifts between feeds. Annual
  peak loads match PJM's reported peaks, and a fresh API query matched the stored prices exactly.

### 2. Exploratory market analysis: `src/analysis/`, notebooks 02–03
Distribution of the spread; when it is volatile (hour, season, weekday, load regime); how it relates to
load forecast error; and what extreme-price events look like. Only train + validation data is used, and every
threshold (load regimes, extreme-event cut-offs) is fitted on training data, so nothing learned here leaks
into the test period.

### 3. Feature engineering: `src/data/features.py`
31 features that are all **known at decision time** (calendar; D-1 load forecast level, shape and change;
recent forecast bias; DA prices for D-1; RT prices and spreads up to D-2; trailing spread and volatility
windows). Every column is registered as `tradable` or `ex_post`, and models can only see tradable ones. A
**look-ahead test** scrambles all information that was not yet public at decision time and checks that no
feature changes. A separate, deliberately naive re-implementation reproduces every feature value.

### 4. Modeling: `src/models/`, notebook 04
Three questions, each against a constant baseline: **direction** (logistic regression; gradient boosting
for comparison), **magnitude** (OLS / ridge), and **risk** (ridge on log |spread|). Hyperparameters are
chosen on expanding-window walk-forward folds inside the training period; models are compared on validation;
splits are always chronological. Direction and magnitude show no useful skill; risk is clearly predictable
and stable across every period.

### 5. Trading signal: `src/backtest/`, `config/strategy.yaml`
Seven candidate signals, each with a stated purpose: flat, always long, always short (the base-rate
benchmark), a domain rule (stay out of spike-prone hours), the plan's probability-threshold model signal, a
risk-filtered short, and a risk-sized short. Thresholds were selected on 18 months of out-of-sample
predictions with a criterion declared before the run (Sharpe-like ratio of daily P&L net of cost, minimum
coverage). The selected strategy, every alternative's settings, the refit models' cut-offs and the cost
assumption were **saved in `config/strategy.yaml` before the test period was opened**, and committed so there
is a timestamped record. The backtest reads everything from that file.

### 6. Backtest: `src/backtest/run.py`, `metrics.py`, notebook 05
Every frozen signal is evaluated **once** on the test period (2025 H2). Hypothetical P&L = position ×
(RT − DA) − cost × |position|, at 1 MW per hour. Metrics: net P&L, P&L per MWh, hit rate, Sharpe-like
ratio, maximum drawdown, profit concentration, a bootstrap confidence interval for mean daily P&L, and
breakdowns by month, hour block and load regime. Results are compared against the benchmarks (flat, always
long, always short) as well as each other.

### 7. Robustness
Cost sensitivity ($0–2/MWh), threshold sensitivity, stability across sub-periods and regimes, and
feature-group ablation. Then a single run on the 2026 holdout.

## Safeguards against fooling ourselves

* **Decision time:** a spread position for operating day *D* is fixed by **10:30 EPT on D−1**, before the
  day-ahead auction closes. Features may use the D-1 morning load forecast, DA prices up to D-1, and RT
  prices / actual load only up to D-2 (complete days). The DA price for *D* is treated as unknown.
* **Timestamps:** joins use PJM's UTC hour; spring-forward days have 23 rows and fall-back days 25. PJM's
  Eastern-time labels are cross-checked against the conversion on every row.
* **Missing forecasts are not filled.** PJM published no D-1 forecast for a few days (0.7% of hours);
  filling them with a later forecast would be look-ahead, so those hours simply get no position.
* **No random splits, no tuning on test.** Train → walk-forward folds → validation → frozen strategy →
  test once → holdout once.
* **Benchmarks and costs everywhere:** every model is compared with a constant baseline, and every signal
  with flat / always-long / always-short, net of an explicit cost assumption.
* **Tests:** 80 offline tests (no API calls) cover DST handling, forecast-vintage selection, the
  look-ahead guarantee, chronological splits, signal logic and running saved strategy files end to end. Use them as your own choice. 

## Limitations

* About 2.5 years of in-sample data; extreme events cluster on a few days, so tail statistics are fragile.
* One pricing location (Western Hub) and RTO-level load; no weather, outage, fuel-price or congestion data.
* The backtest is a research simplification: price-taker, virtual bids always clear, hourly (not
  5-minute) settlement, and a single assumed cost of $0.50/MWh for fees and uplift (sensitivity tested).
* Findings are associations, not causal claims.

## Repository structure

```
config/
  config.example.yaml   settings template (copy to config.yaml and add a PJM API key)
  strategy.yaml         frozen trading strategy (committed before the test period)
src/
  config.py             config loader
  data/                 pjm_client.py, download.py, clean.py, features.py
  analysis/             spreads.py, regimes.py, diagnostics.py, plots.py
  models/               split.py, common.py, classifier.py, baseline.py, selected.py
  backtest/             signals.py, engine.py, selection.py, strategy.py, metrics.py, run.py
notebooks/
  01_data_validation    data checks and audit
  02_market_exploration price and load context
  03_spread_analysis    spread distribution, volatility, forecast error, extreme events
  04_modeling           direction, magnitude and risk models
  05_backtest           frozen strategy vs benchmarks on the test period
tests/                  pytest suite (offline)
data/, reports/         local outputs (git-ignored: derived from PJM data)
```

## Reproduction

1. Python 3.11+, then `pip install -r requirements.txt`.
   To keep notebook outputs (which contain PJM data) out of git: `python -m nbstripout --install`.
2. Get a free PJM Data Miner API key: create a PJM Tools account, then subscribe at apiportal.pjm.com.
   Non-member accounts must first email PJM to confirm internal use (see PJM's Data Miner API guide).
   They are limited to 6 requests/minute; the client throttles automatically.
3. `copy config\config.example.yaml config\config.yaml` and set `pjm.api_key` (or set `PJM_API_KEY`).
4. Run the pipeline:
   ```
   python -m src.data.download      # raw data, cached per year (~45 min the first time)
   python -m src.data.clean         # validated hourly table + validation report
   python -m src.data.features      # tradable features
   python -m pytest                 # 80 offline tests
   python -m src.backtest.run       # evaluate the saved strategy on the test period
   ```
5. Open the notebooks in order (01 → 05) to reproduce the analysis and models.
6. `config/strategy.yaml` holds the strategy used for the reported results. To try your own, save it to a new
   file and evaluate it:
   ```
   python -m src.backtest.strategy --out config/my_strategy.yaml
   python -m src.backtest.run --strategy config/my_strategy.yaml
   ```

> [!NOTE]
> **Research discipline (not enforced by code).** The test period and the 2026 holdout are only meaningful if
> they are looked at *after* a strategy is saved. Freeze your strategy first, ideally commit it so there is a
> dated record, and don't change it after seeing test results; any new idea counts as new research and needs
> fresh, unseen data to evaluate.
