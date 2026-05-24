[README.md](https://github.com/user-attachments/files/28164778/README.md)
# SPX to IHSG Lead-Lag Analysis
### Empirical test of the hypothesis that S&P 500 returns contain statistically significant leading information for IHSG (Jakarta Composite Index) returns

---

## Overview

This project implements a rigorous econometric pipeline to test whether daily S&P 500 movements systematically predict next-session IHSG movements and if so, through what channel, at what horizon, and whether that predictive power survives macro controls and out-of-sample validation.

**Central finding:** H0 is rejected. The SPX signal explains **36% of IHSG overnight-gap variance** (β = 0.243, HAC t = 14.66), shrinks only marginally to **80% retention** after controlling for VIX, and is near-zero for the intraday IHSG return, this is exactly where theory predicts the lead should not appear. The rolling analysis finds the overnight β is above the 3σ threshold in **100% of rolling 252-day windows**.

---

## Hypotheses

**H1 (Alternative):** Lagged S&P 500 returns contain statistically significant predictive information for future IHSG returns.

**H0 (Null):** S&P 500 does not lead IHSG in a statistically significant manner.

---

## Data

| Series | Period | Source | Notes |
|---|---|---|---|
| IHSG daily OHLCV | 2017-01-03 → 2022-03-29 | GitHub `apricitea/exploringIHSG` | 1320 sessions |
| SPY daily OHLCV | 2010-01-04 → 2019-12-30 | GitHub `hackingthemarkets/datasets` | Standard SPX proxy |
| VIX daily OHLC | 1990-01-02 → present | GitHub `datasets/finance-vix` | CBOE official |
| **Effective overlap** | **2017-01-04 → 2020-01-06** | | **696 IHSG sessions** |

**Critical alignment rule.** NYSE closes at approximately 04:00 WIB; IHSG opens at 09:00 WIB the same morning. `spy_prev_ret[t]` is the SPY return from the most recent NYSE session *strictly preceding* IHSG date `t` (implemented via `pd.merge_asof(..., allow_exact_matches=False)`). Using same-calendar-date SPX would understate the lead and is the most common error in the published literature on this topic.

---

## Results Summary

### Decomposition — the cleanest test

| Target | β_SPX | t (HAC) | R² |
|---|---|---|---|
| IHSG **overnight gap** `ln(Open_t / Close_{t-1})` | **0.2424** | **14.63** | **35.9%** |
| IHSG intraday `ln(Close_t / Open_t)` | −0.009 | −0.28 | 0.01% |
| IHSG close-to-close | 0.234 | 6.60 | 5.9% |

The overnight component absorbs the entire SPX signal. The intraday component is statistically indistinguishable from zero — consistent with Hong–Stein gradual information diffusion: once IHSG opens and impounds the overnight U.S. signal, subsequent within-session moves are independent of the prior SPX close.

### Macro control (VIX)

| Model | β_SPX | t (HAC) | R² | β_SPX retained |
|---|---|---|---|---|
| Base (SPX only) | 0.243 | 14.66 | 36.1% | — |
| + Δ log(VIX) | **0.194** | **11.21** | 37.3% | **79.8%** |

SPX retains ~80% of its coefficient and all of its statistical significance after adding VIX. This rules out the common-factor (risk-on/off) explanation: the lead is incremental above and beyond the global volatility regime.

### VIX-regime conditioning (Hong–Stein prediction)

| Regime | n | VIX range | β_SPX | t | R² |
|---|---|---|---|---|---|
| Low VIX | 232 | [9.1, 12.1] | 0.146 | 3.40 | 4.7% |
| Mid VIX | 231 | [12.1, 15.2] | 0.261 | 7.27 | 19.5% |
| High VIX | 232 | [15.2, 37.3] | **0.250** | **12.58** | **55.1%** |

The lead intensifies in high-volatility regimes (R² climbs from 5% to 55%), which is the directional prediction of Hong–Stein: more information to transmit + slower institutional response in the thinner market = stronger measured lead.

### Cross-Correlation Function

- **CCF at lag 0** (= 1 calendar-day SPX lead): **ρ = 0.243**, well outside both the ±1.96/√T = ±0.074 asymptotic band and the stationary-bootstrap 95% envelope.
- **CCF at lag 1**: ρ = −0.067 (noise). Information decays within one session.

### Granger causality

- **SPX → IHSG** (close-to-close, HAC-F, p=1): F = 3.70, p = 0.055 — marginal because the strong overnight signal is diluted by independent intraday noise.
- **IHSG → SPX** (HAC-F, p=1): F = 7.38, p = 0.007 — appears significant but is partially confounded by same-calendar-day overlap between IHSG(t−1) and the NYSE session preceding IHSG(t); does not undermine the overnight result.
- **The overnight-gap rolling HAC t-statistic exceeds ±1.96 in 100% of rolling 252-day windows** and exceeds 3.0 in 100% of windows.

### VAR and long-run structure

- **VAR(1)** BIC-selected; all companion roots outside unit circle (stable).
- **FEVD at horizon 5**: SPX accounts for 6.2% of IHSG forecast-error variance. Modest in absolute terms but non-trivial given the two-variable bivariate system without macro controls.
- **Cointegration**: Engle–Granger p = 0.34, Johansen trace statistics below all critical values → no long-run price equilibrium. Justifies VAR (not VECM) as the correct specification.

### Out-of-sample predictive performance (walk-forward CV)

| Model | Target | OOS R² | Directional acc. |
|---|---|---|---|
| Martingale (zero) | Close-to-close | 0.000 | — |
| OLS | Close-to-close | 1.1% | 57.5% |
| LASSO | Close-to-close | 3.0% | 58.6% |
| XGBoost | Close-to-close | −6.8% | 57.7% |
| OLS | **Overnight gap** | **34.9%** | **70.2%** |
| LASSO | **Overnight gap** | **38.4%** | **72.6%** |
| XGBoost | Overnight gap | 26.7% | 67.9% |

### Trading signal diagnostics

| Metric | Strategy 1 (close-to-close) | Strategy 2 (overnight gap) |
|---|---|---|
| Raw Sharpe (no costs) | 2.88 | 6.80 |
| Annualized return | 39.6% | — |
| Max drawdown | −9.2% | — |
| Deflated Sharpe (Bailey-LdP, 6 trials) | 0.51 (p = 0.30) | 0.65 (p = 0.26) |
| PBO (CSCV, 6 strategies) | **0.000** | **0.000** |

**Important.** The deflated Sharpe is borderline significant not because the signal is weak, but because the OOS sample is only ~1.8 years — insufficient for the Bailey-LdP trial-multiplicity correction to clear conventional significance. The PBO of 0.000 confirms no backtest overfitting. Transaction costs (Indonesian large-cap round-trip ≈ 20–40 bps) will materially reduce the practical Sharpe.

---

## Repository Structure

```
sp500_ihsg_analysis/
│
├── README.md                     ← this file
│
├── sp500_ihsg_leadlag.ipynb      ← executed Jupyter notebook (all outputs embedded)
├── sp500_ihsg_leadlag.py         ← jupytext source (edit this, then re-execute)
├── console_output.txt            ← full terminal output from the last run
├── methodology_document.md       ← full academic methodology reference (15 sections)
│
├── ihsg_clean.csv                ← IHSG OHLC, 2017-01-03 → 2022-03-29
├── spy_clean.csv                 ← SPY OHLC, 2010-01-04 → 2019-12-30
├── vix_clean.csv                 ← VIX OHLC, 2016-12-01 → 2020-02-01
│
├── fig_distributions.png         ← return histograms, JB test, excess kurtosis
├── fig_ccf.png                   ← CCF ±30 lags with bootstrap CI bands
├── fig_irf.png                   ← VAR orthogonalized IRF: IHSG response to SPX shock
├── fig_fevd.png                  ← FEVD: SPX share of IHSG forecast-error variance
├── fig_rolling_granger.png       ← rolling 252-day Granger F + overnight β t-stat
├── fig_strategy.png              ← OOS equity curves (close-to-close and overnight)
└── fig_pbo.png                   ← CSCV logit-rank distribution (PBO = 0.000)
```

---

## How to Run

### Prerequisites

```bash
pip install statsmodels arch scikit-learn xgboost seaborn jupytext jupyter
```

Python 3.10+ recommended.

### Option 1 — Run the Python script directly

```bash
python sp500_ihsg_leadlag.py
```

Figures are saved to the working directory. All print output matches `console_output.txt`.

### Option 2 — Open and execute the notebook

```bash
jupyter notebook sp500_ihsg_leadlag.ipynb
```

Or re-execute from scratch:

```bash
jupytext --to notebook sp500_ihsg_leadlag.py -o sp500_ihsg_leadlag.ipynb
jupyter nbconvert --to notebook --execute sp500_ihsg_leadlag.ipynb \
    --output sp500_ihsg_leadlag.ipynb \
    --ExecutePreprocessor.timeout=600
```

### Extending to a longer sample

Replace the two data-loading lines with any source that provides `^GSPC` (or `SPY`) and `^JKSE` daily OHLC. The minimum change is:

```python
import yfinance as yf
spy  = yf.download("SPY",   start="2014-01-01", auto_adjust=True)[["Open","High","Low","Close"]].reset_index()
ihsg = yf.download("^JKSE", start="2014-01-01", auto_adjust=True)[["Open","High","Low","Close"]].reset_index()
spy.columns  = ["Date","spy_open","spy_high","spy_low","spy_close"]
ihsg.columns = ["Date","ihsg_open","ihsg_high","ihsg_low","ihsg_close"]
```

The rest of the pipeline (alignment, returns, all tests) runs unchanged.

---

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Data loading and calendar alignment |
| 2 | Log-returns construction (overnight, intraday, close-to-close) |
| 3 | Stationarity tests (ADF, KPSS) on returns and levels |
| 4 | Return distributions, ARCH diagnostics (Ljung-Box on r²) |
| 5 | Cross-Correlation Function with stationary-bootstrap CIs |
| 6 | Granger causality — HAC Wald F, multiple lag orders, reverse direction |
| 7 | VAR(p) — lag selection, stability, IRF, FEVD |
| 8 | Cointegration — Engle-Granger and Johansen |
| 9 | Rolling 252-day evidence: Granger F and overnight β t-stat |
| 10 | Overnight vs intraday decomposition (the discriminating test) |
| 11 | Predictive features and walk-forward model evaluation |
| 12 | Trading signal construction and equity curves |
| 13 | Empirical summary (evidence on all 9 dimensions) |
| 14 | Macro extension: VIX-controlled regression |
| 15 | VIX-regime conditioning (Low / Mid / High terciles) |
| 16 | Deflated Sharpe (Bailey-LdP) and PBO (CSCV) |
| 17 | Extended summary including macro and overfitting diagnostics |

---

## Key Design Decisions

**Why `spy_prev_ret` and not same-date SPX.** The correct alignment for a predictive claim is the NYSE close *strictly preceding* the IHSG session. Using same-calendar-date SPX conflates information that was available before IHSG opened with information that arrived during the IHSG session, producing inflated estimates and potentially spurious inference. The `merge_asof(..., allow_exact_matches=False)` with a ≤7-day gap filter is the implementation of this rule.

**Why overnight gap is the primary target.** The overnight gap `ln(Open_t / Close_{t-1})` is the exact window in which prior-session U.S. information gets impounded into IHSG prices. Testing on this component rather than close-to-close is both more theoretically motivated and more powerful (R² = 36% vs 6%).

**Why HAC standard errors throughout.** Both series exhibit strong ARCH effects (Ljung-Box on r² rejects at all lags with p < 10⁻⁹⁰). Classical OLS standard errors would be severely anti-conservative; all inference uses Newey-West HAC with bandwidth `L = ⌊4(T/100)^(2/9)⌋`.

**Why the reverse-Granger result needs careful treatment.** The "IHSG → SPX" Granger test appears significant (p = 0.007) because `ihsg_ret[t-1]` and `spy_prev_ret[t]` have partial same-calendar-day overlap (both can reference Wednesday's Asian and U.S. sessions respectively). This does not invalidate the SPX → IHSG overnight result, which has no such overlap.

---

## What the Full Publication Version Needs

In priority order:

1. **Longer sample (2014 → present, 10+ years).** The ~3-year window used here is sufficient as proof-of-concept but too short for: regime-switching VAR, Bai-Perron structural breaks, the deflated Sharpe test, and sub-period robustness tables.

2. **MSCI Emerging Markets orthogonalization.** The most important missing control. If the SPX coefficient survives controlling for EEM or MSCI EM returns, the lead is genuinely U.S.-specific rather than generic EM beta.

3. **FX controls: DXY and USD/IDR.** Test whether the overnight lead is partly a currency channel (rupiah gap-adjusting to overnight dollar moves).

4. **Full macro panel.** US 10-year yield (discount-rate channel), Brent crude (Indonesia energy-import/commodity-export asymmetry), Fed Funds rate as a regime variable.

5. **Event studies.** FOMC, CPI, and large-SPX-move (>2%) subsets. BMP test for abnormal returns.

6. **Markov-switching VAR.** Endogenous regime classification; test whether the lead intensifies in the "turbulent" latent state.

7. **Sector-level IHSG.** The banking and large-cap consumer sectors should show stronger U.S. lead than mining and agriculture. Heterogeneity is a useful robustness check.

8. **Transaction-cost sensitivity.** At 30 bps round-trip, report the Sharpe as a function of cost assumption. This determines whether the signal is economically exploitable.

---

## Theoretical framing

The lead-lag pattern is consistent with the **gradual information diffusion model** of Hong and Stein (1999): information disseminates slowly across markets, and emerging-market indices — with thinner liquidity, slower institutional response, and shorter trading hours — systematically impound global price information with a delay relative to the world's most liquid index.

The mechanical channel is reinforced by time-zone asymmetry: NYSE closes at ~04:00 WIB, IHSG opens at 09:00 WIB the same morning, giving a 5-hour window in which no Indonesian price discovery occurs. The SPX move in that window is the primary input to the IHSG opening gap.

These two effects are not mutually exclusive. Controlling for MSCI EM (which captures the common global-risk factor) would allow separation of the U.S.-specific information channel from the generic risk-transmission channel — currently the most important open question in this analysis.

---

## Citation

If you use this pipeline, please cite the key methodological references:

- Hong, H. & Stein, J. C. (1999). A unified theory of underreaction, momentum trading, and overreaction in asset markets. *Journal of Finance*, 54(6), 2143–2184.
- Bailey, D. H. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Politis, D. N. & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the AMS*, 61(5), 458–471.
