# A Methodological Framework for Testing the S&P 500 → IHSG Lead-Lag Hypothesis

**Scope.** Econometric design document for a quantitative study of whether U.S. equity returns (S&P 500, SPX) contain statistically significant leading information for the Jakarta Composite Index (IHSG / JKSE) at the daily frequency. The framework is written to be implementation-ready in Python, robust to standard time-series pitfalls, and suitable for academic publication or hedge-fund signal research.

---

## 1. Research Objectives and Hypotheses

### 1.1 Formal hypotheses

Let $r^{SPX}_t$ and $r^{IHSG}_t$ denote daily log returns. Define an information set $\mathcal{F}_{t-1}^{IHSG}$ containing the full history of IHSG returns up to $t-1$, and an augmented set $\mathcal{F}_{t-1}^{IHSG} \cup \mathcal{F}_{t-1}^{SPX}$.

- **H0 (No predictive lead):** $\mathbb{E}[r^{IHSG}_t \mid \mathcal{F}_{t-1}^{IHSG}] = \mathbb{E}[r^{IHSG}_t \mid \mathcal{F}_{t-1}^{IHSG} \cup \mathcal{F}_{t-1}^{SPX}]$
- **H1 (Predictive lead exists):** The conditional expectation strictly differs once SPX history is included, in a way that is statistically significant after controlling for global macro state.

### 1.2 Decision objects

The study must produce defensible answers to four operational questions:

1. **Existence.** Is there a non-zero, statistically significant lead from SPX to IHSG?
2. **Horizon.** What is the optimal lag $k^\ast$ (in trading days) that maximizes information transfer?
3. **Stability.** Is the lead-lag structure stable across volatility regimes, monetary regimes, and structural breaks?
4. **Incrementality.** Does the SPX signal survive after orthogonalizing against macro controls (DXY, USD/IDR, VIX, US10Y, oil, gold, MSCI EM)?

### 1.3 Why "correlation" is insufficient

Contemporaneous correlation conflates three channels: (i) genuine information transmission, (ii) common-factor exposure (global risk-on/off), and (iii) overnight-gap mechanics arising from non-overlapping trading sessions. A lead-lag claim requires temporal asymmetry — i.e., evidence that $\text{Corr}(r^{SPX}_{t-k}, r^{IHSG}_t)$ for $k > 0$ is significantly larger than the symmetric correlation $\text{Corr}(r^{SPX}_{t+k}, r^{IHSG}_t)$ — plus an instrumental story for why that asymmetry exists.

---

## 2. Theoretical Background

The hypothesis is grounded in three well-established mechanisms:

- **Heterogeneous information processing (Hong & Stein, 1999).** Information diffuses gradually across markets; emerging markets, with thinner liquidity and slower institutional response, tend to underreact intraday and complete the price adjustment in subsequent sessions.
- **Time-zone asymmetry.** IHSG trades roughly 12 hours ahead of NYSE close. The U.S. close at 04:00 WIB (Western Indonesia Time) precedes the IHSG open at 09:00 WIB the same morning. Mechanically, any U.S. news after IHSG close is impounded into IHSG with at least a one-session delay.
- **Common factor exposure.** Both indices load on a global risk factor (often proxied by VIX or DXY). The SPX, as the deepest and most liquid index, often serves as the *measurement vehicle* for that factor, making it a natural Granger-causal candidate even absent direct fundamental linkage.

The empirical question is whether, after controlling for those mechanical channels, a *genuine* SPX → IHSG predictive component remains.

---

## 3. Data Requirements

### 3.1 Primary series

| Series | Source | Frequency | Notes |
|---|---|---|---|
| IHSG OHLCV | IDX / Bloomberg / Yahoo `^JKSE` | Daily | Use official IDX close; reconcile against Bloomberg if available |
| S&P 500 OHLCV | Bloomberg / Yahoo `^GSPC` | Daily | Use NYSE close |
| Sample window | 2014-01-01 → present | ≥ 10 years | Spans 2015 EM selloff, 2018 trade war, 2020 COVID, 2022 tightening |

### 3.2 Macro controls — rationale per variable

- **VIX (`^VIX`).** Global risk aversion proxy; conditions both equity returns. Excluding it risks attributing common-shock comovement to SPX → IHSG causation.
- **DXY.** USD strength is mechanically tied to EM equity flows (carry trade, dollar funding stress).
- **USD/IDR.** Local FX channel; rupiah weakness frequently coincides with IHSG selloffs via foreign-investor outflow.
- **US 10-Year Treasury yield.** Discount-rate channel and global duration risk. Strong predictor of EM equity drawdowns during tightening cycles.
- **Fed Funds Rate (or effective Fed Funds).** Slower-moving regime variable; better used as a regime classifier than as a daily regressor.
- **Brent crude.** Indonesia is a net energy *importer* on net oil and a major coal/CPO *exporter*. Brent has asymmetric and sectoral effects on IHSG.
- **Gold (XAU/USD).** Safe-haven and IDR-hedge channel; relevant for risk-off dynamics.
- **MSCI Emerging Markets (`^MXEF` or `EEM`).** Critical orthogonalization variable: if SPX → IHSG disappears after controlling for MSCI EM, the "lead" is just generic EM beta, not specifically a U.S. effect.

### 3.3 Calendar alignment and missing-data protocol

This is where most lead-lag studies fail. Protocol:

1. **Build a master trading calendar** from the *union* of NYSE and IDX trading days, then take the *intersection* for the primary regression sample. Keep the union for event-study extensions.
2. **Index everything to IDX trading dates** for IHSG-as-dependent-variable specifications, since the prediction target is an IHSG-tradable session.
3. **Define the SPX-lag-1 observation carefully.** When IHSG opens on date $t$ (Asia morning), the most recent SPX close is from date $t-1$ (NYSE evening, calendar same date but session prior). This means the "natural" lag-1 SPX is actually the *previous calendar day's* NYSE close. Pseudocode:

   ```
   For each IHSG trading date t_IDX:
       SPX_lag1 = SPX close on the most recent NYSE session
                  strictly preceding t_IDX in wall-clock time
   ```

   This alignment is essential and is a frequent source of spurious lead-lag findings.

4. **Missing data.** Do **not** forward-fill prices for return calculation; this manufactures zero returns and shrinks variance. Either (a) drop the date pair, or (b) for control variables (yields, FX) carry-forward is acceptable since they are level series, not return series. State the choice explicitly in the paper.
5. **Holiday asymmetries.** When IHSG is closed but NYSE is open, cumulate SPX returns over the closed Indonesian window and assign that cumulative return as the relevant lag at IHSG reopen. This handles Idul Fitri / Lebaran multi-day closures cleanly.

### 3.4 Returns specification

Use **log returns**:

$$r_t = \ln(P_t / P_{t-1}) = \ln(P_t) - \ln(P_{t-1})$$

Justification: time-additivity across multi-day holiday windows, symmetric treatment of gains/losses, conventional in the empirical asset-pricing literature. Simple/percentage returns may be reported in robustness tables but should not be the primary specification.

For currency-adjusted IHSG returns (robustness): $r^{IHSG,USD}_t = r^{IHSG}_t + r^{IDR/USD}_t$ (approximation valid for small returns; use exact compounding for large moves).

---

## 4. Exploratory Analysis

### 4.1 Required diagnostics

| Diagnostic | Purpose |
|---|---|
| Return histogram + Q-Q plot | Identify fat tails; informs choice of robust SE / Student-t innovations |
| Jarque-Bera / Anderson-Darling | Formal normality test |
| ACF / PACF of $r_t$ and $r_t^2$ | Detect autocorrelation and volatility clustering |
| Ljung-Box on $r_t^2$ | Confirm ARCH effects |
| Rolling 60-day / 252-day correlation $\rho(r^{SPX}_{t-1}, r^{IHSG}_t)$ | Time-variation in the lead-lag |
| Stationarity tests (§4.2) | Required before VAR / regression |

### 4.2 Stationarity testing protocol

Run all three on **prices** (expect non-stationarity) and **returns** (expect stationarity):

1. **Augmented Dickey-Fuller (ADF).** $H_0$: unit root. Use AIC-selected lag, with constant only for return series and constant+trend for price series.

   $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

   Reject $H_0$ if $\gamma < 0$ significantly.

2. **KPSS.** $H_0$: stationarity. Use as a confirmatory complement to ADF — agreement between the two is stronger evidence than either alone.
3. **Phillips-Perron.** Robust to heteroskedasticity and serial correlation in $\epsilon_t$; useful given known ARCH behavior in financial returns.

Why this matters: Granger causality on non-stationary series produces spurious results. The standard solution — difference until stationary — is the reason we work in returns rather than prices.

### 4.3 Why raw price correlation is dangerous

Two random walks with no causal link can produce $|\rho| > 0.9$ over arbitrary windows; this is the canonical "spurious regression" of Granger and Newbold (1974). All inference must be conducted on stationary transformations (returns or, for cointegration analysis, the residual of a long-run price relationship — see §5.4).

---

## 5. Lead-Lag Statistical Testing

### 5.1 Cross-Correlation Function (CCF)

For lags $k \in \{-30, \ldots, +30\}$ trading days:

$$\hat\rho_{XY}(k) = \frac{\sum_t (r^{SPX}_{t-k} - \bar r^{SPX})(r^{IHSG}_t - \bar r^{IHSG})}{\sqrt{\sum_t (r^{SPX}_{t-k} - \bar r^{SPX})^2 \cdot \sum_t (r^{IHSG}_t - \bar r^{IHSG})^2}}$$

**Interpretation.** Under our alignment convention:
- $k > 0$ → SPX leads IHSG by $k$ days (the hypothesis of interest).
- $k < 0$ → IHSG leads SPX (would be surprising and worth investigating).
- $k = 0$ → contemporaneous (largely driven by overnight-gap mechanics).

**Confidence bands.** Under the null of independent series, asymptotic bands are $\pm 1.96 / \sqrt{T}$. For financial data this is too liberal — fat tails and ARCH inflate Type-I error. Use **bootstrap confidence intervals**: block-bootstrap with block length $\ell \approx T^{1/3}$ (stationary bootstrap of Politis & Romano, 1994), 2000 replications.

### 5.2 Granger Causality

Estimate two restricted/unrestricted pairs:

- **Restricted:** $r^{IHSG}_t = \alpha + \sum_{i=1}^{p} \beta_i r^{IHSG}_{t-i} + u_t$
- **Unrestricted:** $r^{IHSG}_t = \alpha + \sum_{i=1}^{p} \beta_i r^{IHSG}_{t-i} + \sum_{j=1}^{p} \gamma_j r^{SPX}_{t-j} + u_t$

Test $H_0: \gamma_1 = \gamma_2 = \cdots = \gamma_p = 0$ via F-statistic:

$$F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T - 2p - 1)} \sim F(p, T-2p-1)$$

Run for $p \in \{1, 2, 3, 5, 10, 20\}$. Report **HAC-adjusted (Newey-West)** F-tests because residuals exhibit conditional heteroskedasticity.

**Limitations of Granger causality** — state these explicitly in the paper:

- Granger causality is *predictive*, not *structural* causality. A significant result means "SPX history improves linear forecasts of IHSG," not "SPX moves cause IHSG to move."
- Vulnerable to omitted-variable bias: if a third factor (e.g., global liquidity) drives both, SPX may appear to Granger-cause IHSG simply by being the faster-reacting series.
- Linear; misses nonlinear dependence (addressed via transfer entropy, §5.6).

### 5.3 Vector Autoregression (VAR)

Estimate $\mathbf{y}_t = [r^{SPX}_t, r^{IHSG}_t]'$ as:

$$\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{y}_{t-i} + \boldsymbol\epsilon_t, \quad \boldsymbol\epsilon_t \sim (\mathbf{0}, \boldsymbol\Sigma)$$

**Lag selection.** Compare AIC, BIC (SBIC), and HQIC for $p \in \{1, \ldots, 20\}$. BIC tends to prefer parsimonious models; AIC tends toward larger $p$. Report both and justify the chosen $p^\ast$.

**Stability check.** All eigenvalues of the companion matrix must lie inside the unit circle. If not, the VAR is non-stationary and the IRF is not meaningful.

**Impulse Response Function (IRF).** Trace the response of $r^{IHSG}_{t+h}$ to a unit (or one-SD) orthogonalized shock to $r^{SPX}_t$ for $h \in \{0, \ldots, 20\}$. Use **Cholesky ordering** with SPX first (justified by time-zone precedence). Report 95% confidence bands via bootstrap (1000 replications).

**Forecast Error Variance Decomposition (FEVD).** What fraction of $h$-step-ahead IHSG forecast error variance is attributable to SPX shocks? If this fraction is non-trivial (say, > 15%) at $h = 1, 2, 3$, the lead is economically meaningful.

### 5.4 Cointegration (long-run equilibrium)

Returns are stationary, but if log-prices share a common stochastic trend, a cointegration vector exists.

**Engle-Granger two-step.**
1. Regress $\ln P^{IHSG}_t = \alpha + \beta \ln P^{SPX}_t + u_t$.
2. Test $u_t$ for a unit root via ADF. Reject $\Rightarrow$ cointegration.

**Johansen test (preferred for system inference).** Estimate a VECM:

$$\Delta \mathbf{y}_t = \boldsymbol\Pi \mathbf{y}_{t-1} + \sum_{i=1}^{p-1} \boldsymbol\Gamma_i \Delta \mathbf{y}_{t-i} + \boldsymbol\epsilon_t$$

The rank of $\boldsymbol\Pi$ equals the number of cointegrating vectors. Use both the **trace** and **maximum eigenvalue** statistics.

**Interpretation note.** A priori, we expect cointegration to be weak: SPX and IHSG are denominated in different currencies and serve different real economies. A finding of *no* cointegration is the more likely and economically defensible result; this matters because it justifies running the lead-lag analysis purely in return-space (i.e., VAR rather than VECM).

### 5.5 Dynamic Time Warping (DTW)

DTW measures shape similarity allowing local time stretches:

$$DTW(X, Y) = \min_{\pi} \sqrt{\sum_{(i,j) \in \pi} (x_i - y_j)^2}$$

subject to monotonic, continuous warping path $\pi$. Use **constrained DTW** with a Sakoe-Chiba band of width $\pm 30$ days to prevent pathological warps.

**Use case.** Best applied to **cumulative log-return paths** over rolling 60–120 day windows. Yields an "average local lag" interpretation that is robust to small phase distortions in volatile regimes. Compare against random-pair DTW distances (bootstrap from shuffled series) to establish significance.

**Caveat.** DTW does not test causality; it quantifies geometric similarity under temporal warping. Treat as supporting evidence, not as a stand-alone inference tool.

### 5.6 Transfer Entropy

Captures **nonlinear** information flow:

$$T_{X \to Y}(k, \ell) = \sum p(y_{t+1}, y_t^{(\ell)}, x_t^{(k)}) \log \frac{p(y_{t+1} \mid y_t^{(\ell)}, x_t^{(k)})}{p(y_{t+1} \mid y_t^{(\ell)})}$$

where $x_t^{(k)} = (x_t, \ldots, x_{t-k+1})$.

Interpretation: reduction in uncertainty about $y_{t+1}$ when including the history of $x$, beyond what is already explained by the history of $y$.

**Implementation.** Use symbolic transfer entropy (Staniek & Lehnert, 2008) to avoid the curse of dimensionality of kernel density estimates. Establish significance by shuffling $x$ and recomputing 500 times → empirical null distribution → p-value.

**Comparison protocol.** Report transfer entropy alongside Granger causality. Concordance strengthens the conclusion; if Granger is significant but transfer entropy is not, the lead-lag is captured by linear projection and there is no additional nonlinear signal (a useful finding, not a contradiction).

---

## 6. Regime-Based Analysis

### 6.1 Regime definitions

Define ex-ante, before running any tests:

| Regime axis | Operationalization |
|---|---|
| Equity bull/bear | 200-day SMA of SPX; bear if SPX < 0.8 × 52-week high |
| Volatility | VIX < 15 (low), 15–25 (mid), > 25 (high) |
| Monetary | Fed funds rising / on hold / cutting (rolling 90d slope) |
| Crisis windows | 2015–16 EM selloff, 2018 trade war, 2020 COVID, 2022 tightening |

### 6.2 Tests within regimes

For each regime $R$:
- Subset-sample Granger causality with HAC SE.
- Subset CCF peaks: $k^\ast_R = \arg\max_k \hat\rho_{XY}(k \mid R)$.
- Test equality of $k^\ast$ and peak $\rho$ across regimes via bootstrap.

### 6.3 Markov regime-switching VAR

Estimate a two-state MS-VAR (Hamilton, 1989):

$$\mathbf{y}_t = \mathbf{c}_{S_t} + \sum_{i=1}^{p} \mathbf{A}_{i, S_t} \mathbf{y}_{t-i} + \boldsymbol\epsilon_t, \quad \boldsymbol\epsilon_t \sim N(\mathbf{0}, \boldsymbol\Sigma_{S_t})$$

with $S_t \in \{1, 2\}$ a hidden Markov chain. Smoothed state probabilities $P(S_t = j \mid \mathcal{F}_T)$ provide an endogenous regime classification — typically interpreted ex-post as "calm" and "turbulent."

**Key inferential output.** The ratio of off-diagonal coefficients $(\mathbf{A}_{i,1})_{21}$ vs $(\mathbf{A}_{i,2})_{21}$ tells you whether the SPX → IHSG lead intensifies in the turbulent regime — a well-documented stylized fact in EM literature.

### 6.4 Structural break detection

- **Bai-Perron (2003)** multiple structural break test on the Granger regression coefficients.
- **Quandt-Andrews / Sup-F** for a single unknown break date.
- Cross-validate breakpoints against known macro events (2008 GFC, 2013 Taper Tantrum, 2020 COVID, 2022 Fed pivot).

### 6.5 Rolling-window analysis

Estimate Granger F-statistic and peak CCF lag over rolling 252-day windows. Plot the time series of:
- F-statistic with 5% critical value reference line,
- Optimal lag $k^\ast_t$,
- Maximum cross-correlation $\rho^\ast_t$.

This produces the single most communicative chart for the paper: *the SPX → IHSG lead is not a constant — it strengthens in stress, weakens in calm*.

---

## 7. Predictive Modeling Layer

### 7.1 Why predictive modeling matters

Statistical significance ≠ economic significance. A predictor can be Granger-significant but produce negligible $R^2$ once realistic costs are imposed. The predictive layer answers: *can this lead-lag be turned into a forecast or a tradable signal?*

### 7.2 Feature engineering

**Target.** $y_t = r^{IHSG}_t$ (open-to-close) or, for tradability, $r^{IHSG, \text{overnight}}_t = \ln(O^{IHSG}_t / C^{IHSG}_{t-1})$ — the overnight gap is the most directly hypothesis-relevant target.

**Features.**
- Lagged SPX returns: $r^{SPX}_{t-1}, r^{SPX}_{t-2}, \ldots, r^{SPX}_{t-5}$
- SPX rolling realized volatility: 5d, 21d
- SPX drawdown from rolling 60d high
- SPX momentum: $r^{SPX}_{[t-21, t-1]}$
- VIX level and change
- VIX × $r^{SPX}_{t-1}$ interaction
- Macro: $\Delta$DXY, $\Delta$US10Y, $\Delta$USD/IDR, $r^{Brent}_{t-1}$, $r^{Gold}_{t-1}$, $r^{MSCI EM}_{t-1}$
- IHSG own lags as controls (so the SPX coefficient is *incremental*)

### 7.3 Models

| Model | Role |
|---|---|
| OLS with Newey-West SE | Baseline, interpretable, paper-friendly |
| LASSO / Elastic Net | Sparse selection; reveals which lags survive penalization |
| Ridge | Stabilizes when features are correlated (likely) |
| Random Forest | Captures nonlinearities and interactions |
| XGBoost | State-of-the-art tabular; report SHAP for interpretability |
| LSTM (optional) | Sequence model; only if sample is large enough and gain over XGBoost is justified |

### 7.4 Validation — non-negotiable rules

- **Walk-forward (expanding window) cross-validation.** Initial train 2014–2018, predict 2019; expand and re-fit annually. **Never** use shuffled k-fold on time series.
- **Purged & embargoed CV** (López de Prado, 2018) if the target window overlaps with feature windows.
- **Strict temporal split:** every feature for predicting $r^{IHSG}_t$ must use information available *strictly before* the IHSG open on $t$.
- **No data leakage from look-ahead normalization.** Rolling z-scores / RobustScaler must be fit on the trailing window only.
- Set seeds and log them. Report mean ± SD over ≥ 5 seeds for any stochastic model.

### 7.5 Metrics

| Metric | Formula / definition | Interpretation |
|---|---|---|
| Out-of-sample $R^2_{OOS}$ | $1 - \frac{\sum (r_t - \hat r_t)^2}{\sum (r_t - \bar r_{\text{train}})^2}$ | Compare against in-sample-mean benchmark, not zero |
| RMSE | $\sqrt{\frac{1}{T}\sum (r_t - \hat r_t)^2}$ | Magnitude error |
| MAE | $\frac{1}{T}\sum \|r_t - \hat r_t\|$ | Robust to outliers |
| Directional accuracy | $\frac{1}{T}\sum \mathbb{1}[\text{sign}(r_t) = \text{sign}(\hat r_t)]$ | The single most important number for trading |
| Diebold-Mariano | Equal-predictive-accuracy test vs benchmark | Statistical comparison of forecasts |
| Sharpe of long/short strategy | $\sqrt{252} \cdot \bar r_{\text{strat}} / \sigma_{\text{strat}}$ | Economic significance |
| Information Coefficient (IC) | $\text{Corr}(\hat r_t, r_t)$ rolling | Stability of predictive ranking |

### 7.6 Overfitting safeguards

- Report **in-sample vs out-of-sample** $R^2$ side by side. A ratio > 2 is a red flag.
- Use **adjusted $R^2$** and **AIC/BIC** for nested model comparison.
- Run a **deflated Sharpe ratio** (Bailey & López de Prado, 2014) to correct for trial multiplicity if you have searched over many feature/lag combinations.
- Compute **probability of backtest overfitting** (PBO) for the final strategy.

---

## 8. Event-Based Analysis

### 8.1 Event windows

Construct event days for:

- Large SPX moves: $|r^{SPX}_{t-1}| > 2\%$ (top and bottom 2.5% of distribution)
- FOMC announcement days
- U.S. CPI release days
- Banking-stress episodes (March 2023 SVB; 2008 GFC; 1998 LTCM-adjacent)
- COVID crash window: 2020-02-20 to 2020-04-07

### 8.2 Event-study design

Estimate **abnormal returns** for IHSG using a market model with MSCI EM as the benchmark (so the question becomes: *did IHSG react beyond what global EM beta predicts?*):

$$r^{IHSG}_t = \alpha + \beta \, r^{MSCI EM}_t + \epsilon_t$$

Define $AR_t = r^{IHSG}_t - (\hat\alpha + \hat\beta r^{MSCI EM}_t)$. Compute cumulative abnormal returns (CAR) over $[0, +1], [0, +2], [0, +5]$ trading days post-event.

Test $H_0: \overline{CAR} = 0$ via cross-sectional t-test and the BMP (Boehmer-Musumeci-Poulsen) test (robust to event-induced variance changes).

### 8.3 Measurements

- **Response delay.** Days until cumulative IHSG abnormal return reaches 50%, 75%, 100% of the eventual move.
- **Magnitude ratio.** Median $|AR^{IHSG}_{t+1}| / |r^{SPX}_t|$ conditional on the event.
- **Recovery time.** Days until $AR$ returns to within $\pm 0.5\sigma$ of zero.

---

## 9. Robustness Checks

These belong in the paper's robustness section (often the most-scrutinized part by referees):

1. **Frequency robustness.** Repeat the core Granger test at weekly (Wednesday-to-Wednesday) and monthly frequencies. Daily significance that disappears at weekly horizons suggests microstructure, not fundamental lead.
2. **Sub-period stability.** Pre-2018 / 2018–2021 / post-2021. Pre/post-COVID split is mandatory.
3. **Currency adjustment.** Repeat with $r^{IHSG, USD}$ (using USD/IDR). If the SPX lead disappears in USD terms, the "lead" is partly an FX phenomenon.
4. **Overnight vs intraday decomposition.** Split IHSG return into overnight ($C_{t-1} \to O_t$) and intraday ($O_t \to C_t$). The hypothesis predicts the overnight component should carry most of the SPX-leading signal; the intraday component should not.
5. **Heteroskedasticity-robust inference.** Use **Newey-West (HAC)** SE with bandwidth $L = \lfloor 4(T/100)^{2/9} \rfloor$ for all regression-based inference. Alternatively, GARCH-filter the residuals and re-test.
6. **Fat tails.** Re-estimate VAR with Student-$t$ innovations; compare coefficients.
7. **Outlier sensitivity.** Re-run with winsorization at 1% / 99%; report whether conclusions change.
8. **Reverse-causality placebo.** Run the symmetric Granger test IHSG → SPX. A small but non-zero coefficient is fine; a large one indicates contemporaneous comovement masquerading as a lead.
9. **Sub-sector robustness.** Test SPX lead on IHSG sector indices (banking, consumer, energy, mining). Heterogeneity is informative — banking and large-cap consumer typically show the strongest U.S. lead.
10. **Multiple-testing correction.** If you test many lags, apply Bonferroni or Benjamini-Hochberg FDR to the lag-by-lag p-values.

---

## 10. Econometric Pitfalls and False-Positive Sources

Document these explicitly in the limitations section:

| Pitfall | Consequence | Mitigation |
|---|---|---|
| Trading-day misalignment | Spurious lead-1 | Strict wall-clock alignment (§3.3) |
| Non-stationary inputs | Spurious regression | Returns + stationarity tests |
| Omitted global factor | SPX appears causal but is just a proxy | Control for VIX, DXY, MSCI EM |
| ARCH effects in residuals | Inflated t-statistics | HAC SE / GARCH filter |
| Look-ahead in features | Inflated OOS performance | Walk-forward, purged CV |
| Survivorship in macro panels | Biased coefficients | Use full available history of each series |
| Cherry-picking lag $k$ | Multiplicity bias | Report all lags 1–30; correct p-values |
| Regime-conditional sampling | Biased pooled estimate | Explicit regime stratification |
| Microstructure noise at daily level | Distorted contemporaneous correlation | Use close-to-close consistently |

---

## 11. Implementation

### 11.1 Recommended Python stack

```
Data & alignment       : pandas, numpy, exchange_calendars, pandas_market_calendars
Acquisition            : yfinance, pandas_datareader, fredapi
Statistics & econometrics
    Stationarity, VAR  : statsmodels (tsa.stattools, tsa.vector_ar)
    Cointegration      : statsmodels.tsa.vector_ar.vecm
    GARCH / vol        : arch
    HAC SE             : statsmodels (cov_type='HAC')
Regime switching       : statsmodels.tsa.regime_switching.MarkovRegression
Structural breaks      : ruptures
DTW                    : dtaidistance, tslearn
Transfer entropy       : PyIF, IDTxl
Machine learning       : scikit-learn, xgboost, lightgbm
Sequence models        : tensorflow / keras, pytorch
Backtesting & metrics  : vectorbt, pyfolio, empyrical
Plots                  : matplotlib, seaborn, plotly
Reproducibility        : hydra (configs), mlflow (experiment tracking)
```

### 11.2 Project structure

```
sp500_ihsg_leadlag/
├── README.md
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── regime.yaml
├── data/
│   ├── raw/                  # immutable downloads
│   ├── interim/              # aligned, cleaned
│   └── processed/            # feature matrices
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── align_calendars.py
│   │   └── returns.py
│   ├── eda/
│   │   ├── distributions.py
│   │   ├── stationarity.py
│   │   └── rolling_corr.py
│   ├── tests/
│   │   ├── ccf.py
│   │   ├── granger.py
│   │   ├── var.py
│   │   ├── cointegration.py
│   │   ├── dtw.py
│   │   └── transfer_entropy.py
│   ├── regimes/
│   │   ├── ms_var.py
│   │   ├── bai_perron.py
│   │   └── rolling_granger.py
│   ├── models/
│   │   ├── linear.py
│   │   ├── tree_ensembles.py
│   │   └── lstm.py
│   ├── events/
│   │   └── event_study.py
│   ├── backtest/
│   │   └── signal.py
│   └── utils/
│       ├── stats.py
│       └── plots.py
├── notebooks/                # exploratory only, never source-of-truth
├── reports/
│   ├── figures/
│   └── tables/
└── tests/                    # unit tests for src/
```

### 11.3 Workflow pipeline

1. **Download & cache** raw OHLCV and macro series → `data/raw/`.
2. **Align calendars**, compute log returns, handle holidays → `data/interim/`.
3. **EDA**: stationarity, distributions, rolling correlations → `reports/figures/`.
4. **Lead-lag tests**: CCF, Granger (multiple $p$), VAR + IRF/FEVD, cointegration, DTW, transfer entropy → results table.
5. **Regime analysis**: Markov-switching VAR, structural breaks, rolling Granger.
6. **Predictive layer**: feature engineering → walk-forward CV → linear / LASSO / XGBoost / (LSTM) → metrics table.
7. **Event studies** on FOMC, CPI, large-move days, crisis windows.
8. **Robustness**: frequencies, sub-periods, currency, overnight/intraday split, HAC, fat-tails.
9. **Trading-signal construction** (§13).
10. **Reporting**: compile tables and figures into the paper template.

### 11.4 Suggested paper structure

1. **Abstract** — hypothesis, methods (one line each), headline result, one-line economic significance.
2. **Introduction** — motivation, contribution, related literature (Eun & Shim 1989, Hamao Masulis Ng 1990, Forbes & Rigobon 2002, Hong & Stein 1999, recent IHSG-specific work).
3. **Data** — sources, alignment protocol, descriptive statistics table.
4. **Methodology** — sections 4–9 above, with full equations.
5. **Results**
   - 5.1 Static lead-lag (CCF, Granger, VAR)
   - 5.2 Long-run relationship (cointegration)
   - 5.3 Nonlinear dependence (transfer entropy, DTW)
   - 5.4 Regime conditioning
   - 5.5 Predictive performance
   - 5.6 Event studies
6. **Robustness** — all checks from §9.
7. **Economic interpretation** — what mechanism explains the finding; consistency with Hong-Stein or with common-factor exposure.
8. **Trading signal evaluation** — Sharpe, drawdown, turnover, transaction-cost sensitivity.
9. **Limitations & future work**.
10. **Conclusion**.

---

## 12. Visualization Plan

Build these as named figures with consistent styling:

| Figure | Content |
|---|---|
| F1 | Price levels (twin axis) and log returns, 2014–present |
| F2 | Return distributions with Q-Q plots vs Normal and Student-$t$ |
| F3 | Rolling 60d and 252d cross-correlation $\rho(r^{SPX}_{t-1}, r^{IHSG}_t)$ |
| F4 | CCF bar chart for $k \in [-30, +30]$ with bootstrap confidence bands |
| F5 | Granger F-statistic vs lag $p$, with 5% and 1% critical lines |
| F6 | IRF: response of IHSG to one-SD SPX shock, 95% bootstrap bands |
| F7 | FEVD stacked area: fraction of IHSG forecast variance from SPX vs own |
| F8 | Markov-switching smoothed regime probabilities over time, overlaid with VIX |
| F9 | Rolling 252d Granger F-statistic with crisis-window shading |
| F10 | Out-of-sample predicted vs realized IHSG returns (and directional accuracy by regime) |
| F11 | Event-study CAR around large-SPX-move days and FOMC days |
| F12 | Cumulative P&L of the SPX-based signal, with drawdowns and Sharpe annotations |

---

## 13. From Inference to Trading Signal

If H1 is supported, the natural translation to a tradable signal is:

### 13.1 Signal construction

$$s_t = w_1 \, r^{SPX}_{t-1} + w_2 \, \widetilde{r}^{SPX}_{[t-5, t-1]} + w_3 \, (\Delta VIX_{t-1}) + w_4 \, (\text{macro})$$

where weights $w_i$ come from the walk-forward LASSO or XGBoost out-of-fold predictions. Normalize $s_t$ to a rolling z-score with trailing window (e.g., 60d) so the signal is regime-stationary.

### 13.2 Position rule

- **Long IHSG (or LQ45 futures, or representative ETF)** if $s_t > +\tau$
- **Flat or short** if $s_t < -\tau$
- $\tau$ calibrated on the training period; typical choice $\tau \in [0.3, 0.7]$ (in z-score units).

### 13.3 Execution realism

- Trade at **IHSG open**, since the signal is built from prior-evening SPX close — that is the only honest implementation.
- Include realistic transaction costs (round-trip ≈ 20–40 bps for Indonesian large-caps; more for small-caps).
- Apply position limits and a stop-loss; report performance both with and without these overlays.

### 13.4 Performance reporting

Required metrics:
- Annualized return, volatility, Sharpe, Sortino, Calmar
- Max drawdown and time-to-recovery
- Hit ratio, average win / average loss
- Turnover
- Sensitivity to transaction cost assumption (plot Sharpe vs cost)
- **Deflated Sharpe ratio** to correct for multiple testing across hyperparameters

### 13.5 Honesty checklist

- Compare against IHSG buy-and-hold and against a momentum-only benchmark.
- Report the worst rolling 1-year window, not just lifetime statistics.
- State explicitly: in which regimes the signal works, in which it fails.

---

## 14. Research Limitations to Anticipate

State these proactively in the paper:

1. **Granger ≠ structural causality.** All inference is predictive, not identifying.
2. **The 10-year sample contains a small number of true crisis events.** Regime-conditioned estimates have limited statistical power.
3. **Daily frequency cannot distinguish intraday transmission channels.** Intraday data (1-min) would refine the lag estimate but is out of scope.
4. **Endogeneity of macro controls.** USD/IDR is itself partly driven by SPX-correlated factors. Use cautiously in interpretation.
5. **External validity.** Results are specific to IHSG; sector indices and individual stocks may behave differently.
6. **Microstructure differences.** IHSG has had circuit-breaker rule changes and short-sale restrictions that may have shifted the response function across the sample.

---

## 15. Headline Deliverables Checklist

A defensible paper or signal report should produce all of the following:

- [ ] Cleaned, calendar-aligned dataset with explicit alignment documentation
- [ ] Stationarity test table (ADF/KPSS/PP on prices and returns)
- [ ] CCF chart with bootstrap CIs and table of $k^\ast$ and $\rho^\ast$
- [ ] Granger causality table across $p \in \{1, 2, 3, 5, 10, 20\}$ with HAC SE
- [ ] VAR estimation output, IRF figure, FEVD figure
- [ ] Cointegration test (Engle-Granger + Johansen)
- [ ] Transfer entropy result with shuffled-null p-value
- [ ] Markov-switching VAR with regime-conditional coefficients
- [ ] Rolling-window Granger time series
- [ ] Predictive-model OOS metrics with walk-forward CV
- [ ] Event-study CAR table
- [ ] Full robustness battery (§9)
- [ ] Trading-signal P&L, Sharpe, drawdown, turnover
- [ ] Deflated Sharpe and PBO

---

*End of methodology document.*
