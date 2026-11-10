# Demand Planning Process

## 1. Purpose
This document describes the weekly demand planning cycle and how statistical forecasts from the OpsBrain forecasting engine are produced, reviewed, and used by planners.

## 2. Planning Cycle
| Day | Activity | Owner |
|-----|----------|-------|
| Monday | Forecast refresh: models retrained on data through the previous Sunday | Supply Chain Analytics |
| Tuesday | Exception review: categories with forecast error or interval width above threshold | Demand Planners |
| Wednesday | Consensus meeting: planners, merchandising, and logistics agree on overrides | Demand Planning Lead |
| Thursday | Approved forecast published to replenishment and warehouse capacity planning | Demand Planners |

## 3. Forecast Levels
- Forecasts are produced at two levels: **product category** and **department**.
- A series is forecast only if it has at least 365 days of history. Newer or sparse categories are planned with the department forecast and a manual category split.
- The standard forecast horizon is 28 days.

## 4. How Forecasts Are Produced
- Three methods are compared for every series: an XGBoost model, a 28-day moving average, and a same-weekday-last-week (seasonal naive) forecast.
- Each method is evaluated on six rolling 28-day backtests. The method with the lowest backtest error becomes the **champion** for that series and is the one published.
- The XGBoost model predicts how far demand will deviate from its 28-day average, so for series with little structure it stays close to the moving average.
- Each forecast has a 95% prediction interval calibrated from backtest errors. The interval widens with the forecast horizon.
- Accuracy is reported on the most recent 28 days, which are held out from model training and method selection.

## 5. Accuracy Targets and Escalation
- The primary accuracy metric is **WAPE** (weighted absolute percentage error). MAPE is reported only for days with non-zero demand.
- Target WAPE: below 25% for high-volume categories (10+ units per day); below 60% for low-volume categories.
- If a category misses its WAPE target for two consecutive weeks, the analytics team investigates data quality, structural changes (new or discontinued products), and model drift before retraining.
- If 95% interval coverage on the holdout falls below 85%, intervals are recalibrated.

## 6. Manual Overrides
- Planners may override a statistical forecast for known events (promotions, product launches, discontinuations).
- Every override must be recorded in the forecast adjustment log with the reason, the size of the adjustment, and the approver.
- Overrides larger than 20% of the statistical forecast require Demand Planning Lead approval.
- Override accuracy is reviewed monthly; overrides that do not improve accuracy are discussed in the consensus meeting.

## 7. Structural Changes
When a category is launched, discontinued, or re-organized, its history no longer represents future demand. The analytics team limits model training to the stable period and flags the affected series until 365 days of new history exist.
