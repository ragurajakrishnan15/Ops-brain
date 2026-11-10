# Operations KPI and Terms Glossary

## 1. Forecast Accuracy Metrics
- **MAE (Mean Absolute Error):** average of |actual − forecast| in units. Easy to interpret, but not comparable across categories of different size.
- **WAPE (Weighted Absolute Percentage Error):** sum of |actual − forecast| divided by sum of actual demand. The primary accuracy metric because it is comparable across categories and stays defined on days with zero demand.
- **MAPE (Mean Absolute Percentage Error):** average of |actual − forecast| / actual. Undefined when actual demand is zero, so it is calculated only over days with demand.
- **RMSE (Root Mean Squared Error):** penalizes large misses more heavily than MAE.
- **Interval coverage:** the share of actual values that fall inside the 95% prediction interval. Well-calibrated intervals have coverage close to 95%.
- **Champion method:** the forecasting method (XGBoost, 28-day moving average, or seasonal naive) with the lowest backtest error for a series.

## 2. Delivery Metrics
- **Late delivery rate:** late order lines divided by all order lines, as a percentage. An order line is late when actual shipping days exceed scheduled shipping days (late delivery risk flag = 1).
- **On-time delivery:** 100% minus the late delivery rate.
- **Delivery status values:** Advance shipping (arrived before schedule), Shipping on time, Late delivery, Shipping canceled.
- **Actual vs. scheduled shipping days:** "Days for shipping (real)" versus "Days for shipment (scheduled)".

## 3. Inventory Risk Score
The inventory risk score ranks categories for review on a 0–100 scale:

**Risk score = 0.6 × late delivery rate (%) + 0.4 × demand volatility (%)**

- Demand volatility is the coefficient of variation of daily units (standard deviation / mean), expressed as a percentage and capped at 100.
- Low-volume categories often reach the volatility cap, so their scores should be read together with their average daily units.

## 4. Order and Customer Terms
- **Order vs. order line:** one order can contain several order lines (items). Demand is measured in units across order lines.
- **Valid demand:** all order statuses except CANCELED and SUSPECTED_FRAUD (see the Order Status and Cancellation Policy).
- **Customer segments:** Consumer, Corporate, Home Office.
- **Markets:** Africa, Europe, LATAM, Pacific Asia, USCA.

## 5. Shipping Modes
| Mode in order data | Policy name | Transit time |
|--------------------|-------------|--------------|
| Standard Class | Standard Shipping | 5–7 business days |
| Second Class | Second Class | 3–5 business days |
| First Class | First Class | 1–2 business days |
| Same Day | Same Day | Within 24 hours |

## 6. Inventory Terms
- **Reorder point (ROP):** (average daily demand × lead time) + safety stock.
- **Safety stock:** 1.65 × standard deviation of daily demand × √(lead time), targeting a 95% service level.
- **Days of supply:** on-hand inventory divided by average daily demand.
