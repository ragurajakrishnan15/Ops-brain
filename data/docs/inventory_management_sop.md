# Standard Operating Procedure: Inventory Management

## 1. Purpose
This SOP defines the process for monitoring, replenishing, and managing inventory levels across distribution warehouses.

## 2. Scope
Applies to all product categories across North America, LATAM, and EMEA distribution centers.

## 3. Reorder Point Calculation
The reorder point (ROP) for each SKU is calculated as:

**ROP = (Average Daily Demand × Lead Time) + Safety Stock**

Where:
- Average Daily Demand = rolling 28-day average of units shipped
- Lead Time = average supplier fulfillment time in days (varies by category)
- Safety Stock = 1.65 × standard deviation of daily demand × sqrt(lead time)

The 1.65 multiplier targets a 95% service level.

## 4. Safety Stock Policy
- Category A items (top 20% by revenue): 2-week safety stock minimum
- Category B items (next 30%): 1-week safety stock minimum
- Category C items (remaining): 3-day safety stock minimum

## 5. Late Delivery Escalation
When actual shipping time exceeds planned shipping time:
1. Flag the order as "late delivery risk" in the tracking system
2. If delay exceeds 3 days: escalate to Supply Chain Manager
3. If delay exceeds 7 days: escalate to VP Operations with root cause analysis

## 6. Demand Forecasting Integration
- Forecasts are refreshed weekly using the demand forecasting model (see the Demand Planning Process)
- Forecast accuracy is tracked via WAPE and MAE; MAPE is reported only for days with non-zero demand
- When a category misses its WAPE target for two consecutive weeks, the model is investigated and retrained
- Manual overrides require documentation in the forecast adjustment log

## 7. KPIs
| Metric | Target | Frequency |
|--------|--------|-----------|
| Inventory Turnover | > 8x/year | Monthly |
| Stockout Rate | < 2% | Weekly |
| Forecast Accuracy (WAPE, high-volume categories) | < 25% | Weekly |
| On-Time Delivery | > 95% | Daily |
| Days of Supply | 14-21 days | Weekly |

## 8. Review Schedule
This SOP is reviewed quarterly by the Supply Chain Analytics team.
Last review: Q3 2025
Next review: Q4 2025
