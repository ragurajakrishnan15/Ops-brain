# Order Status and Cancellation Policy

## 1. Purpose
Defines what each order status means, which orders count as demand for planning and reporting, and how canceled and suspected-fraud orders are handled.

## 2. Order Status Definitions
| Status | Meaning | Counts as demand? |
|--------|---------|-------------------|
| PENDING | Order received, awaiting processing | Yes |
| PENDING_PAYMENT | Awaiting payment confirmation | Yes |
| PAYMENT_REVIEW | Payment flagged for manual review | Yes |
| ON_HOLD | Paused at customer or warehouse request | Yes |
| PROCESSING | Being picked and packed | Yes |
| COMPLETE | Shipped and delivered | Yes |
| CLOSED | Completed and financially settled | Yes |
| CANCELED | Canceled by customer or operations | **No** |
| SUSPECTED_FRAUD | Blocked by fraud screening | **No** |

## 3. Demand Reporting Rules
- Demand forecasts, daily demand tables, and inventory risk scores exclude CANCELED and SUSPECTED_FRAUD order lines.
- Excluded orders remain in the order history with a flag so cancellation and fraud rates can still be analyzed.
- Revenue reporting uses the same exclusions.

## 4. Cancellation Handling
1. Customer-initiated cancellations are accepted until the order reaches PROCESSING.
2. After PROCESSING, a cancellation is handled as a return under the Returns and Reverse Logistics section of the Shipping & Logistics Policy.
3. Operations-initiated cancellations (stockout, address failure) require a customer notification within 24 hours and are tracked by reason code.
4. A category whose weekly cancellation rate exceeds 5% is reviewed by the Supply Chain Manager.

## 5. Suspected Fraud Review
1. Orders flagged by fraud screening move to SUSPECTED_FRAUD and are not released to the warehouse.
2. The payments team reviews flagged orders within one business day.
3. Orders cleared by review return to PENDING and enter normal processing; their demand is then counted from the original order date.
4. Confirmed fraud is canceled and reported monthly by market and customer segment.

## 6. Shipping Canceled Status
The delivery status "Shipping canceled" corresponds to orders canceled after a shipping label was created. These orders are excluded from on-time delivery metrics.
