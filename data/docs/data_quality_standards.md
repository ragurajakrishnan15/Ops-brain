# Data Quality Standards for Operations Analytics

## 1. Data Freshness
- Order data: Updated daily by 6:00 AM UTC
- Inventory levels: Updated every 4 hours
- Shipping status: Real-time (event-driven updates)
- Forecast outputs: Refreshed weekly on Monday mornings

## 2. Completeness Requirements
Every order record must have:
- Order ID (non-null, unique per line item)
- Order date (valid date, not in the future)
- Customer ID (non-null)
- Product name and category (non-null)
- Quantity (positive integer)
- Revenue (non-negative)
- Shipping mode (one of: Standard, Second Class, First Class, Same Day)

## 3. Validation Rules
- Discount rate must be between 0.0 and 1.0
- Unit price must be positive
- Ship date must be on or after order date
- Late delivery risk flag must be 0 or 1

## 4. Missing Data Handling
- Missing discount rates: Default to 0.0 (no discount)
- Missing ship dates: Flag for manual review
- Missing customer location: Exclude from geographic analysis but include in demand totals

## 5. Outlier Detection
- Daily demand values exceeding 3 standard deviations from the rolling 28-day mean are flagged
- Revenue outliers (single order > $10,000) are reviewed for data entry errors
- Negative profit values are reviewed but not automatically excluded (valid for loss-leader promotions)

## 6. Audit Trail
- All data transformations are logged with timestamps
- Raw data is preserved in its original form in the staging layer
- Feature engineering SQL is version-controlled in Git
- Model training data snapshots are versioned with DVC
