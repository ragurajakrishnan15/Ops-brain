"""
One-time data setup script.

Run this after downloading the dataset to data/raw/DataCoSupplyChainDataset.csv.

Usage:
    python scripts/setup_data.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline.ingest import create_clean_orders_table, ingest_supply_chain
from src.data_pipeline.transforms import (
    build_daily_demand_table,
    build_department_tables,
    build_feature_table,
    build_inventory_risk_view,
)
from src.rag_assistant.ingest_docs import ingest_all


def main():
    print("=" * 60)
    print("OpsBrain Data Setup")
    print("=" * 60)

    # Step 1: Ingest raw data
    print("\n[1/6] Ingesting raw supply chain data...")
    try:
        ingest_supply_chain()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo get started:")
        print("1. Download the DataCo Smart Supply Chain dataset from Kaggle:")
        print(
            "   https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
        )
        print("2. Place DataCoSupplyChainDataset.csv in data/raw/ (or set RAW_CSV_PATH)")
        print("3. Run this script again")
        sys.exit(1)

    # Step 2: Clean and standardize
    print("\n[2/6] Creating clean orders table...")
    create_clean_orders_table()

    # Step 3: Build daily demand
    print("\n[3/6] Building daily demand aggregation...")
    build_daily_demand_table()

    # Step 4: Build ML features
    print("\n[4/6] Building ML feature tables (lags, rolling averages, seasonality)...")
    build_feature_table()
    build_department_tables()

    # Step 5: Build inventory risk view
    print("\n[5/6] Building inventory risk view...")
    build_inventory_risk_view()

    # Step 6: Ingest documents for RAG
    print("\n[6/6] Ingesting operations documents for RAG assistant...")
    ingest_all()

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train forecasting models:  python -m src.forecasting.train")
    print("  2. Launch dashboard:          streamlit run streamlit_app.py")
    print("  3. (Optional) View MLflow:    mlflow ui")


if __name__ == "__main__":
    main()
