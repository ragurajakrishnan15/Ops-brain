"""Tests for the deployment data bundle and the late-delivery statistics."""

import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.analytics import late_delivery
from src.utils import bundle

# --- data bundle ---------------------------------------------------------------------


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(bundle, "PROJECT_ROOT", root)
    monkeypatch.setattr(bundle, "DUCKDB_PATH", str(root / "data/processed/warehouse.duckdb"))
    monkeypatch.setattr(bundle, "CHROMA_PERSIST_DIR", str(root / "data/processed/chroma_store"))
    monkeypatch.setattr(bundle, "MLFLOW_EXPERIMENT", "opsbrain-forecast")
    return root


def _make_zip(path, entries: dict[str, str]) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


GOOD_ENTRIES = {
    "data/processed/warehouse.duckdb": "db",
    "data/processed/chroma_store/chroma.sqlite3": "chroma",
    "mlruns/123/meta.yaml": "name: opsbrain-forecast\n",
}


def test_bundle_round_trip_with_checksum(fake_project, tmp_path):
    archive = tmp_path / "bundle.zip"
    digest = _make_zip(archive, GOOD_ENTRIES)
    assert not bundle.data_is_present()

    count = bundle.download_bundle(archive.as_uri(), sha256=digest)
    assert count == 3 and bundle.data_is_present()
    assert bundle.ensure_data() == "present"


def test_bundle_checksum_mismatch_rejected(fake_project, tmp_path):
    archive = tmp_path / "bundle.zip"
    _make_zip(archive, GOOD_ENTRIES)
    with pytest.raises(ValueError, match="checksum"):
        bundle.download_bundle(archive.as_uri(), sha256="0" * 64)
    assert not bundle.data_is_present()


@pytest.mark.parametrize("bad_name", ["../evil.py", "src/utils/config.py", "mlruns/../../evil.py"])
def test_bundle_rejects_unexpected_paths(fake_project, tmp_path, bad_name):
    archive = tmp_path / "bundle.zip"
    _make_zip(archive, {**GOOD_ENTRIES, bad_name: "x"})
    with pytest.raises(ValueError, match="Unexpected path"):
        bundle.download_bundle(archive.as_uri())
    assert not (fake_project / "data").exists()  # nothing extracted


# --- late-delivery statistics ----------------------------------------------------------


@pytest.fixture(scope="module")
def orders():
    """Lateness depends only on shipping mode; every other column is noise."""
    rng = np.random.default_rng(7)
    n = 6000
    modes = rng.choice(
        ["Standard Class", "Second Class", "Same Day", "First Class"], n, p=[0.55, 0.2, 0.1, 0.15]
    )
    late_prob = (
        pd.Series(modes)
        .map({"Standard Class": 0.4, "Second Class": 0.8, "Same Day": 0.48, "First Class": 1.0})
        .to_numpy()
    )
    return pd.DataFrame(
        {
            "late": (rng.random(n) < late_prob).astype(int),
            "shipping_mode": modes,
            "market": rng.choice(["Africa", "Europe", "LATAM", "USCA"], n),
            "customer_segment": rng.choice(["Consumer", "Corporate", "Home Office"], n),
            "department_name": rng.choice(["Apparel", "Fitness", "Golf"], n),
            "category_name": rng.choice([f"cat{i}" for i in range(8)], n),
            "order_region": rng.choice([f"region{i}" for i in range(6)], n),
            "quantity": rng.integers(1, 5, n),
            "discount_rate": rng.uniform(0, 0.25, n),
            "planned_ship_days": pd.Series(modes)
            .map({"Standard Class": 4, "Second Class": 2, "Same Day": 0, "First Class": 1})
            .to_numpy(),
            "actual_ship_days": rng.integers(0, 6, n),
        }
    )


def test_chi_square_ranks_the_real_driver_first(orders):
    table = late_delivery.chi_square_tests(orders)
    assert table.iloc[0]["driver"] == "Shipping mode"
    assert table.iloc[0]["p_value"] < 1e-10 and table.iloc[0]["cramers_v"] > 0.3
    assert (table.iloc[1:]["cramers_v"] < 0.1).all()


def test_logistic_regression_recovers_effect_and_handles_separation(orders):
    odds, summary = late_delivery.logistic_regression(orders)
    # First Class is always late (perfect separation) -> excluded and reported
    assert summary["first_class_late_pct"] == 100.0
    assert summary["excluded_first_class_lines"] == int(
        (orders["shipping_mode"] == "First Class").sum()
    )
    # True odds ratio Second Class vs Standard: (0.8/0.2) / (0.4/0.6) = 6
    second = odds.set_index("term").loc["Shipping mode: Second Class"]
    assert second["ci_low"] < 6 < second["ci_high"]
    # Noise columns add nothing beyond shipping mode
    assert summary["lr_p_value_extra_drivers"] > 0.01
    assert summary["pseudo_r2_full"] - summary["pseudo_r2_shipping_only"] < 0.01
