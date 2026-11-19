"""Materialize auditable descriptive marts and compact dashboard inputs. No model fitting."""

from pathlib import Path
import json
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "business"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "analysis/data/warehouse.duckdb"), read_only=True)
    con.execute("SET threads=2")
    con.execute("SET memory_limit='2GB'")
    con.execute((ROOT / "additions/business_metrics.sql").read_text())
    names = [
        "mart_store_day",
        "mart_category_day",
        "mart_hour_city",
        "mart_persistence",
    ]
    for name in names:
        df = con.execute(f"SELECT * FROM {name} ORDER BY ALL").df()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(name, len(df), flush=True)
    checks = {
        "facts": con.execute("SELECT COUNT(*) FROM int_daily").fetchone()[0],
        "sum_store_product_days": con.execute(
            "SELECT SUM(product_days) FROM mart_store_day"
        ).fetchone()[0],
        "sum_category_product_days": con.execute(
            "SELECT SUM(product_days) FROM mart_category_day"
        ).fetchone()[0],
        "hours": con.execute("SELECT SUM(observations) FROM mart_hour_city").fetchone()[
            0
        ],
        "daily_oos_hours": con.execute(
            "SELECT SUM(oos_hours) FROM mart_store_day"
        ).fetchone()[0],
        "hourly_oos_observations": con.execute(
            "SELECT SUM(oos_observations) FROM mart_hour_city"
        ).fetchone()[0],
        "field_window_mismatches": con.execute(
            "SELECT COUNT(*) FROM int_daily WHERE stock_hour6_22_cnt<>oos_hours"
        ).fetchone()[0],
        "split_dates": con.execute(
            "SELECT split,COUNT(*) n,MIN(dt),MAX(dt),MIN(label_dt),MAX(label_dt),AVG(label) FROM model_dataset GROUP BY split ORDER BY split"
        ).fetchall(),
        "feature_date_not_before_label": con.execute(
            "SELECT COUNT(*) FROM model_dataset WHERE dt>=label_dt"
        ).fetchone()[0],
    }
    assert (
        checks["facts"]
        == checks["sum_store_product_days"]
        == checks["sum_category_product_days"]
        == 4850000
    )
    assert checks["hours"] == checks["facts"] * 16
    assert checks["daily_oos_hours"] == checks["hourly_oos_observations"]
    assert (
        checks["field_window_mismatches"]
        == checks["feature_date_not_before_label"]
        == 0
    )
    (OUT / "integrity_checks.json").write_text(
        json.dumps(checks, indent=2, default=str)
    )
    global_daily = con.execute(
        "SELECT dt,SUM(product_days) product_days,SUM(oos_hours) oos_hours,SUM(severe_days) severe_days,SUM(close_oos_days) close_oos_days FROM mart_store_day GROUP BY dt ORDER BY dt"
    ).df()
    global_daily.to_csv(OUT / "global_daily.csv", index=False)
    # Comparative store rates for first/last 7-day windows of the held-out week / preceding week.
    comparison = con.execute("""
      SELECT city_id,store_id,
      SUM(CASE WHEN dt>='2024-06-26' THEN product_days ELSE 0 END) test_product_days,
      SUM(CASE WHEN dt>='2024-06-26' THEN severe_days ELSE 0 END) test_severe_days,
      SUM(CASE WHEN dt<'2024-06-26' THEN product_days ELSE 0 END) prev_product_days,
      SUM(CASE WHEN dt<'2024-06-26' THEN severe_days ELSE 0 END) prev_severe_days
      FROM mart_store_day WHERE dt>='2024-06-19' GROUP BY city_id,store_id
    """).df()
    comparison["test_rate"] = comparison.test_severe_days / comparison.test_product_days
    comparison["prev_rate"] = comparison.prev_severe_days / comparison.prev_product_days
    comparison["change_pp"] = (comparison.test_rate - comparison.prev_rate) * 100
    comparison.sort_values("change_pp", ascending=False).to_csv(
        OUT / "store_week_comparison.csv", index=False
    )
    con.close()


if __name__ == "__main__":
    main()
