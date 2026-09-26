"""Align the public Shanxi 15-minute market file to the hourly pipeline schema.

The source repository stores the header in a mojibake encoding, so this
adapter deliberately uses the documented column positions instead of relying
on Chinese header text.  Four 15-minute records are averaged into one hourly
record.  Future actual measurements are retained only as historical columns;
the regression configuration uses forecast columns and calendar variables as
known covariates.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Dataset" / "Shanxi electricity market data" / "Shanxi_electricity_15min_2024-2026.csv"
OUT = ROOT / "data" / "SHANXI" / "shanxi_aligned.csv"


def main() -> None:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    if df.shape[1] < 21:
        raise ValueError(f"Expected at least 21 source columns, got {df.shape[1]}")

    # Positional schema documented by ronaldowzy/price_forecast.
    names = [
        "date", "slot", "is_holiday", "is_weekend",
        "direct_load_forecast", "tie_load_forecast", "wind_forecast",
        "solar_forecast", "market_import_forecast", "tie_line_forecast",
        "thermal_forecast", "direct_load_actual", "tie_load_actual",
        "wind_actual", "solar_actual", "pumped_storage_actual",
        "thermal_actual", "market_import_actual", "tie_line_actual",
        "da_price", "y",
    ]
    df = df.iloc[:, :21].copy()
    df.columns = names
    # The source uses ``24:00`` for the final interval of a day.  Normalize it
    # to the next day's midnight before parsing.
    date = pd.to_datetime(df["date"], errors="coerce")
    slot = df["slot"].astype(str).str.strip()
    is_24 = slot.str.startswith("24:")
    normalized_slot = slot.where(~is_24, "00" + slot.str[2:])
    df["ds"] = pd.to_datetime(date.astype(str) + " " + normalized_slot, errors="coerce")
    df.loc[is_24, "ds"] = df.loc[is_24, "ds"] + pd.Timedelta(days=1)
    if df["ds"].isna().any():
        raise ValueError(f"Invalid timestamps: {int(df['ds'].isna().sum())}")

    numeric = [c for c in names if c not in {"date", "slot"}]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep the target and all physical fields, but aggregate to hourly to make
    # the comparison with the hourly Real-E experiment meaningful.
    df = df.sort_values("ds").set_index("ds")
    out = df[numeric].resample("1h").mean().reset_index()
    out["hour_of_day"] = out["ds"].dt.hour
    out["day_of_week"] = out["ds"].dt.dayofweek
    out["is_holiday"] = out["is_holiday"].fillna(0.0)
    out = out.drop(columns=["date", "slot"], errors="ignore")
    ordered = [
        "ds", "y", "da_price", "direct_load_forecast", "tie_load_forecast",
        "wind_forecast", "solar_forecast", "market_import_forecast",
        "tie_line_forecast", "thermal_forecast", "direct_load_actual",
        "tie_load_actual", "wind_actual", "solar_actual", "pumped_storage_actual",
        "thermal_actual", "market_import_actual", "tie_line_actual",
        "hour_of_day", "day_of_week", "is_holiday",
    ]
    out = out[ordered].sort_values("ds").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[OK] {OUT}")
    print(f"shape={out.shape} range={out.ds.min()} -> {out.ds.max()}")
    print(f"duplicates={out.ds.duplicated().sum()} non_hourly={(out.ds.diff().dropna() != pd.Timedelta(hours=1)).sum()}")
    print(f"target_missing={out.y.isna().sum()} feature_missing={int(out.drop(columns=['ds','y']).isna().sum().sum())}")


if __name__ == "__main__":
    main()
