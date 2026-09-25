# -*- coding: utf-8 -*-
"""RealE 数据集对齐：FR / DE-LU → 统一小时级长表，列名与 configs/columns/REALE 完全一致。

架构映射（CAS-EPF 2.0）：
  Y = 日前价（y_）                ← Day Ahead Prices
  Z = 日前预报（genf_ / ntc_ / total_*Forecast） ← Generation Forecast / NTC / Total
  X = 历史实际（total_Actual、燃料 Actual Aggregated） ← Total / Actual Generation per Production Type

列名前缀（对齐后列名 = 配置列名）：
  y_       ← "Day-ahead Price [EUR/MWh]"
  genf_    ← "Scheduled Generation [MW] (D) - BZN|{zone}"
  ntc_     ← "Net Position (MW)"（取第一列 = Daily 日前）
  total_   ← "Day-ahead Total Load Forecast ..." + "Actual Total Load ..."
  (无前缀)  ← "{fuel}  - Actual Aggregated [MW]"（无 (D)/(D-1) 后缀）

分辨率/重复（实测）：
  FR    ：全部小时级，干净（DST 回拨小时重复 9 个）。
  DE-LU ：Prices/GenForecast/NTC = 小时级 ×2 重复（DE+LU）；Total/Fuel = 15 分钟 ×2 重复。
聚合：Total/Fuel 按小时取均值；其余按小时取首个非空。DST 重复小时在 floor("h") 时被折叠（已知事项）。
"""
import os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# The local Real-E release is organized as OLD (=Original on Zenodo).
RAW = REPO_ROOT / "Real-E" / "OLD"
OUT_DIR = REPO_ROOT / "data" / "REALE"

FUELS = {
    "FR": ["Biomass", "Fossil Gas", "Fossil Hard coal", "Fossil Oil",
           "Hydro Pumped Storage", "Hydro Run-of-river and poundage",
           "Hydro Water Reservoir", "Nuclear", "Solar", "Waste", "Wind Onshore"],
    "DE-LU": ["Biomass", "Fossil Brown coal/Lignite", "Fossil Gas", "Fossil Hard coal",
              "Fossil Oil", "Geothermal", "Hydro Pumped Storage",
              "Hydro Run-of-river and poundage", "Hydro Water Reservoir", "Nuclear",
              "Other renewable", "Solar", "Waste", "Wind Offshore", "Wind Onshore", "Other"],
}


def parse_start(t: pd.Series) -> pd.Series:
    """区间时间 'DD.MM.YYYY HH:MM[-SS] - ... [(CET/CEST)]' → 起点 naive datetime。"""
    s = t.astype(str).str.replace(r"\s*\(CET/CEST\)\s*$", "", regex=True)
    start = s.str.split(" - ", regex=False).str[0].str.replace("/", ".", regex=False)
    return pd.to_datetime(start, dayfirst=True)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"n/e": pd.NA, "N/E": pd.NA, "": pd.NA}), errors="coerce")


def _time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "MTU" in c or c.lower().startswith("time"):
            return c
    return df.columns[0]


def _to_hourly(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    """按起点小时聚合。agg='first' 去重取首个非空；agg='mean' 15min→小时均值。"""
    df = df.copy()
    df["time"] = parse_start(df[_time_col(df)])
    df = df.drop(columns=[c for c in df.columns if c not in ("time",) and (c.startswith(("MTU", "Time")) or "AREA" in c or c == "Area")])
    val_cols = [c for c in df.columns if c != "time"]
    df = df.sort_values("time")
    if agg == "first":
        g = df.groupby(df["time"].dt.floor("h"), as_index=False).agg(
            {c: lambda s: (s.dropna().iloc[0] if s.dropna().size else pd.NA) for c in val_cols})
    elif agg == "mean":
        g = df.groupby(df["time"].dt.floor("h"), as_index=False).agg({c: "mean" for c in val_cols})
    else:
        raise ValueError(agg)
    return g.rename(columns={"time": "time"})


def read_prices(path: str, zone: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({"time": parse_start(df[_time_col(df)])})
    out["y_Day-ahead Price [EUR/MWh]"] = num(df["Day-ahead Price [EUR/MWh]"])
    return _hourly_dedup(out)


def _hourly_dedup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = df["time"].dt.floor("h")
    return df.groupby("time", as_index=False).agg(
        {c: lambda s: (s.dropna().iloc[0] if s.dropna().size else pd.NA) for c in df.columns if c != "time"})


def read_gen_forecast(path: str, zone: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    raw = f"Scheduled Generation [MW] (D) - BZN|{zone}"
    out = pd.DataFrame({"time": parse_start(df[_time_col(df)])})
    out[f"genf_{raw}"] = num(df[raw])
    return _hourly_dedup(out)


def read_ntc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({"time": parse_start(df[_time_col(df)])})
    out["ntc_Net Position (MW)"] = num(df["Net Position (MW)"])  # 第一列 = Daily 日前
    return _hourly_dedup(out)


def read_total(path: str, zone: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({"time": parse_start(df[_time_col(df)])})
    out[f"total_Day-ahead Total Load Forecast [MW] - BZN|{zone}"] = num(
        df[f"Day-ahead Total Load Forecast [MW] - BZN|{zone}"])
    out[f"total_Actual Total Load [MW] - BZN|{zone}"] = num(df[f"Actual Total Load [MW] - BZN|{zone}"])
    df2 = out.copy()
    df2["time"] = df2["time"].dt.floor("h")
    return df2.groupby("time", as_index=False).mean(numeric_only=True)  # 15min→小时均值


def read_fuel(path: str, fuels: list) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({"time": parse_start(df[_time_col(df)])})
    for f in fuels:
        out[f"{f}  - Actual Aggregated [MW]"] = num(df[f"{f}  - Actual Aggregated [MW]"])
    df2 = out.copy()
    df2["time"] = df2["time"].dt.floor("h")
    return df2.groupby("time", as_index=False).mean(numeric_only=True)  # 15min→小时均值


def align(zone: str) -> pd.DataFrame:
    b = RAW
    parts = [
        read_prices(str(b / "Day Ahead Prices" / "BZN" / f"BZN {zone}.csv"), zone),
        read_gen_forecast(str(b / "Generation Forecast - Day ahead" / "BZN" / f"BZN {zone}.csv"), zone),
        read_ntc(str(b / "Implicit Allocations Net Transfer Capacities" / "BZN" / f"BZN {zone}.csv")),
        read_total(str(b / "Total" / "BZN" / f"BZN {zone}.csv"), zone),
        read_fuel(str(b / "Actual Generation per Production Type" / "BZN" / f"BZN {zone}.csv"), FUELS[zone]),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="time", how="outer")
    # 目标价放最后（与 regress config 一致），时间列在前
    cols = [c for c in out.columns if c != "y_Day-ahead Price [EUR/MWh]"]
    out = out[["time"] + [c for c in cols if c != "time"] + ["y_Day-ahead Price [EUR/MWh]"]]
    return out.sort_values("time").reset_index(drop=True)


def main() -> None:
    for zone in ["FR", "DE-LU"]:
        out = align(zone)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dst = str(OUT_DIR / f"{zone}_aligned.csv")
        out.to_csv(dst, index=False)
        print(f"[OK] {dst}")
        print(f"  形状={out.shape}  时间范围={out.time.min()} -> {out.time.max()}")
        print(f"  列={list(out.columns)}")
        print(f"  任一行含 NaN 的行数={int(out.isna().sum(axis=1).gt(0).sum())} / {len(out)}")
        print(f"  目标价非空数={out['y_Day-ahead Price [EUR/MWh]'].notna().sum()}")
        print()


if __name__ == "__main__":
    main()
