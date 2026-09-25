# -*- coding: utf-8 -*-
"""广东数据集对齐：产出统一小时级长表（实时价目标）。

架构映射（CAS-EPF 2.0，实时价主实验）：
  Y = 实时电价（Real-time Prices，目标）
  X = 实时实际（Real-time Data，历史外生 → 进冻结 TSFM）
  Z = 日前计划（Day-ahead Data，未来可得 → 进树模型）
  da_price = 日前电价（D 日发布，未来可得 → 可选 Z 特征）

输出列：
  ds         小时时间戳
  y          实时电价（目标，CNY/MWh）
  da_price   日前电价（未来已知，可选 Z）
  rt_*       实时实际（X，15min→小时均值）
  da_*       日前计划（Z，15min→小时均值）
"""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "Dataset" / "Guangdong electricity market data"
OUT = REPO_ROOT / "data" / "GD" / "guangdong_aligned.csv"

# 列名映射（原始 → 简洁）
MW_COLS = {
    "Local_Power_Output(MW)": "Local_Power_Output",
    "In-Province_Type_A_Power(MW)": "Type_A_Power",
    "In-Province_Type_B_Power(MW)": "Type_B_Power",
    "Guangdong-Hong_Kong_Interconnector(MW)": "GZ_HK_Interconnector",
    "System_Load(MW)": "System_Load",
    "West-to-East_Power_Transmission(MW)": "West_to_East",
}


def read_wide_price(path: str) -> pd.DataFrame:
    """宽表小时价（Date + Hour_00..Hour_23）→ 长表 (ds, price)。"""
    df = pd.read_csv(path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]  # 去 BOM
    id_vars = [c for c in df.columns if c != "Date" and c.startswith("Hour_")]
    long = df.melt(id_vars=["Date"], value_vars=id_vars, var_name="Hour", value_name="price")
    long["hour"] = long["Hour"].str.replace("Hour_", "", regex=False).astype(int)
    long["ds"] = pd.to_datetime(long["Date"]) + pd.to_timedelta(long["hour"], unit="h")
    return long[["ds", "price"]]


def read_15min_data(path: str) -> pd.DataFrame:
    """15 分钟数据（Date + Time + MW 列）→ 小时均值 (ds, 各 MW 列)。"""
    df = pd.read_csv(path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    df["hour"] = df["Time"].str.split(":").str[0].astype(int)
    df["ds"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["hour"], unit="h")
    grp = df.groupby("ds", as_index=False)[list(MW_COLS.keys())].mean()
    grp.columns = ["ds"] + [MW_COLS[c] for c in MW_COLS.keys()]
    return grp


def main() -> None:
    y = read_wide_price(str(RAW / "Guangdong_Electricity_Spot_Real-time_Prices_2024-2025.csv")).rename(
        columns={"price": "y"}
    )
    da_price = read_wide_price(str(RAW / "Guangdong_Electricity_Spot_Day-ahead_Prices_2024-2025.csv")).rename(
        columns={"price": "da_price"}
    )
    x = (
        read_15min_data(str(RAW / "Guangdong_Electricity_Spot_Real-time_Data_2024-2025.csv"))
        .add_prefix("rt_")
        .rename(columns={"rt_ds": "ds"})
    )
    z = (
        read_15min_data(str(RAW / "Guangdong_Electricity_Spot_Day-ahead_Data_2024-2025.csv"))
        .add_prefix("da_")
        .rename(columns={"da_ds": "ds"})
    )

    out = (
        y.merge(da_price, on="ds", how="inner")
        .merge(x, on="ds", how="inner")
        .merge(z, on="ds", how="inner")
    )
    out = out.sort_values("ds").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"[OK] 输出 {OUT}")
    print(f"  形状: {out.shape}  时间范围: {out.ds.min()} -> {out.ds.max()}")
    print(f"  NaN: y={out.y.isna().sum()}  da_price={out.da_price.isna().sum()}  "
          f"rt={out.filter(like='rt_').isna().sum().sum()}  da={out.filter(like='da_').isna().sum().sum()}")
    print(f"  列: {list(out.columns)}")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
