
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ------------------------- I/O configuration ----------------------------------
INPUT_XLSX  = Path("prod_cons.xlsx")   
OUTPUT_XLSX = Path("community_results_by_month.xlsx")

# ------------------------- Reading and parsing --------------------------------
def read_excel_any(path: Path) -> pd.DataFrame:

    try:
        df = pd.read_excel(path, header=[0, 1])
        # Heuristics: expect at least one ('B\d+', 'load') column
        has_buildings = any(
            isinstance(c, tuple) and re.match(r"^B\d+$", str(c[0])) for c in df.columns
        )
        if has_buildings:
            return df
    except Exception:
        pass
    # Fallback
    return pd.read_excel(path)

def find_datetime_columns(df: pd.DataFrame):
    """
    Locate ('*', 'Date'), ('*', 'Month'), ('*', 'Time') when MultiIndex is present,
    or 'Date','Month','Time' when flat.
    Accept also 'Day' in place of 'Date'.
    """
    def find_sub(name_options):
        lower_opts = set(name_options)
        if isinstance(df.columns, pd.MultiIndex):
            for c in df.columns:
                sub = str(c[1]).strip().lower()
                if sub in lower_opts:
                    return c
        else:
            for c in df.columns:
                if str(c).strip().lower() in lower_opts:
                    return c
        return None

    date_col  = find_sub(["date", "day"])
    month_col = find_sub(["month"])
    time_col  = find_sub(["time"])
    if date_col is None or month_col is None or time_col is None:
        raise ValueError("Could not find Date/Day, Month, and Time columns in the header.")
    return date_col, month_col, time_col

def parse_time_to_start(s: str) -> str:
    """
    Accept 'HH:MM - HH:MM' or 'HH:MM'; return 'HH:MM' (start).
    """
    s = str(s)
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        return f"{hh:02d}:{mm:02d}"
    m = re.match(r"\s*(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    return "00:00"

def build_datetime_index(df: pd.DataFrame, assumed_year: int | None = None) -> pd.DatetimeIndex:
    """
    Build a DatetimeIndex from Date (day of month), Month, and Time (range or hh:mm).
    If assumed_year is None, infer a plausible year from the data start (defaults to 2024).
    """
    date_col, month_col, time_col = find_datetime_columns(df)

    day_series   = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")
    month_series = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
    time_series  = pd.Series(df[time_col].astype(str).map(parse_time_to_start))

    if assumed_year is None:
        first_m = int(month_series.dropna().iloc[0])
        last_m  = int(month_series.dropna().iloc[-1])
        assumed_year = 2024 if first_m == 1 and last_m in (12, 1) else 2024

    dt_date = pd.to_datetime(
        day_series.astype(str) + "-" + month_series.astype(str) + "-" + str(assumed_year),
        dayfirst=True, errors="coerce"
    )
    dt_time = pd.to_datetime(time_series, format="%H:%M", errors="coerce").dt.time

    if dt_date.isna().any() or pd.isna(dt_time).any():
        raise ValueError("Failed to construct DateTime. Please verify the formats of Date/Month/Time.")
    dt = pd.to_datetime(dt_date.dt.date.astype(str) + " " + pd.Series(dt_time).astype(str))
    return pd.DatetimeIndex(dt)

def list_buildings(df: pd.DataFrame) -> list[str]:
    """
    Identify buildings 'B\d+' from the first (top) level of a MultiIndex header.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected a two-row header with buildings at top level.")
    bset = {c[0] for c in df.columns if isinstance(c, tuple) and re.match(r"^B\d+$", str(c[0]))}
    return sorted(bset, key=lambda x: int(x[1:]))

def find_subcol(df: pd.DataFrame, bld: str, target: str) -> tuple:
    """
    Retrieve the precise MultiIndex column for a building and target subcolumn.
    target ∈ {'load','pv_gen','grid_export','grid_import'}.
    """
    assert isinstance(df.columns, pd.MultiIndex)
    synonyms = {
        "load": ["load"],
        "pv_gen": ["pv gen", "pv_gen", "pvgen"],
        "grid_export": ["grid export", "export"],
        "grid_import": ["grid import", "import"],
    }
    wanted = set(synonyms[target])
    for c in df.columns:
        if isinstance(c, tuple) and c[0] == bld:
            sub = str(c[1]).strip().lower()
            if sub in wanted:
                return c
    raise KeyError(f"Subcolumn '{target}' not found for {bld}.")

# --------------------------- Load and normalize -------------------------------
df = read_excel_any(INPUT_XLSX)
dt_index = build_datetime_index(df)  # robust to 'Date/Month/Time' with ranges like '00:00 - 01:00'
buildings = list_buildings(df)

# Build a clean wide table with MultiIndex columns (building, sub)
wide = pd.DataFrame(index=dt_index)
for b in buildings:
    for target in ["load", "pv_gen", "grid_export", "grid_import"]:
        col = find_subcol(df, b, target)
        wide[(b, target)] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

# Sort columns for a stable order
wide = wide.sort_index(axis=1, level=[0, 1])

# --------------------- Per–timestep energy accounting ------------------------
loads   = {b: wide[(b, "load")].clip(lower=0.0)   for b in buildings}
pv_gens = {b: wide[(b, "pv_gen")].clip(lower=0.0) for b in buildings}

# 1) Self-consumption
self_used = {}
for b in buildings:
    self_used[b] = pd.concat([loads[b], pv_gens[b]], axis=1).min(axis=1)

# 2) Deficits and surpluses
deficits = {b: (loads[b] - self_used[b]).clip(lower=0.0) for b in buildings}
surplus  = {b: (pv_gens[b] - self_used[b]).clip(lower=0.0) for b in buildings}

df_def = pd.DataFrame(deficits)
df_sur = pd.DataFrame(surplus)

sum_def = df_def.sum(axis=1)
sum_sur = df_sur.sum(axis=1)
shared_total = pd.concat([sum_def, sum_sur], axis=1).min(axis=1)

# Proportional allocation within the community
w_def = df_def.div(sum_def.replace(0.0, np.nan), axis=0).fillna(0.0)
w_sur = df_sur.div(sum_sur.replace(0.0, np.nan), axis=0).fillna(0.0)

df_shared_in  = w_def.mul(shared_total, axis=0)
df_shared_out = w_sur.mul(shared_total, axis=0)

# Residual exchange with the external grid
df_from_grid = df_def - df_shared_in
df_to_grid   = df_sur - df_shared_out

# Assemble per–timestep metrics (if needed downstream)
metrics = [
    "load", "pv_gen", "self_used",
    "shared_in", "shared_out",
    "from_grid", "to_grid",
    "net_shared", "deficit",
    "share_in_of_deficit", "total_surplus",
    "share_out_of_surplus", "self_use_pct"
]
per_step = {}
for b in buildings:
    s_load = loads[b]
    s_pv   = pv_gens[b]
    s_self = self_used[b]
    s_def  = (s_load - s_self).clip(lower=0.0)
    s_sur  = (s_pv   - s_self).clip(lower=0.0)

    df_b = pd.DataFrame({
        "load":        s_load,
        "pv_gen":      s_pv,
        "self_used":   s_self,
        "shared_in":   df_shared_in[b],
        "shared_out":  df_shared_out[b],
        "from_grid":   df_from_grid[b],
        "to_grid":     df_to_grid[b],
    })
    df_b["net_shared"] = df_b["shared_in"] - df_b["shared_out"]
    df_b["deficit"]    = s_def
    df_b["total_surplus"] = s_sur

    # Percentages for diagnostics at time-step level (guarded divisions)
    df_b["share_in_of_deficit"]  = np.where(df_b["deficit"] > 0,
                                            100.0 * df_b["shared_in"] / df_b["deficit"], 0.0)
    df_b["share_out_of_surplus"] = np.where(df_b["total_surplus"] > 0,
                                            100.0 * df_b["shared_out"] / df_b["total_surplus"], 0.0)
    df_b["self_use_pct"] = np.where(df_b["pv_gen"] > 0,
                                    100.0 * df_b["self_used"] / df_b["pv_gen"], 0.0)

    per_step[b] = df_b

per_step = pd.concat(per_step, axis=1)  # MultiIndex columns: (building, metric)
# Reorder columns for readability
per_step = per_step.reindex(
    columns=pd.MultiIndex.from_product([buildings, [
        "load", "pv_gen", "self_used",
        "shared_in", "shared_out", "from_grid", "to_grid",
        "net_shared", "deficit", "share_in_of_deficit",
        "total_surplus", "share_out_of_surplus", "self_use_pct"
    ]])
)

# ---------------------------- Annual aggregates ------------------------------
annual_rows = []
for b in buildings:
    s = per_step[b]
    load_sum        = s["load"].sum()
    pv_sum          = s["pv_gen"].sum()
    self_used_sum   = s["self_used"].sum()
    shared_in_sum   = s["shared_in"].sum()
    shared_out_sum  = s["shared_out"].sum()
    from_grid_sum   = s["from_grid"].sum()
    to_grid_sum     = s["to_grid"].sum()
    net_shared_sum  = s["net_shared"].sum()
    deficit_sum     = s["deficit"].sum()
    surplus_sum     = s["total_surplus"].sum()

    share_in_of_deficit_pct  = 100.0 * shared_in_sum  / deficit_sum if deficit_sum  > 0 else 0.0
    share_out_of_surplus_pct = 100.0 * shared_out_sum / surplus_sum if surplus_sum > 0 else 0.0
    self_use_pct_annual      = 100.0 * self_used_sum  / pv_sum      if pv_sum      > 0 else 0.0

    annual_rows.append({
        "building": b,
        "load": load_sum,
        "pv_gen": pv_sum,
        "self_used": self_used_sum,
        "shared_in": shared_in_sum,
        "shared_out": shared_out_sum,
        "from_grid": from_grid_sum,
        "to_grid": to_grid_sum,
        "net_shared": net_shared_sum,
        "deficit": deficit_sum,
        "share_in_of_deficit_pct": share_in_of_deficit_pct,
        "total_surplus": surplus_sum,
        "share_out_of_surplus_pct": share_out_of_surplus_pct,
        "self_use_pct": self_use_pct_annual,
    })

annual = pd.DataFrame(annual_rows).set_index("building")
annual = annual[[
    "load", "pv_gen", "self_used",
    "shared_in", "shared_out",
    "from_grid", "to_grid", "net_shared",
    "deficit", "share_in_of_deficit_pct",
    "total_surplus", "share_out_of_surplus_pct",
    "self_use_pct"
]]

# ---------------------------- Monthly aggregates -----------------------------
# We aggregate the same indicators per calendar month (YYYY-MM) for each building.
monthly_rows = []
for b in buildings:
    s = per_step[b].copy()
    # Use PeriodIndex for clean month labels, then convert to string 'YYYY-MM'
    month_periods = s.index.to_period("M")
    # Precompute monthly sums for additive metrics
    g = s.groupby(month_periods)

    load_sum_m        = g["load"].sum()
    pv_sum_m          = g["pv_gen"].sum()
    self_used_sum_m   = g["self_used"].sum()
    shared_in_sum_m   = g["shared_in"].sum()
    shared_out_sum_m  = g["shared_out"].sum()
    from_grid_sum_m   = g["from_grid"].sum()
    to_grid_sum_m     = g["to_grid"].sum()
    net_shared_sum_m  = g["net_shared"].sum()
    deficit_sum_m     = g["deficit"].sum()
    surplus_sum_m     = g["total_surplus"].sum()

    # Energy-weighted percentages per month (guard divisions by zero)
    share_in_of_deficit_pct_m  = 100.0 * (shared_in_sum_m / deficit_sum_m.replace(0.0, np.nan))
    share_out_of_surplus_pct_m = 100.0 * (shared_out_sum_m / surplus_sum_m.replace(0.0, np.nan))
    self_use_pct_m             = 100.0 * (self_used_sum_m  / pv_sum_m.replace(0.0, np.nan))

    # Replace NaNs (months with zero denominators) by 0.0 for readability
    share_in_of_deficit_pct_m  = share_in_of_deficit_pct_m.fillna(0.0)
    share_out_of_surplus_pct_m = share_out_of_surplus_pct_m.fillna(0.0)
    self_use_pct_m             = self_use_pct_m.fillna(0.0)

    # Assemble rows
    for m in load_sum_m.index:
        monthly_rows.append({
            "month": str(m),     # 'YYYY-MM'
            "building": b,
            "load": load_sum_m.loc[m],
            "pv_gen": pv_sum_m.loc[m],
            "self_used": self_used_sum_m.loc[m],
            "shared_in": shared_in_sum_m.loc[m],
            "shared_out": shared_out_sum_m.loc[m],
            "from_grid": from_grid_sum_m.loc[m],
            "to_grid": to_grid_sum_m.loc[m],
            "net_shared": net_shared_sum_m.loc[m],
            "deficit": deficit_sum_m.loc[m],
            "share_in_of_deficit_pct": share_in_of_deficit_pct_m.loc[m],
            "total_surplus": surplus_sum_m.loc[m],
            "share_out_of_surplus_pct": share_out_of_surplus_pct_m.loc[m],
            "self_use_pct": self_use_pct_m.loc[m],
        })

monthly = pd.DataFrame(monthly_rows).set_index(["month", "building"]).sort_index()
monthly = monthly[[
    "load", "pv_gen", "self_used",
    "shared_in", "shared_out",
    "from_grid", "to_grid", "net_shared",
    "deficit", "share_in_of_deficit_pct",
    "total_surplus", "share_out_of_surplus_pct",
    "self_use_pct"
]]

# ------------------------------- Output --------------------------------------
with pd.ExcelWriter(OUTPUT_XLSX) as xlw:
    # Time-step sheets by building
    for b in buildings:
        df_b = per_step[b].copy()
        df_b.index.name = "DateTime"
        df_b.to_excel(xlw, sheet_name=b)
    # Annual summary
    annual.to_excel(xlw, sheet_name="Annual_Summary")
    # Monthly summary (YYYY-MM × building)
    monthly.to_excel(xlw, sheet_name="Monthly_Summary")

# Console print (decimal point)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print("Annual indicators by building (kWh and %):")
print(annual)
print("\nMonthly indicators (kWh and %) by month and building:")
print(monthly)
print(f"\nResults saved to: {OUTPUT_XLSX.resolve()}")
