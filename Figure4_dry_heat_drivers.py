import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pymannkendall as mk
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import os

# -------------------- Font Settings ---------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'   # Roman font for math mode
plt.rcParams['mathtext.it'] = 'Arial:italic'  # Italic font for math mode
plt.rcParams['mathtext.bf'] = 'Arial:bold'    # Bold font for math mode

def significant_mk_rows(array2d, significance_level=0.05):
    """
    Performs Mann-Kendall trend test on each row of a 2D array.
    Returns the slope and p-value for each row.

    Parameters:
        array2d : 2D numpy.ndarray
            Input array where each row represents a time series.
        significance_level : float
            Significance level, default is 0.05.

    Returns:
        slopes : numpy.ndarray
            Trend slope for each row.
        p_values : numpy.ndarray
            p-value for each row.
    """
    nrows = array2d.shape[0]
    slopes = np.full(nrows, np.nan)
    p_values = np.full(nrows, np.nan)

    for i in range(nrows):
        row = array2d[i, :]
        if np.all(np.isnan(row)):  # Skip if all values are NaN
            continue
        try:
            mk_result = mk.original_test(row, alpha=significance_level)
            # Normalizing slope by the mean of the row
            slopes[i] = mk_result.slope / np.mean(row)
            p_values[i] = mk_result.p
        except Exception as e:
            print(f"Row {i} failed: {e}")
            continue

    return slopes, p_values

def fill_results(target, target_sig, arrays):
    """Simplifies the assignment logic for bulk trend calculations."""
    for j, arr in enumerate(arrays):
        target[:, j], target_sig[:, j] = significant_mk_rows(arr)

base_path1 = '.../Drivers/3days_SPEI128/com/'

# List of filename prefixes (excluding extensions)
prefixes = [
    "_com_Heat_len_in_com_Heat",
    "_com_Dry_len_in_com_Dry",
    "_com_Heat_len_in_com_dis",
    "_com_Dry_len_in_com_dis",
    "_year_Heat_len_in_year_dis",
    "_year_Dry_len_in_year_dis",   
    "_com_Heat_fre_in_year_Heat",
    "_year_Heat_fre_in_year_dis",
    "_com_Dry_fre_in_year_Dry",
    "_year_Dry_fre_in_year_dis",
]

dataset = ['Observed', 'ALL', 'NAT', 'ssp245', 'ssp585']   # Directory names
dataset_1 = ['Obs', 'ALL', 'NAT', 'ssp245', 'ssp585']
suffix1 = "_temporal_change_1951_2013.npy"  # Historical suffix
suffix2 = "_temporal_change_2016_2097.npy"  # Future suffix

# =========== Observed Features =====================
obs_ratios = {}
for i, prefix in enumerate(prefixes, start=1):
    fpath = base_path1 + dataset[0] + "/event_ratio/" + dataset_1[0] + prefix + suffix1
    # Loading specific indices and slicing time steps
    obs_ratios[i] = np.load(fpath)[[0, 1, 8, 9], 9:-9]

# =========== ALL (All Forcings) Features ============
ALL_ratios = {}
for i, prefix in enumerate(prefixes, start=1):
    fpath = base_path1 + dataset[1] + "/event_ratio/" + dataset_1[1] + prefix + suffix1
    ALL_ratios[i] = np.load(fpath)[:, 9:]

# =========== NAT (Natural Forcings) Features ========
NAT_ratios = {}
for i, prefix in enumerate(prefixes, start=1):
    fpath = base_path1 + dataset[2] + "/event_ratio/" + dataset_1[2] + prefix + suffix1
    NAT_ratios[i] = np.load(fpath)[:, 9:]
    
# =========== SSP245 Features ========================
ssp245_ratios = {}
for i, prefix in enumerate(prefixes, start=1):
    fpath = base_path1 + dataset[3] + "/event_ratio/" + dataset_1[3] + prefix + suffix2
    ssp245_ratios[i] = np.load(fpath)

# =========== SSP585 Features ========================
ssp585_ratios = {}
for i, prefix in enumerate(prefixes, start=1):
    fpath = base_path1 + dataset[4] + "/event_ratio/" + dataset_1[4] + prefix + suffix2
    ssp585_ratios[i] = np.load(fpath)

# ============= Initialize Trend Arrays ================
obs    = np.full((4, 10), np.nan);  obs_sig    = np.full((4, 10), np.nan)
ALL    = np.full((16, 10), np.nan); ALL_sig    = np.full((16, 10), np.nan)
NAT    = np.full((11, 10), np.nan); NAT_sig    = np.full((11, 10), np.nan)
ssp245 = np.full((13, 10), np.nan); ssp245_sig = np.full((13, 10), np.nan)
ssp585 = np.full((13, 10), np.nan); ssp585_sig = np.full((13, 10), np.nan)

# ============= Calculate Trends ================
# Obs
fill_results(obs, obs_sig, obs_ratios.values())
# ALL
fill_results(ALL, ALL_sig, ALL_ratios.values())
# NAT
fill_results(NAT, NAT_sig, NAT_ratios.values())
# ssp245
fill_results(ssp245, ssp245_sig, ssp245_ratios.values())
# ssp585
fill_results(ssp585, ssp585_sig, ssp585_ratios.values())

# ================== Construct Data Structure ====================
Period = ["SSP585", "SSP245", "NAT", "ALL", "OBS"]

Ratio = ["CPL", "DPL", "DPL_Year", "CPF_Year", "DPF_Year"]

Event = ["Heatwave", "Drought"]

# Mapping 10 columns -> (Ratio, Event) pairs
ratio_event_map = [
    ("CPL", "Heatwave"),
    ("CPL", "Drought"),
    ("DPL", "Heatwave"),
    ("DPL", "Drought"),
    ("DPL_Year", "Heatwave"),
    ("DPL_Year", "Drought"),
    ("CPF_Year", "Heatwave"),
    ("CPF_Year", "Drought"),
    ("DPF_Year", "Heatwave"),
    ("DPF_Year", "Drought"),
]

# Map results for each Period
results = {
    "OBS": (obs, obs_sig),
    "ALL": (ALL, ALL_sig),
    "NAT": (NAT, NAT_sig),
    "SSP245": (ssp245, ssp245_sig),
    "SSP585": (ssp585, ssp585_sig),
}

records = []

for period, (trend_arr, sig_arr) in results.items():
    n_models = trend_arr.shape[0]  # Number of rows (models/samples)
    for model_id in range(n_models):
        for j, (ratio, event) in enumerate(ratio_event_map):
            records.append({
                "Period": period,
                "Ratio": ratio,
                "Event": event,
                "Model_ID": model_id + 1,
                "trend": trend_arr[model_id, j],
                "sig": sig_arr[model_id, j],
            })

df = pd.DataFrame(records)

# ============ Data Summary ============ #
summary = []
grouped = df.groupby(["Period", "Ratio", "Event"])

for (period, ratio, event), group in grouped:
    values = group["trend"].dropna().values
    if len(values) == 0:
        continue

    median = np.median(values)
    p5 = np.percentile(values, 5)
    p95 = np.percentile(values, 95)

    # Significance check: If >90% of models show the same sign
    positive_ratio = np.mean(values > 0)
    negative_ratio = np.mean(values < 0)
    sig_flag = (positive_ratio >= 0.9) or (negative_ratio >= 0.9)

    summary.append({
        "Period": period,
        "Ratio": ratio,
        "Event": event,
        "median": median,
        "p5": p5,
        "p95": p95,
        "sig_flag": sig_flag,
        "ratio_event": f"{ratio}_{event}"  # For sorting
    })

df_summary = pd.DataFrame(summary)

# Define sorting order
period_order = ["OBS", "ALL", "NAT", "SSP245", "SSP585"]
ratio_event_order = [f"{r}_{e}" for r, e in ratio_event_map]

# Apply categorical sorting
df_summary["Period"] = pd.Categorical(df_summary["Period"], categories=period_order, ordered=True)
df_summary["ratio_event"] = pd.Categorical(df_summary["ratio_event"], categories=ratio_event_order, ordered=True)

df_summary = df_summary.sort_values(["Period", "ratio_event"]).reset_index(drop=True)

# ================ Plotting ===============================
# 1) Pivot to a wide table to group Heatwave/Drought pairs
period_order = ["OBS", "ALL", "NAT", "SSP245", "SSP585"]
ratio_order   = ["CPL", "DPL", "DPL_Year", "CPF_Year", "DPF_Year"]

_need_cols = ["Period", "Ratio", "Event", "median", "p5", "p95", "sig_flag"]
tmp = df_summary[_need_cols].copy()

# Pivot to wide format: columns become multi-level (metric, Event)
pv = tmp.pivot(index=["Period", "Ratio"], columns="Event")

# Flatten column names: median_heatwave, p5_drought, etc.
pv.columns = [f"{metric}_{event.lower()}" for metric, event in pv.columns]
pv = pv.reset_index()

keep_cols = [
    "Period", "Ratio",
    "median_heatwave", "p5_heatwave", "p95_heatwave", "sig_flag_heatwave",
    "median_drought",  "p5_drought",  "p95_drought",  "sig_flag_drought",
]
df_pair = pv[[c for c in keep_cols if c in pv.columns]].copy()

# Sort by defined order
df_pair["Period"] = pd.Categorical(df_pair["Period"], categories=period_order, ordered=True)
df_pair["Ratio"]  = pd.Categorical(df_pair["Ratio"],  categories=ratio_order,   ordered=True)
df_pair = df_pair.sort_values(["Period", "Ratio"]).reset_index(drop=True)

# Drop incomplete groups where either Heatwave or Drought data is missing
mask_complete = df_pair[["median_heatwave", "median_drought"]].notna().all(axis=1)
df_pair = df_pair[mask_complete].reset_index(drop=True)

# 2) Draw: Each (Period, Ratio) is a group with two bars (Heatwave, Drought)
bar_height = 0.2    # Thickness of a single bar
inner_gap   = -0.01  # Space between bars within a group
group_gap   = 0.45   # Space between adjacent groups
period_gap  = 0.70   # Extra spacing between different Periods

palette_all = sns.color_palette("tab20c", n_colors=20)
# Select specific colors from palette
palette = [palette_all[i] for i in [4, 12]]
color_map = {"heat": palette[0], "dry": palette[1]}
alpha_set = 0.9

draw_rows = []         # For storing expanded plotting data
period_centers = {}    # Y-axis centers for each Period label
y_cursor = 0.0

for period in period_order:
    subP = df_pair[df_pair["Period"] == period]
    if subP.empty:
        continue

    centers_in_period = []
    for _, row in subP.iterrows():
        y_center = y_cursor

        # Position bars for Heat and Dry above/below the center
        y_heat = y_center + (bar_height/2 + inner_gap/2)
        y_dry  = y_center - (bar_height/2 + inner_gap/2)

        draw_rows.append({
            "y": y_heat,
            "median": row["median_heatwave"],
            "p5": row["p5_heatwave"],
            "p95": row["p95_heatwave"],
            "sig": bool(row["sig_flag_heatwave"]) if pd.notna(row["sig_flag_heatwave"]) else False,
            "event": "heat",
            "Period": row["Period"],
            "Ratio": row["Ratio"],
        })
        draw_rows.append({
            "y": y_dry,
            "median": row["median_drought"],
            "p5": row["p5_drought"],
            "p95": row["p95_drought"],
            "sig": bool(row["sig_flag_drought"]) if pd.notna(row["sig_flag_drought"]) else False,
            "event": "dry",
            "Period": row["Period"],
            "Ratio": row["Ratio"],
        })

        centers_in_period.append(y_center)
        y_cursor += (2*bar_height + inner_gap + group_gap)

    if centers_in_period:
        period_centers[period] = (centers_in_period[0] + centers_in_period[-1]) / 2.0
        y_cursor += period_gap

df_draw = pd.DataFrame(draw_rows)

# ===== Main Plotting =====
fig, ax = plt.subplots(figsize=(5, 8))

# Vertical zero line
ax.axvline(0, color="gray", linestyle="--", lw=1)

# Plot bars + error bars + significance stars
for r in df_draw.itertuples():
    ax.barh(
        y=r.y,
        width=r.median,
        height=bar_height,
        color=color_map[r.event],
        edgecolor="none",
        zorder=2,
        alpha=alpha_set
    )
    # Uncertainty: p5 to p95
    if pd.notna(r.p5) and pd.notna(r.p95):
        ax.hlines(y=r.y, xmin=r.p5, xmax=r.p95, color="k", lw=0.7, zorder=3)

    # Significance markers
    if r.sig and pd.notna(r.median):
        # Place star slightly outside the error bar range
        if r.median >= 0:
            xpos, ha = (r.p95 if pd.notna(r.p95) else r.median) * 1.02, "left"
        else:
            xpos, ha = (r.p5 if pd.notna(r.p5) else r.median) * 1.02, "right"
        ax.text(xpos, r.y-0.18, "*", va="center", ha=ha, fontsize=10, color="k", zorder=4)

# Y-axis: Period labels
yticks = [period_centers[p] for p in period_order if p in period_centers]
ax.set_yticks(yticks)
ax.set_yticklabels([p for p in period_order if p in period_centers])

# Set Y limits
if len(df_draw) > 0:
    ax.set_ylim(min(df_draw["y"]) - bar_height - 0.3, max(df_draw["y"]) + bar_height + 0.3)

# X-axis settings
ax.set_xlabel("Trend (%/year)")
ax.set_xlim(-0.015, 0.025)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.005))

def custom_formatter(x, pos):
    if np.isclose(x, 0):
        return "0"
    elif np.isclose(x, -0.01) or np.isclose(x, 0.01) or np.isclose(x, 0.02):
        return f"{x:.2f}"
    else:
        return ""
ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

# Manual Legend
legend_elems = [
    Line2D([0], [0], color=color_map["heat"], lw=8, label="Heatwave", alpha=alpha_set),
    Line2D([0], [0], color=color_map["dry"],  lw=8, label="Drought", alpha=alpha_set),
]
ax.legend(handles=legend_elems, frameon=False, loc="lower left")

# Remove standard Y-axis ticks
ax.set_yticks([])
ax.set_ylabel("")
ax.tick_params(axis="y", left=False)
ax.spines['left'].set_visible(False)

# Styling
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(width=0.5)

# Horizontal separation lines
ax.axhline(y=4.12, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=9.0, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=13.9, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=18.8, color='gray', linestyle='--', linewidth=1)

# Formula annotations
ax.text(0.0015, 9.6, r"$\mathrm{Dur}_\mathrm{comp}$/ $\mathrm{Dur}_\mathrm{dry-comp}$ or $\mathrm{Dur}_\mathrm{heat-comp}$", 
        fontsize=10, color="k", zorder=4)
ax.text(0.0015, 10.5, r"$\mathrm{Dur}_\mathrm{dry-comp}$ or $\mathrm{Dur}_\mathrm{heat-comp}$ /$\mathrm{Dur}_\mathrm{dist-comp}$", 
        fontsize=10, color="k", zorder=4)
ax.text(0.0015, 11.35, r"$\mathrm{Dur}_\mathrm{dry}$ or $\mathrm{Dur}_\mathrm{heat}$ /$\mathrm{Dur}_\mathrm{dist}$",  
        fontsize=10, color="k", zorder=4)
ax.text(0.0015, 12.2, r"$\mathrm{Fre}_\mathrm{comp}$/ $\mathrm{Fre}_\mathrm{dry}$ or $\mathrm{Fre}_\mathrm{heat}$",  
        fontsize=10, color="k", zorder=4)
ax.text(0.0015, 13.0, r"$\mathrm{Fre}_\mathrm{dry}$ or $\mathrm{Fre}_\mathrm{heat}$ /$\mathrm{Fre}_\mathrm{dist}$",  
        fontsize=10, color="k", zorder=4)

# Subplot Label (b)
ax.text(0, 1, 'b', transform=ax.transAxes, 
        fontsize=13, fontweight='bold', va='top', ha='left')