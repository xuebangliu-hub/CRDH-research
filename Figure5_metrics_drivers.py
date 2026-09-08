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
plt.rcParams['mathtext.rm'] = 'Arial'         # Roman style for math mode
plt.rcParams['mathtext.it'] = 'Arial:italic'  # Italic style for math mode
plt.rcParams['mathtext.bf'] = 'Arial:bold'    # Bold style for math mode

def significant_mk_rows(array2d, significance_level=0.05):
    """
    Performs Mann-Kendall trend test on each row of a 2D array.
    Returns the slope (normalized by the mean) and p-value for each row.

    Parameters:
        array2d : 2D numpy.ndarray (rows = time series)
        significance_level : float (default 0.05)

    Returns:
        slopes : Normalized trend slopes
        p_values : Significance p-values
    """
    nrows = array2d.shape[0]
    slopes = np.full(nrows, np.nan)
    p_values = np.full(nrows, np.nan)

    for i in range(nrows):
        row = array2d[i, :]
        if np.all(np.isnan(row)): 
            continue
        try:
            mk_result = mk.original_test(row, alpha=significance_level)
            # Normalizing slope by mean for relative comparison
            slopes[i] = mk_result.slope / np.mean(row)
            p_values[i] = mk_result.p
        except Exception as e:
            print(f"Row {i} failed: {e}")
            continue

    return slopes, p_values

def fill_results(target, target_sig, arrays):
    """Utility function to batch calculate trends for different variable lists."""
    for j, arr in enumerate(arrays):
        target[:, j], target_sig[:, j] = significant_mk_rows(arr)

base_path1 = '.../Drivers/3days_SPEI128/com/'

# =========== Load Observation Attributes (OBS) =====================    
obs_com_len = np.load(base_path1 + '/Observed/event_com_temporal/Obs_com_len_year_temporal_change_1951_2023.npy')[[0, 1, 8, 9], 9:-9]
obs_dis_len_com = np.load(base_path1 + '/Observed/event_com_temporal/Obs_dis_len_com_temporal_change_1951_2023.npy')[[0, 1, 8, 9], 9:-9]
obs_dis_len_year = np.load(base_path1 + '/Observed/event_temporal/Obs_dis_len_year_temporal_change_1950_2023.npy')[[0, 1, 8, 9], 9:-9]
obs_com_fre = np.load(base_path1 + '/Observed/event_com_temporal/Obs_com_fre_year_temporal_change_1951_2023.npy')[[0, 1, 8, 9], 9:-9]
obs_dis_fre = np.load(base_path1 + '/Observed/event_com_temporal/Obs_dis_fre_year_temporal_change_1951_2023.npy')[[0, 1, 8, 9], 9:-9]

# =========== Load CMIP6 Historical Attributes (ALL & NAT) =====================    
ALL_com_len = np.load(base_path1 + '/ALL/event_com_temporal/All_com_len_year_temporal_change_1951_2013.npy')[:,9:]
ALL_dis_len_com = np.load(base_path1 + '/ALL/event_com_temporal/All_dis_len_com_temporal_change_1951_2013.npy')[:,9:]
ALL_dis_len_year = np.load(base_path1 + '/ALL/event_temporal/ALL_dis_len_year_temporal_change_1951_2013.npy')[:,9:]
ALL_com_fre = np.load(base_path1 + '/ALL/event_com_temporal/ALL_com_fre_year_temporal_change_1951_2013.npy')[:,9:]
ALL_dis_fre = np.load(base_path1 + '/ALL/event_com_temporal/ALL_dis_fre_year_temporal_change_1951_2013.npy')[:,9:]

NAT_com_len = np.load(base_path1 + '/NAT/event_com_temporal/NAT_com_len_year_temporal_change_1951_2013.npy')[:,9:]
NAT_dis_len_com = np.load(base_path1 + '/NAT/event_com_temporal/NAT_dis_len_com_temporal_change_1951_2013.npy')[:,9:]
NAT_dis_len_year = np.load(base_path1 + '/NAT/event_temporal/NAT_dis_len_year_temporal_change_1951_2013.npy')[:,9:]
NAT_com_fre = np.load(base_path1 + '/NAT/event_com_temporal/NAT_com_fre_year_temporal_change_1951_2013.npy')[:,9:]
NAT_dis_fre = np.load(base_path1 + '/NAT/event_com_temporal/NAT_dis_fre_year_temporal_change_1951_2013.npy')[:,9:]

# =========== Load Future Attributes (SSP245 & SSP585) =====================
ssp245_com_len = np.load(base_path1 + '/ssp245/event_com_temporal/ssp245_com_len_year_temporal_change_2016_2097.npy')
ssp245_dis_len_com = np.load(base_path1 + '/ssp245/event_com_temporal/ssp245_dis_len_com_temporal_change_2016_2097.npy')
ssp245_dis_len_year = np.load(base_path1 + '/ssp245/event_temporal/ssp245_dis_len_year_temporal_change_2016_2099.npy')
ssp245_com_fre = np.load(base_path1 + '/ssp245/event_com_temporal/ssp245_com_fre_year_temporal_change_2016_2097.npy')
ssp245_dis_fre = np.load(base_path1 + '/ssp245/event_com_temporal/ssp245_dis_fre_year_temporal_change_2016_2097.npy')

ssp585_com_len = np.load(base_path1 + '/ssp585/event_com_temporal/ssp585_com_len_year_temporal_change_2016_2097.npy')
ssp585_dis_len_com = np.load(base_path1 + '/ssp585/event_com_temporal/ssp585_dis_len_com_temporal_change_2016_2097.npy')
ssp585_dis_len_year = np.load(base_path1 + '/ssp585/event_temporal/ssp585_dis_len_year_temporal_change_2016_2099.npy')
ssp585_com_fre = np.load(base_path1 + '/ssp585/event_com_temporal/ssp585_com_fre_year_temporal_change_2016_2097.npy')
ssp585_dis_fre = np.load(base_path1 + '/ssp585/event_com_temporal/ssp585_dis_fre_year_temporal_change_2016_2097.npy')

# ============= Calculate Trends ================
obs = np.full((4, 5), np.nan);     obs_sig = np.full((4, 5), np.nan)
ALL = np.full((16, 5), np.nan);    ALL_sig = np.full((16, 5), np.nan)
NAT = np.full((11, 5), np.nan);    NAT_sig = np.full((11, 5), np.nan)
ssp245 = np.full((13, 5), np.nan); ssp245_sig = np.full((13, 5), np.nan)
ssp585 = np.full((13, 5), np.nan); ssp585_sig = np.full((13, 5), np.nan)

# Process all groups
fill_results(obs, obs_sig, [obs_com_len, obs_dis_len_com, obs_dis_len_year, obs_com_fre, obs_dis_fre])
fill_results(ALL, ALL_sig, [ALL_com_len, ALL_dis_len_com, ALL_dis_len_year, ALL_com_fre, ALL_dis_fre])
fill_results(NAT, NAT_sig, [NAT_com_len, NAT_dis_len_com, NAT_dis_len_year, NAT_com_fre, NAT_dis_fre])
fill_results(ssp245, ssp245_sig, [ssp245_com_len, ssp245_dis_len_com, ssp245_dis_len_year, ssp245_com_fre, ssp245_dis_fre])
fill_results(ssp585, ssp585_sig, [ssp585_com_len, ssp585_dis_len_com, ssp585_dis_len_year, ssp585_com_fre, ssp585_dis_fre])

# ================== Organize Data for Plotting ====================
Period = ["SSP585", "SSP245", "NAT", "ALL", "OBS"]
Ratio = ["CPL", "DPL", "DPL_Year", "CPF_Year", "DPF_Year"]

results = {
    "OBS": (obs, obs_sig),
    "ALL": (ALL, ALL_sig),
    "NAT": (NAT, NAT_sig),
    "SSP245": (ssp245, ssp245_sig),
    "SSP585": (ssp585, ssp585_sig),
}

records = []
for period, (slopes, sigs) in results.items():
    nrows, nratio = slopes.shape
    for i in range(nrows):
        for j in range(nratio):
            records.append({
                "Period": period,
                "Member": i + 1,
                "Ratio": Ratio[j],
                "Trend": slopes[i, j],
                "sig": sigs[i, j]
            })

df = pd.DataFrame(records)

# ================== Chart Plotting ====================
fig, ax = plt.subplots(figsize=(5, 8))

period_levels = list(df["Period"].unique())
ratio_levels = list(df["Ratio"].unique())

palette_all = sns.color_palette("tab20c", n_colors=20)
palette = [palette_all[i] for i in [4, 5, 7, 0, 2]]
color_map = dict(zip(ratio_levels, palette))
dodge_width = 0.7 / len(ratio_levels)
group_gap = 0.8 # Gap between vertical period groups

# Reference line at zero trend
ax.axvline(x=0, color='gray', linewidth=1, linestyle='--')

# --- Plot Individual Members (Scatters) ---
for i, row in df.iterrows():
    y_base = period_levels.index(row["Period"]) * group_gap
    hue_index = ratio_levels.index(row["Ratio"])
    y_val = y_base + (hue_index - (len(ratio_levels) - 1) / 2) * dodge_width
    x_val = row["Trend"]
    c = color_map[row["Ratio"]]

    if row["sig"] > 0.05:  # Not significant: Hollow circle
        ax.scatter(x_val, y_val, s=50, facecolors='none',
                    edgecolors=c, linewidth=2, alpha=0.75)
    else:  # Significant: Solid circle
        ax.scatter(x_val, y_val, s=70, facecolors=c,
                    edgecolors='none', alpha=0.75)

# --- Plot Distribution (Boxplots) ---
for i, period in enumerate(period_levels):
    for j, ratio in enumerate(ratio_levels):
        subdata = df[(df["Period"] == period) & (df["Ratio"] == ratio)]["Trend"]
        if subdata.empty: continue
        y_val = i * group_gap + (j - (len(ratio_levels) - 1) / 2) * dodge_width
        ax.boxplot(
            subdata,
            positions=[y_val],
            widths=0.03,
            patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black', linewidth=0.8),
            medianprops=dict(linestyle='-', color='black', linewidth=0.8),
            whiskerprops=dict(linestyle='--', color='black', linewidth=0.8),
            capprops=dict(linestyle='--', color='black', linewidth=0.8),
            flierprops=dict(marker='None'),
            vert=False
        )

# Format X-axis ticks (e.g., 0.0 -> 0)
def custom_formatter(x, pos):
    return "0" if x == 0 else f"{x:.2f}"
ax.xaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))

# --- Build Legends ---
sig_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='none', markersize=10, alpha=0.6, label='Significant'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='gray', markersize=9, linewidth=2, label='Not Significant')
]

ax.legend(handles=sig_handles, title='', loc='lower left',
          bbox_to_anchor=(-0.02, 0), frameon=False)

# Axis Configuration
ax.set_xlabel("Trend/Mean (%/year)")
ax.set_ylabel("")
ax.set_xlim(-0.02, 0.03)
ax.set_yticks([i * group_gap for i in range(len(period_levels))])
ax.set_yticklabels(period_levels)
ax.set_ylim(-0.4, 3.6)

# Aesthetics: Remove Y-axis spine and set tick width
ax.tick_params(axis="y", left=False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.tick_params(width=0.5)

# Horizontal separator lines between Periods
for y in [0.38, 1.19, 2.0, 2.81]:
    ax.axhline(y=y, color='gray', linestyle='--', linewidth=1)

# Variable labels for the CPL, DPL, etc. metrics
ax.text(0.0049, 1.29, r"$\mathrm{Dur}_\mathrm{comp}$", fontsize=10, color="k")
ax.text(0.005, 1.43, r"$\mathrm{Dur}_\mathrm{dist-comp}$", fontsize=10, color="k")
ax.text(0.005, 1.57, r"$\mathrm{Dur}_\mathrm{dist}$", fontsize=10, color="k")
ax.text(0.0049, 1.71, r"$\mathrm{Fre}_\mathrm{comp}$", fontsize=10, color="k")
ax.text(0.005, 1.85, r"$\mathrm{Fre}_\mathrm{dist}$", fontsize=10, color="k")

# Add panel label (e.g., 'a')
ax.text(-0.15, 1, 'a', transform=ax.transAxes, 
        fontsize=13, fontweight='bold', va='top', ha='left')
