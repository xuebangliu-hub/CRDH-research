import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Arial'

# Paths and Data Loading
main_path = '.../Detection_attribution/3days_SPEI128/'
# Detection and Attribution results (Scaling Factors)
# Rows typically represent lower bound, median, and upper bound (e.g., 5th, 50th, 95th percentiles)
beat_Rf = np.load(main_path + '/DA_percent.npy')
beat_Rd = np.load(main_path + '/DA_ratio_total.npy')
beat_Rde = np.load(main_path + '/DA_ratio.npy')

# --------------------------- Plotting Attribution Scaling Factors -----------------------------
categories = ["ANT", "NAT"] # Anthropogenic and Natural forcings
fig, axes = plt.subplots(3, 1, figsize=(2, 8))
fig.subplots_adjust(hspace=0.3) 

# Function to format Y-axis (converts 0.0 to 0)
def custom_formatter(y, pos):
    return "0" if y == 0 else f"{y:.1f}"

# Function to format Y-axis with 2 decimal places
def custom_formatter_2(y, pos):
    return "0" if y == 0 else f"{y:.2f}"

# Plotting scaling factors for three indicators
for idx, (data, label) in enumerate(zip([beat_Rf, beat_Rd, beat_Rde], ['b', 'e', 'h'])):
    ax = axes[idx]
    colors_fill = ['lightcoral', 'skyblue']
    colors_line = ['firebrick', 'royalblue']
    
    # Reference lines for attribution (0 = no signal, 1 = perfectly matched)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)

    x_pos = [0.25, 0.75]
    for i in range(2):
        # Draw confidence interval as filled box
        ax.fill_between([x_pos[i] - 0.07, x_pos[i] + 0.07], data[0, i], data[2, i], 
                        color=colors_fill[i], alpha=0.9, linewidth=0)
        # Draw median as a thick horizontal line
        ax.hlines(data[1, i], x_pos[i] - 0.07, x_pos[i] + 0.07, colors=colors_line[i], linewidth=4)
    
    ax.set_ylabel("Scaling factors")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.0)
    ax.set_yticks([-0.5, 0, 1.0, 2.0, 3.0])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter))
    
    # Visual styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)
    
    # Add panel label (b, e, h)
    ax.text(-0.35, 1.1, label, transform=ax.transAxes, 
            fontsize=13, fontweight='bold', va='top', ha='left')

# --------------------------- Calculation of Attributable Trends -----------------------------

def significant_mk(array1, significance_level=0.05):
    """Calculates the Sen's Slope using Mann-Kendall test."""
    mk_result = mk.original_test(array1, alpha=significance_level)
    slope = mk_result[7] 
    return slope

def compute_trend_all(obs, all_forc, nat_forc, scaling_factors):
    """
    Computes attributable trends for OBS, ANT, and NAT.
    ANT trend is derived from (ALL - NAT) multiplied by the scaling factor.
    """
    # Use specific periods for OBS
    period_obs = obs[[0, 1, 8, 9], :-11]
    trend_obs = np.full((4, 1), np.nan)
    for i in range(4):
        trend_obs[i] = significant_mk(period_obs[i, :])
    
    trend_all = np.full((3, 3), np.nan)
    
    # Column 0: OBS mean and 5th/95th percentiles
    trend_all[:, 0] = [np.percentile(trend_obs, 5), np.mean(trend_obs), np.percentile(trend_obs, 95)]
    
    # Column 1: Attributable ANT trend = scaling_factor * trend(ALL - NAT)
    diff_mean = np.mean(all_forc, axis=0) - np.mean(nat_forc, axis=0)
    trend_all[:, 1] = significant_mk(diff_mean) * scaling_factors[:3, 0]
    
    # Column 2: Attributable NAT trend = scaling_factor * trend(NAT)
    trend_all[:, 2] = significant_mk(np.mean(nat_forc, axis=0)) * scaling_factors[:3, 1]
    
    return trend_all

# Data Loading for Trend Computation
# (Loading Frequency, Duration, and Composite Duration datasets)
main_path1 = '.../Temporal_Change/3days_SPEI128/percent/'
trend_all_Rf = compute_trend_all(
    np.load(main_path1 + 'All_obs_compound_percent_temporal_change_1950_2023.npy')[:,9:],
    np.load(main_path1 + 'CMIP6_ALL_percent_temporal_change_1950_2014.npy')[:,9:],
    np.load(main_path1 + 'CMIP6_NAT_percent_temporal_change_1950_2014.npy')[:,9:],
    beat_Rf
)

main_path_dur = '.../Temporal_Change/3days_SPEI128/ratio_total/'
trend_all_Rd = compute_trend_all(
    np.load(main_path_dur + 'All_obs_ratio_total_temporal_change_1950_2023.npy')[:,9:],
    np.load(main_path_dur + 'CMIP6_ALL_ratio_total_temporal_change_1950_2014.npy')[:,9:],
    np.load(main_path_dur + 'CMIP6_NAT_ratio_total_temporal_change_1950_2014.npy')[:,9:],
    beat_Rd
)

main_path_comp = '.../Temporal_Change/3days_SPEI128/ratio/'
trend_all_Rde = compute_trend_all(
    np.load(main_path_comp + 'All_obs_ratio_temporal_change_1950_2023.npy')[:,9:],
    np.load(main_path_comp + 'CMIP6_ALL_ratio_temporal_change_1950_2014.npy')[:,9:],
    np.load(main_path_comp + 'CMIP6_NAT_ratio_temporal_change_1950_2014.npy')[:,9:],
    beat_Rde
)

# --------------------------- Plotting Attributable Trend Contributions -----------------------------

fig, axes = plt.subplots(3, 1, figsize=(2, 8))
fig.subplots_adjust(hspace=0.3)
groups = ['OBS', 'ANT', 'NAT']
bar_colors = ['darkgray', 'lightcoral', 'skyblue']

for idx, (trend_data, label, ylim, fmt_func) in enumerate(zip(
    [trend_all_Rf, trend_all_Rd, trend_all_Rde], 
    ['c', 'f', 'i'], 
    [(-0.1, 0.2), (-0.05, 0.1), (-0.08, 0.16)],
    [custom_formatter, custom_formatter_2, custom_formatter_2])):
    
    ax = axes[idx]
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    means = trend_data[1, :]
    yerr_low = trend_data[0, :]
    yerr_high = trend_data[2, :]
    
    x = np.array([0, 1, 2], dtype=float)
    # Adjust alignment for visualization
    x[0] += 0.4
    x[2] -= 0.4
    
    ax.bar(x, means, width=0.3, color=bar_colors, alpha=0.9, linewidth=0) 
    for i in range(3):
        ax.plot([x[i], x[i]], [yerr_low[i], yerr_high[i]], color='black', lw=1)

    ax.set_xlim(0, 2)
    ax.set_xticks(x)
    ax.set_ylim(ylim)
    ax.set_xticklabels(groups)
    ax.set_ylabel('Attributable trends')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_func))
    
    ax.text(-0.4, 1.1, label, transform=ax.transAxes, 
            fontsize=13, fontweight='bold', va='top', ha='left')