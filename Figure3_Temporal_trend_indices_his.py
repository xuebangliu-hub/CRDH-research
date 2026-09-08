import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk
import matplotlib.ticker as mticker

# -------------------- Font Settings ---------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'         # Roman style for math mode
plt.rcParams['mathtext.it'] = 'Arial:italic'  # Italic style for math mode
plt.rcParams['mathtext.bf'] = 'Arial:bold'    # Bold style for math mode

def row_anomalies_three(arr1, arr2, arr3):
    """
    Calculate row-wise anomalies for three 2D arrays (subtracting the row mean).
    The arrays can have different shapes.

    Parameters:
        arr1, arr2, arr3: numpy.ndarray, 2D arrays.

    Returns:
        anomalies1, anomalies2, anomalies3: Corresponding anomaly arrays.
    """
    results = []
    for arr in (arr1, arr2, arr3):
        arr = np.asarray(arr, dtype=float)
        row_means = np.nanmean(arr, axis=1, keepdims=True)
        results.append(arr - row_means)
    
    return results[0], results[1], results[2]

def percentile_band_with_mean(data, lower=5, upper=95):
    """
    Calculates the percentile range and mean for each column of a 2D array.
    Used for plotting shaded uncertainty bands.
    
    Parameters
    ----
    data : np.ndarray
        shape = (n_sources, n_years). Rows are data sources, columns are years.
    lower : float
        Lower percentile (default 5).
    upper : float
        Upper percentile (default 95).

    Returns
    ----
    low_vals : np.ndarray
        Lower percentile for each year.
    mean_vals : np.ndarray
        Mean value for each year.
    high_vals : np.ndarray
        Upper percentile for each year.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array (n_sources, n_years)")

    low_vals = np.nanpercentile(data, lower, axis=0)
    mean_vals = np.nanmean(data, axis=0)
    high_vals = np.nanpercentile(data, upper, axis=0)
    return low_vals, mean_vals, high_vals

# ------------------- Load Data -------------------------------
# Frequency (Fre) data
main_path_fre = '.../Temporal_Change/3days_SPEI128/percent/'
CDHW_Rf_obs = np.load(main_path_fre + 'All_obs_compound_percent_temporal_change_1950_2023.npy')[:,9:]
CDHW_Rf_all = np.load(main_path_fre + 'CMIP6_ALL_percent_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rf_nat = np.load(main_path_fre + 'CMIP6_NAT_percent_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rf_obs, CDHW_Rf_all, CDHW_Rf_nat = row_anomalies_three(CDHW_Rf_obs, CDHW_Rf_all, CDHW_Rf_nat)

# Duration (Dur) data
main_path_dur = '.../3days_SPEI128/ratio_total/'
CDHW_Rd_obs = np.load(main_path_dur + 'All_obs_ratio_total_temporal_change_1950_2023.npy')[:,9:]
CDHW_Rd_all = np.load(main_path_dur + 'CMIP6_ALL_ratio_total_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rd_nat = np.load(main_path_dur + 'CMIP6_NAT_ratio_total_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rd_obs, CDHW_Rd_all, CDHW_Rd_nat = row_anomalies_three(CDHW_Rd_obs, CDHW_Rd_all, CDHW_Rd_nat)

# Composite Duration (Dur-comp) data
main_path_comp = '.../Temporal_Change/3days_SPEI128/ratio/'
CDHW_Rde_obs = np.load(main_path_comp + 'All_obs_ratio_temporal_change_1950_2023.npy')[:,9:]
CDHW_Rde_all = np.load(main_path_comp + 'CMIP6_ALL_ratio_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rde_nat = np.load(main_path_comp + 'CMIP6_NAT_ratio_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rde_obs, CDHW_Rde_all, CDHW_Rde_nat = row_anomalies_three(CDHW_Rde_obs, CDHW_Rde_all, CDHW_Rde_nat)

# ------------------- Trend Analysis -------------------------------
def significant_mk(array1, significance_level=0.05):
    """ Calculate Mann-Kendall trend slope and p-value. """
    mk_result = mk.original_test(array1, alpha=significance_level)
    slope = mk_result[7]
    p_value = mk_result[2]
    return slope, p_value

# ------------------- Visualization -------------------------------
years1 = np.arange(1960, 2023)
years2 = np.arange(1960, 2014)

fig, axes = plt.subplots(3, 1, figsize=(4, 8))
fig.subplots_adjust(hspace=0.3) # Increase vertical spacing

# --- Common Plotting Logic ---
def setup_axis(ax, years_obs, obs_data, years_model, all_data, nat_data, ylabel, ylim, yticks, label_letter):
    low_obs, mean_obs, high_obs = percentile_band_with_mean(obs_data)
    low_all, mean_all, high_all = percentile_band_with_mean(all_data)
    low_nat, mean_nat, high_nat = percentile_band_with_mean(nat_data)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Plot Observation
    ax.fill_between(years_obs, low_obs, high_obs, color='gray', alpha=0.5)
    ax.plot(years_obs, mean_obs, color='black', label='OBS')
    
    # Plot CMIP6 ALL (Anthropogenic + Natural)
    ax.fill_between(years_model, low_all, high_all, color='lightcoral', alpha=0.5)
    ax.plot(years_model, mean_all, color='firebrick', label='ALL')
    
    # Plot CMIP6 NAT (Natural only)
    ax.fill_between(years_model, low_nat, high_nat, color='skyblue', alpha=0.5)
    ax.plot(years_model, mean_nat, color='royalblue', label='NAT')
    
    ax.set_ylabel(ylabel)
    ax.set_xlim(1960, 2022)
    ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)

    # Legend formatting
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, loc='lower center', 
              bbox_to_anchor=(0.5, 0), ncol=len(handles), borderaxespad=0.3)

    # Style Adjustments
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)

    # Subplot Index label
    ax.text(-0.2, 1.1, label_letter, transform=ax.transAxes, 
            fontsize=13, fontweight='bold', va='top', ha='left')

# --- Panel A: Frequency ---
setup_axis(axes[0], years1, CDHW_Rf_obs, years2, CDHW_Rf_all, CDHW_Rf_nat, 
           r"$\mathrm{Fre}_\mathrm{CRDH}$ anomaly (%)", (-10, 10), None, 'a')

# --- Panel B: Duration ---
def custom_formatter_decimal(y, pos):
    return "0" if y == 0 else f"{y:.1f}"

setup_axis(axes[1], years1, CDHW_Rd_obs, years2, CDHW_Rd_all, CDHW_Rd_nat, 
           r"$\mathrm{Dur}_\mathrm{CRDH}$ anomaly (%)", (-5, 5), [-5.0, -2.5, 0, 2.5, 5.0], 'd')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_decimal))

# --- Panel C: Composite Duration ---
def custom_formatter_int(y, pos):
    return "0" if y == 0 else f"{y:.0f}"

setup_axis(axes[2], years1, CDHW_Rde_obs, years2, CDHW_Rde_all, CDHW_Rde_nat, 
           r"$\mathrm{Dur}_\mathrm{CRDH-comp}$ anomaly (%)", (-8, 8), [-8, -4, 0, 4, 8], 'g')
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_int))
