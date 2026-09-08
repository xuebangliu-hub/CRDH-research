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
        Lower percentile for each time step.
    mean_vals : np.ndarray
        Mean value for each time step.
    high_vals : np.ndarray
        Upper percentile for each time step.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array (n_sources, n_years)")

    low_vals = np.nanpercentile(data, lower, axis=0)
    mean_vals = np.nanmean(data, axis=0)
    high_vals = np.nanpercentile(data, upper, axis=0)
    return low_vals, mean_vals, high_vals

# -------------------- Load Data --------------------
main_path_fre = '.../3days_SPEI128/percent/'
CDHW_Rf_obs = np.load(main_path_fre + 'All_obs_compound_percent_temporal_change_1950_2023.npy')[:,9:]
CDHW_Rf_all = np.load(main_path_fre + 'CMIP6_ALL_percent_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rf_ssp245 = np.load(main_path_fre + 'CMIP6_ssp245_percent_temporal_change_2015_2100.npy')
CDHW_Rf_ssp585 = np.load(main_path_fre + 'CMIP6_ssp585_percent_temporal_change_2015_2100.npy')

main_path_dur = '.../3days_SPEI128/ratio_total/'
CDHW_Rd_obs = np.load(main_path_dur + 'All_obs_ratio_total_temporal_change_1950_2023.npy')[:,9:]
CDHW_Rd_all = np.load(main_path_dur + 'CMIP6_ALL_ratio_total_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rd_ssp245 = np.load(main_path_dur + 'CMIP6_ssp245_compound_ratio_total_temporal_change_2015_2100.npy')
CDHW_Rd_ssp585 = np.load(main_path_dur + 'CMIP6_ssp585_compound_ratio_total_temporal_change_2015_2100.npy')

main_path_comp = '.../3days_SPEI128/ratio/'
CDHW_Rde_obs = np.load(main_path_comp + 'All_obs_ratio_temporal_change_1950_2023.npy')[:,9:]
CDHW_Rde_all = np.load(main_path_comp + 'CMIP6_ALL_ratio_temporal_change_1950_2014.npy')[:,9:]
CDHW_Rde_ssp245 = np.load(main_path_comp + 'CMIP6_ssp245_ratio_temporal_change_2015_2100.npy')
CDHW_Rde_ssp585 = np.load(main_path_comp + 'CMIP6_ssp585_ratio_temporal_change_2015_2100.npy')

# ------------------- Plotting Future Time Series -------------------------------
years3 = np.arange(2016, 2100)
fig, axes = plt.subplots(3, 1, figsize=(3, 9))
fig.subplots_adjust(hspace=0.3)

panel_labels = ['c', 'f', 'i']
y_labels = [r"$\mathrm{Fre}_\mathrm{CRDH}$ (%)", 
            r"$\mathrm{Dur}_\mathrm{CRDH}$ (%)", 
            r"$\mathrm{Dur}_\mathrm{CRDH-comp}$ (%)"]
data_pairs = [
    (CDHW_Rf_ssp245, CDHW_Rf_ssp585),
    (CDHW_Rd_ssp245, CDHW_Rd_ssp585),
    (CDHW_Rde_ssp245, CDHW_Rde_ssp585)
]

for i, ax in enumerate(axes):
    ssp245_data, ssp585_data = data_pairs[i]
    low_245, mean_245, high_245 = percentile_band_with_mean(ssp245_data)
    low_585, mean_585, high_585 = percentile_band_with_mean(ssp585_data)
    
    ax.axvline(x=2100, color='gray', linestyle='--', linewidth=1)
    
    # Uncertainty bands
    ax.fill_between(years3, low_585, high_585, color='indianred', alpha=0.5, lw=0)
    ax.fill_between(years3, low_245, high_245, color='royalblue', alpha=0.5, lw=0)
    
    # Ensemble means
    ax.plot(years3, mean_245, color='navy', label='SSP245')
    ax.plot(years3, mean_585, color='maroon', label='SSP585')
    
    ax.set_ylabel(y_labels[i], fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0, 50)
    ax.set_xlim(2015, 2120)
    ax.set_xticks(np.arange(2015, 2101, 25))
    
    if i == 0:
        ax.legend(frameon=False, loc='upper left', fontsize=12)
        
    # Spine and tick styling
    ax.spines['top'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, left=True, labelleft=True, right=True, labelright=False)
    
    # Subplot index
    ax.text(-0.22, 1.1, panel_labels[i], transform=ax.transAxes, 
            fontsize=13, fontweight='bold', va='top', ha='left')

# ------------------- Plotting Century-End (2090s) Change -------------------------------
fig_end, axes_end = plt.subplots(3, 1, figsize=(0.5, 9))
fig_end.subplots_adjust(hspace=0.3)

for i, ax in enumerate(axes_end):
    ssp245_data, ssp585_data = data_pairs[i]
    
    # Get values for the last available year (end of century)
    low_245, mean_245, high_245 = np.array(percentile_band_with_mean(ssp245_data))[:, -1]
    low_585, mean_585, high_585 = np.array(percentile_band_with_mean(ssp585_data))[:, -1]
    
    x_pos = [0.15, 0.30]
    # SSP245 Bar/Line
    ax.fill_between([x_pos[0]-0.07, x_pos[0]+0.07], low_245, high_245, color='royalblue', alpha=0.9, lw=0)
    ax.hlines(mean_245, x_pos[0]-0.07, x_pos[0]+0.07, colors='navy', linewidth=4)
    
    # SSP585 Bar/Line
    ax.fill_between([x_pos[1]-0.07, x_pos[1]+0.07], low_585, high_585, color='indianred', alpha=0.9, lw=0)
    ax.hlines(mean_585, x_pos[1]-0.07, x_pos[1]+0.07, colors='maroon', linewidth=4)
    
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 50)
    
    # Hide all visual elements to create a floating "end-point" comparison
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('white') # Make bottom spine invisible against white bg
    ax.tick_params(axis='x', colors='white')
    
    # Add hidden labels to maintain alignment with the main plot
    ax.text(-0.22, 1.1, panel_labels[i], transform=ax.transAxes, 
            fontsize=13, fontweight='bold', va='top', ha='left', color='white')