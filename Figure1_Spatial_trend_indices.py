import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# -------------------- Font Settings ---------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'         # Roman font for math mode
plt.rcParams['mathtext.it'] = 'Arial:italic'  # Italic font for math mode
plt.rcParams['mathtext.bf'] = 'Arial:bold'    # Bold font for math mode


# -------------------- 1960-2023 Data Loading ---------------------
main_path2 = '.../Spatial_Trend/3days_SPEI128/'

data_files = {
    'trend': {
        'Rde': main_path2 + 'ratio/obs/Obs_ratio_IPCC_MK_all_1960_2023.npy',
        'Rf': main_path2 + 'percent/obs/Obs_percent_IPCC_MK_all_1960_2023.npy',
        'Rd': main_path2 + 'ratio_total/obs/Obs_ratio_total_IPCC_MK_all_1960_2023.npy'
    }
}

trend_CDHW_Rf = np.load(data_files['trend']['Rf'])
trend_CDHW_Rd = np.load(data_files['trend']['Rd'])
trend_CDHW_Rde = np.load(data_files['trend']['Rde'])

# -------------------- Plotting Coordinates ---------------------
lon = np.arange(-178.75, 180, 2.5)
lat = np.arange(88.75, -90, -2.5)


# -------------------- Classification & Norm Settings ---------------------
norms = [
    mcolors.BoundaryNorm(np.arange(-0.24, 0.241, 0.06), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.08, 0.081, 0.02), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.16, 0.161, 0.04), 256, extend='both')
]

levels_ticks = [
    np.arange(-0.24, 0.241, 0.12),
    np.arange(-0.08, 0.081, 0.04),
    np.arange(-0.16, 0.161, 0.08)
]

# Color palette (Reversed Red-Blue)
cmaps = [plt.get_cmap('RdBu').reversed()] * 3
data_list = [trend_CDHW_Rf, trend_CDHW_Rd, trend_CDHW_Rde]

cbar_labels = [
    r'Historical trend in $\mathrm{Fre}_\mathrm{CRDH}$ (%/year)',
    r'Historical trend in $\mathrm{Dur}_\mathrm{CRDH}$ (%/year)',
    r'Historical trend in $\mathrm{Dur}_\mathrm{CRDH-comp}$ (%/year)'
]

labels = ['b', 'c', 'd']  # Subplot index labels
labels_threshold = ["1960-2023", "1960-2023", "1960-2023"]

# -------------------- Custom Colorbar Formatter ---------------------
def zero_only_formatter(x, pos):
    """ Displays 0 as '0', others keep original format """
    if x == 0:
        return '0'
    else:
        return f'{x:g}'

formatter = FuncFormatter(zero_only_formatter)

proj = ccrs.Robinson()
fig, axs = plt.subplots(1, 3, figsize=(12, 3), subplot_kw={"projection": proj}, constrained_layout=True)
axs = axs.flatten()

# -------------------- Plotting Loop (1960-2023) ---------------------
for i, ax in enumerate(axs):
    pcm = ax.pcolormesh(lon, lat, data_list[i], cmap=cmaps[i], norm=norms[i],
                        transform=ccrs.PlateCarree(), shading='auto')
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.08, extend='both', ticks=levels_ticks[i])
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_labels[i], fontsize=14)
    
    # Set border width
    ax.spines['geo'].set_linewidth(0.5)
    # Set tick parameters
    ax.tick_params(width=0.5, length=3)
    
    ax.coastlines(linewidth=0.5)

    # Add subplot index (Bold)
    ax.text(0, 1, labels[i], transform=ax.transAxes, 
            fontsize=15, fontweight='bold', va='top', ha='left')

    # Add period label on the left side of each subplot
    ax.text(
        0.01, 0.5, labels_threshold[i], transform=ax.transAxes,
        fontsize=14, va='center', ha='left'
    )

# -------------------- Significance & IPCC Regions ---------------------
IPCC_land_shp = '.../IPCC_Region/'
sig_shps = [
    main_path2 + 'percent/obs/Sig_shp_1960_2023/0.shp',
    main_path2 + 'ratio_total/obs/Sig_shp_1960_2023/0.shp',
    main_path2 + 'ratio/obs/Sig_shp_1960_2023/0.shp'
]

for j, ax in enumerate(axs[0:3]):
    ax.add_geometries(Reader(IPCC_land_shp + 'IPCC_Land_Region.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='gray', linewidth=0.5, linestyle='--')
    ax.add_geometries(Reader(sig_shps[j]).geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1)
    

# -------------------- 1960-1990 and 1991-2023 Data Loading ---------------------
main_path1 = '.../Spatial_Trend/3days_SPEI128/'

data_files = {
    'before': {
        'Rde': main_path1 + 'ratio/obs/Obs_ratio_IPCC_MK_all_1960_1990.npy',
        'Rf': main_path1 + 'percent/obs/Obs_percent_IPCC_MK_all_1960_1990.npy',
        'Rd': main_path1 + 'ratio_total/obs/Obs_ratio_total_IPCC_MK_all_1960_1990.npy'
    },
    'last': {
        'Rde': main_path1 + 'ratio/obs/Obs_ratio_IPCC_MK_all_1991_2023.npy',
        'Rf': main_path1 + 'percent/obs/Obs_percent_IPCC_MK_all_1991_2023.npy',
        'Rd': main_path1 + 'ratio_total/obs/Obs_ratio_total_IPCC_MK_all_1991_2023.npy'
    }
}

before_CDHW_Rf = np.load(data_files['before']['Rf'])
before_CDHW_Rd = np.load(data_files['before']['Rd'])
before_CDHW_Rde = np.load(data_files['before']['Rde'])
last_CDHW_Rf = np.load(data_files['last']['Rf'])
last_CDHW_Rd = np.load(data_files['last']['Rd'])
last_CDHW_Rde = np.load(data_files['last']['Rde'])

# -------------------- Norms & Ticks for Comparative Plotting ---------------------
norms = [
    mcolors.BoundaryNorm(np.arange(-0.24, 0.241, 0.06), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.08, 0.081, 0.02), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.16, 0.161, 0.04), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.48, 0.49, 0.12), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.20, 0.21, 0.05), 256, extend='both'),
    mcolors.BoundaryNorm(np.arange(-0.40, 0.41, 0.1), 256, extend='both')
]

levels_ticks = [
    np.arange(-0.24, 0.241, 0.12),
    np.arange(-0.08, 0.081, 0.04),
    np.arange(-0.16, 0.161, 0.08),
    np.arange(-0.48, 0.49, 0.24),
    np.arange(-0.20, 0.21, 0.1),
    np.arange(-0.40, 0.41, 0.2)
]

data_list = [before_CDHW_Rf, before_CDHW_Rd, before_CDHW_Rde, last_CDHW_Rf, last_CDHW_Rd, last_CDHW_Rde]

cbar_labels = [
    r'Historical trend in $\mathrm{Fre}_\mathrm{CRDH}$ (%/year)',
    r'Historical trend in $\mathrm{Dur}_\mathrm{CRDH}$ (%/year)',
    r'Historical trend in $\mathrm{Dur}_\mathrm{CRDH-comp}$ (%/year)',
    r'Historical trend in $\mathrm{Fre}_\mathrm{CRDH}$ (%/year)',
    r'Historical trend in $\mathrm{Dur}_\mathrm{CRDH}$ (%/year)',
    r'Historical trend in $\mathrm{Dur}_\mathrm{CRDH-comp}$ (%/year)'
]

labels_threshold = ["1960-1990", "1960-1990", "1960-1990", 
                    "1991-2023", "1991-2023", "1991-2023"]

labels = ['e', 'f', 'g', 'h', 'i', 'j']

# -------------------- Plotting Loop (Comparative Periods) ---------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 6), subplot_kw={"projection": proj}, constrained_layout=True)
axs = axs.flatten()

for i, ax in enumerate(axs):
    pcm = ax.pcolormesh(lon, lat, data_list[i], cmap=plt.get_cmap('RdBu').reversed(), norm=norms[i],
                        transform=ccrs.PlateCarree(), shading='auto')
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.08, extend='both', ticks=levels_ticks[i])
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_labels[i], fontsize=14)
    
    ax.spines['geo'].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=3)
    ax.coastlines(linewidth=0.5)
    
    # Subplot index
    ax.text(0, 1, labels[i], transform=ax.transAxes, 
            fontsize=15, fontweight='bold', va='top', ha='left')

    # Period Label
    ax.text(
        0.01, 0.5, labels_threshold[i], transform=ax.transAxes,
        fontsize=14, va='center', ha='left'
    )

# -------------------- Significance & IPCC Regions (Comparative) ---------------------
sig_shps = [
    main_path1 + 'percent/obs/Sig_shp_1960_1990/0.shp',
    main_path1 + 'ratio_total/obs/Sig_shp_1960_1990/0.shp',
    main_path1 + 'ratio/obs/Sig_shp_1960_1990/0.shp',
    main_path1 + 'percent/obs/Sig_shp_1991_2023/0.shp',
    main_path1 + 'ratio_total/obs/Sig_shp_1991_2023/0.shp',
    main_path1 + 'ratio/obs/Sig_shp_1991_2023/0.shp'
]

for j, ax in enumerate(axs[0:6]):
    ax.add_geometries(Reader(IPCC_land_shp + 'IPCC_Land_Region.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='gray', linewidth=0.5, linestyle='--')
    ax.add_geometries(Reader(sig_shps[j]).geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1)

