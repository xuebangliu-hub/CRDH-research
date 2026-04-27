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


# -------------------- Data Loading ---------------------
main_path = '.../Spatial_Trend/3days_SPEI128/'

# Loading SSP2-4.5 Scenarios
ssp245_CDHW_Rde = np.load(main_path + 'ratio/ssp245/ssp245_ratio_IPCC_MK_all_2015_2100.npy')
ssp245_CDHW_Rf  = np.load(main_path + 'percent/ssp245/ssp245_percent_IPCC_MK_all_2015_2100.npy')
ssp245_CDHW_Rd  = np.load(main_path + 'ratio_total/ssp245/ssp245_ratio_total_IPCC_MK_all_2015_2100.npy')

# Loading SSP5-8.5 Scenarios
ssp585_CDHW_Rde = np.load(main_path + 'ratio/ssp585/ssp585_ratio_IPCC_MK_all_2015_2100.npy')
ssp585_CDHW_Rf  = np.load(main_path + 'percent/ssp585/ssp585_percent_IPCC_MK_all_2015_2100.npy')
ssp585_CDHW_Rd  = np.load(main_path + 'ratio_total/ssp585/ssp585_ratio_total_IPCC_MK_all_2015_2100.npy')

lon = np.arange(-178.75, 180, 2.5)
lat = np.arange(88.75, -90, -2.5)

# Shapefile paths
IPCC_land_shp = '.../IPCC_Region/'
Sig_paths = [
    main_path + '/percent/ssp245/Sig_shp_2015_2100/0.shp',
    main_path + '/percent/ssp585/Sig_shp_2015_2100/0.shp',
    main_path + '/ratio_total/ssp245/Sig_shp_2015_2100/0.shp',
    main_path + '/ratio_total/ssp585/Sig_shp_2015_2100/0.shp',
    main_path + '/ratio/ssp245/Sig_shp_2015_2100/0.shp',
    main_path + '/ratio/ssp585/Sig_shp_2015_2100/0.shp'
]

data_list = [
    ssp245_CDHW_Rf, ssp585_CDHW_Rf,
    ssp245_CDHW_Rd, ssp585_CDHW_Rd,
    ssp245_CDHW_Rde, ssp585_CDHW_Rde
]

colorbar_labels = [
    r'Future trend in $\mathrm{Fre}_\mathrm{CRDH}$ (%/year)',
    r'Future trend in $\mathrm{Fre}_\mathrm{CRDH}$ (%/year)',
    r'Future trend in $\mathrm{Dur}_\mathrm{CRDH}$ (%/year)',
    r'Future trend in $\mathrm{Dur}_\mathrm{CRDH}$ (%/year)',
    r'Future trend in $\mathrm{Dur}_\mathrm{CRDH-comp}$ (%/year)',
    r'Future trend in $\mathrm{Dur}_\mathrm{CRDH-comp}$ (%/year)'
]

# Classification & Norm settings
norms = [mcolors.BoundaryNorm(np.arange(-0.6, 0.61, 0.1), 256, extend='both')] * 6

colorbar_ticks = [np.arange(-0.6, 0.61, 0.3)] * 6

# -------------------- Custom Colorbar Formatter ---------------------
def zero_only_formatter(x, pos):
    """ Displays 0 as '0', while keeping original format for others """
    if abs(x) < 1e-8:  # Account for floating point errors
        return '0'
    else:
        return f'{x:g}'

formatter = FuncFormatter(zero_only_formatter)

labels = ['a','b','d','e', 'g', 'h']  # Subplot index labels

# -------------------- Plotting ---------------------
proj = ccrs.Robinson()
fig, axs = plt.subplots(3, 2, figsize=(8, 9), subplot_kw={"projection": proj}, constrained_layout=True)
axs = axs.flatten()

for i, ax in enumerate(axs):
    # Plot spatial data
    mesh = ax.pcolormesh(lon, lat, data_list[i], 
                         cmap='RdBu_r', norm=norms[i], transform=ccrs.PlateCarree(), shading='auto')
    
    # Add horizontal colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.08, ticks=colorbar_ticks[i], extend='both')
    cbar.ax.xaxis.set_major_formatter(formatter)  # Only modify '0' display
    
    # Add IPCC Land Region boundaries (dashed gray)
    ax.add_geometries(Reader(IPCC_land_shp + 'IPCC_Land_Region.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='gray', linewidth=0.5, linestyle='--')
    
    # Add coastlines
    ax.coastlines(linewidth=0.5)
    
    # Add Significance boundaries (solid black)
    ax.add_geometries(Reader(Sig_paths[i]).geometries(),
                      ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=1)
    
    # Set border line width
    ax.spines['geo'].set_linewidth(0.5) 
    
    # Set tick parameters
    ax.tick_params(width=0.5, length=3)
    
    # Colorbar label settings
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(colorbar_labels[i], fontsize=14)

    # Add subplot label (Bold)
    ax.text(0, 1, labels[i], transform=ax.transAxes, 
            fontsize=15, fontweight='bold', va='top', ha='left')
