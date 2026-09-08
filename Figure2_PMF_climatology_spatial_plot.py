import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

# -------------------- Font settings --------------------
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'   # upright font in math mode
plt.rcParams['mathtext.it'] = 'Arial:italic'  # italic requests also use Arial
plt.rcParams['mathtext.bf'] = 'Arial:bold'    # bold requests also use Arial
# plt.rcParams['mathtext.fontset'] = 'stix'

# -------------------- Read data --------------------
main_path1 = '.../Compound/180days/results/SI/bootstrap_PMF/'

# cli_obs_all = np.load(main_path1 + 'All_obs_PMF_spatial_distribution_1950_2023.npy')[:,:,9:-9]
# cli_obs1 = np.nanmean(cli_obs_all,axis=2)
# cli_obs = np.load(main_path1 + 'All_obs_PMF_mean_spatial_distribution_1960_2022.npy')
# cli_ALL = np.load(main_path1 + 'CMIP6_ALL_PMF_mean_spatial_distribution_1960_2013.npy')
cli_obs1 = np.load(main_path1 + 'All_obs_mean/P1_1960_1990_pmf_boot.npy')
cli_obs2 = np.load(main_path1 + 'All_obs_mean/P2_1991_2022_pmf_boot.npy')
# cli_obs = np.percentile(cli_obs_all, 70, axis=2)
cli_ALL1 = np.load(main_path1 + 'CMIP6_ALL_mean/P1_1960_1986_pmf_boot.npy')
cli_ALL2 = np.load(main_path1 + 'CMIP6_ALL_mean/P2_1987_2013_pmf_boot.npy')#  [:,:,:31]
# cli_ALL = np.percentile(cli_ALL_all, 40, axis=2)


# -------------------- Plot settings --------------------
lon = np.arange(-178.75, 180, 2.5)
lat = np.arange(88.75, -90, -2.5)

# -------------------- Level & norm settings --------------------
# Color palette
# cmaps = [plt.get_cmap('OrRd')]*4
# norms = [
#     mcolors.BoundaryNorm(np.arange(1, 5.1, 0.5), 256, extend='both'),
#     mcolors.BoundaryNorm(np.arange(1, 5.1, 0.5), 256, extend='both'),
#     mcolors.BoundaryNorm(np.arange(1, 5.1, 0.5), 256, extend='both'),
#     mcolors.BoundaryNorm(np.arange(1, 5.1, 0.5), 256, extend='both')
# ]

# ----------- Construct custom colormap (first color set to white) -----------
bounds1 = np.arange(1, 2.1, 0.2)
bounds2 = np.arange(1, 3.1, 0.4)
n_intervals = len(bounds1) - 1  # = 8

# total bins = intervals + 2 (because extend='both')
n_colors = n_intervals + 2  # = 10

base_cmap = plt.get_cmap('OrRd')

# Generate 10 colors
colors = base_cmap(np.linspace(0, 1, n_colors))

# Set the first color to white
colors[0] = [1, 1, 1, 1]

custom_cmap = mcolors.ListedColormap(colors)
cmaps = [custom_cmap] * 4

# norm: note the use of N=n_colors
norms = [
    mcolors.BoundaryNorm(bounds1, n_colors, extend='both'),
    mcolors.BoundaryNorm(bounds2, n_colors, extend='both'),
    mcolors.BoundaryNorm(bounds1, n_colors, extend='both'),
    mcolors.BoundaryNorm(bounds2, n_colors, extend='both')
]

levels_ticks = [
    np.arange(1, 2.1, 0.2),
    np.arange(1, 3.1, 0.4),
    np.arange(1, 2.1, 0.2),
    np.arange(1, 3.1, 0.4)
]

data_list = [cli_obs1, cli_ALL1, cli_obs2, cli_ALL2]

cbar_labels = [
    r'Spatial distribution of mean $\mathrm{PMF}_\mathrm{OBS}$',
    r'Spatial distribution of mean $\mathrm{PMF}_\mathrm{ALL}$',
    r'Spatial distribution of mean $\mathrm{PMF}_\mathrm{OBS}$',
    r'Spatial distribution of mean $\mathrm{PMF}_\mathrm{ALL}$',
]

subplot_labels = ['a', 'b', 'c', 'd']  # subplot numbering
# subplot_labels = ['A', 'B', 'C', 'D']  # subplot numbering
labels_threshold = ["1960-1990", "1960-1986", "1991-2023", "1987-2013"]

# -------------------- Custom colorbar formatter --------------------
def zero_only_formatter(x, pos):
    """Display 0 as '0', others keep original formatting"""
    if x == 0:
        return '0'
    else:
        return f'{x:g}'

formatter = FuncFormatter(zero_only_formatter)

proj = ccrs.Robinson()
fig, axs = plt.subplots(2, 2, figsize=(8.2, 6), subplot_kw={"projection": proj}, constrained_layout=True)
axs = axs.flatten()

fig.set_constrained_layout_pads(
    w_pad=0.12,
    h_pad=0.05,
    # wspace=0.05,
    # hspace=0.05
)
# fig.subplots_adjust(wspace=0.01, hspace=0.35)

# -------------------- Plotting loop --------------------
for i, ax in enumerate(axs):
    pcm = ax.pcolormesh(lon, lat, data_list[i], cmap=cmaps[i], norm=norms[i],
                        transform=ccrs.PlateCarree(), shading='auto')
    cbar = fig.colorbar(pcm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.08, extend='both', ticks=levels_ticks[i]) # , ticks=levels_ticks[i]
    cbar.ax.xaxis.set_major_formatter(formatter)  # only change the display of 0
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar_labels[i], fontsize=14)
    
    # Set border width
    ax.spines['geo'].set_linewidth(0.5)  # projection border linewidth
    # Set tick linewidth
    ax.tick_params(width=0.5, length=3)
    
    ax.coastlines(linewidth=0.5)
    # ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax.text(0.02, 1, subplot_labels[i],
                    transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight='bold')
    
    # ✅ Add label at the middle-left position of each subplot
    ax.text(
        0.02, 0.5, labels_threshold[i], transform=ax.transAxes,
        fontsize=14, va='center', ha='left'
    )

# -------------------- Save figure --------------------
save_path = ".../results/Figure/3days_SPEI128/main_figure/NC/"
plt.savefig(save_path + 'Spatial_bootstrap_PMF_Obs_All_1960_2022.jpg', dpi=500, bbox_inches='tight')