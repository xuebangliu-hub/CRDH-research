import numpy as np
import tifffile as tf
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd 
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Arial'

# Model configuration: ERA5 for Max Temp and PET, 20CRV3 for Precipitation
model = '20CRV3_ERA5' 
year = 2010
com_path = '.../Compound_event/' 
com = np.load(com_path + model + f'/Compound_Length_{year}.npy')

# Calculate compound event frequency
com_fre = np.sum(~np.isnan(com), axis=2)
com_fre = com_fre.astype(float)
com_fre[com_fre == 0] = np.nan

# Input paths for Drought and Heatwave events
dry_path_in = '.../Observed/' + model  + '/3days_events/'
heat_path_in = '.../ERA5/3days_events/'

# Load Temperature and SPEI datasets
tmx_all = np.load('.../ERA5/tasmax/ERA5_tasmax_1940_2024.npy')
spei_path = '.../SPEI/Last_SPEI/Reanalysis/' + 'SPEI_180days_' + model + '_*.npy'
spei_data = glob.glob(spei_path)
spei_all = np.load(spei_data[0]) 

# Coordinate indices
i, j = 11, 91
lon = np.arange(-178.75, 180, 2.5)
lat = np.arange(88.75, -90, -2.5)

# ========= Load Hottest Day Data ===========
hottest_d = np.load('.../hottest day/hottest_day.npy')
temp_value = -9999
hottest_d_temp = np.nan_to_num(hottest_d, nan=temp_value)
# Convert to integer
hottest_day = hottest_d_temp.astype(int)[i, j]

# ================ Load Max Temp and SPEI Data =============
com_count = com_fre[i, j]

tmx_cli_path = '.../Heatwaves/ERA5/'
tmx_cli = np.load(tmx_cli_path + '/Climatology/Heat_climatology_ERA5_1982_2014.npy')[i, j, :]

# Build complete date sequence including Feb 29 for leap years
full_dates_with_leap = pd.date_range(f'{1940}-01-01', f'{year-1}-12-31', freq='D')
tmx_year = tmx_all[i, j, len(full_dates_with_leap):len(full_dates_with_leap)+365]
# Slice window around hottest day (±135 days)
tmx = tmx_year[hottest_day-135:hottest_day+136]

# SPEI date sequence logic
full_dates_with_leap_spei = pd.date_range(f'{1950}-01-01', f'{year-1}-12-31', freq='D')
spei_year = spei_all[i, j, len(full_dates_with_leap_spei):len(full_dates_with_leap_spei)+365]
spei = spei_year[hottest_day-135:hottest_day+136]

# ==== Plotting ====
x = np.arange(1, 272)

palette_all = sns.color_palette("tab20b", n_colors=20)
palette = [palette_all[idx] for idx in [14, 9]]
color = {"heat": palette[0], "dry": palette[1]}

# Start and End dates for the x-axis
start_date = datetime.date(2010, 3, 3)
end_date   = datetime.date(2010, 11, 28)
dates = [start_date + datetime.timedelta(days=d) for d in range(len(x))]

# ---------------- Event Processing Function ----------------
def merge_and_filter(mask, min_len=3, merge_gap=0):
    """
    mask: bool array, True if event condition is met
    merge_gap: Events with gaps <= merge_gap days will be merged
    min_len: Keep only merged events with duration >= min_len days
    Returns: [(start, end), ...]
    """
    slices = []
    start = None
    for k, val in enumerate(mask):
        if val and start is None:
            start = k
        elif not val and start is not None:
            slices.append([start, k])
            start = None
    if start is not None:
        slices.append([start, len(mask)])

    if not slices:
        return []

    # Merge based on merge_gap
    merged = [slices[0]]
    for s, e in slices[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Filter based on min_len
    filtered = [(s, e) for s, e in merged if e - s >= min_len]
    return filtered

# ---------------- Visualization ----------------
fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

# --- SPEI Curve (Drought) ---
line_spei, = ax1.plot(dates, spei, color=color["dry"], linewidth=1, alpha=0.8, label='SPEI')
hl_spei = ax1.hlines(y=-1.28, xmin=dates[0], xmax=dates[-1],
                     color='darkgoldenrod', linestyle='--', linewidth=1, label='SPEI threshold')
fill_spei_patch = mpatches.Patch(color=color["dry"], label='Drought')

ax1.fill_between(dates, spei, -1.28, where=(spei < -1.28),
                 interpolate=True, color=color["dry"])

plt.xlabel('Warm season', fontsize=12)
plt.ylabel('SPEI', fontsize=12)

# ---------------- Y-axis Formatting and Ticks ----------------
yticks = [1.0, 0, -1.0, -1.28, -2.0]
ax1.set_yticks(yticks)
ax1.set_ylim(-2.5, 1.0)

def custom_formatter_with_threshold(y, pos):
    if y == 0:
        return "0"
    elif y == -1.28:
        return "-1.28"
    else:
        return f"{y:.1f}"

# --- Tmax Curve (Heatwave) ---
ax2 = ax1.twinx()
line_tmx, = ax2.plot(dates, tmx, color=color["heat"], linewidth=1, alpha=0.8, label='Tmax')
line_tmx_cli, = ax2.plot(dates, tmx_cli, color='brown', linestyle='--', linewidth=1, label='Tmax threshold')

mask_heat = tmx > tmx_cli
event_heat = merge_and_filter(mask_heat, min_len=3, merge_gap=0)
for start, end in event_heat:
    ax2.fill_between(dates[start:end], tmx[start:end], tmx_cli[start:end],
                     color=color["heat"], interpolate=True)
fill_heat_patch = mpatches.Patch(color=color["heat"], label='Heatwave')

plt.ylabel('Maximum temperature (K)', fontsize=12)

# ---------------- Date Axis Formatting ----------------
ax1.set_xlim(dates[0], dates[-1])
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
extra_ticks = [dates[0], dates[-1]]
ax1.set_xticks(list(ax1.get_xticks()) + list(mdates.date2num(extra_ticks)))

def custom_date_formatter(x_val, pos):
    date = mdates.num2date(x_val).date()
    if date == start_date or date == end_date or date.day == 1:
        return date.strftime("%m/%d")
    return ""

ax1.xaxis.set_major_formatter(mticker.FuncFormatter(custom_date_formatter))
plt.setp(ax1.get_xticklabels(), rotation=0)

# Synchronize Right Axis
ax2.set_xlim(ax1.get_xlim())
ax2.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
ax2.xaxis.set_major_formatter(ax1.xaxis.get_major_formatter())

ax1.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_with_threshold))
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Set border line width
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)
ax1.tick_params(width=0.5)

# ---------------- Legend Configuration ----------------
# Row 1: Heatwave related | Row 2: Drought related
handles = [line_tmx, line_spei, line_tmx_cli, 
           hl_spei, fill_heat_patch, fill_spei_patch] 

labels = [h.get_label() for h in handles]

# Place legend inside at the bottom center
ax1.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.005),
           ncol=3, frameon=False, borderaxespad=0, fontsize=12)

# Add Subplot label (Panel 'a')
ax1.text(-0.075, 1.02, 'a', transform=ax1.transAxes, 
        fontsize=14, fontweight='bold', va='top', ha='left')
