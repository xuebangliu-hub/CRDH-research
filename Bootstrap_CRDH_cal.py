# -*- coding: utf-8 -*-
"""
Multi-source CRDH 271-day full-year block bootstrap parallel computation program

Objectives:
1. Heatwave datasets include ERA5, CPC, BE;
2. Drought datasets include 20CRV3_ERA5, JRA55_ERA5, MSWEP_GLEAM, MSWEP_hPET;
3. The 3 heatwave datasets are paired with the 4 drought datasets, resulting in 12 combinations;
4. Each combination intersects with two target periods based on its given common time range:
       P1: 1960–1990
       P2: 1991–2022
5. To reduce the impact of differing sample lengths between the early and late periods, only combinations satisfying the specified sample length are retained:
       P1: only combinations with actual year count = 31 years;
       P2: only combinations with actual year count = 29 or 32 years;
6. Each selected combination and period undergoes independent 271-day full-year block bootstrap;
7. Each selected combination first computes its own PMF_boot;
8. Finally, the PMF_boot of selected combinations for P1 and P2 are averaged equally across grid points;
9. Only the following files are saved:
       P1_1960_1990_pmf_boot.npy
       P2_1991_2022_pmf_boot.npy

CRDH definition is consistent with the original code:
    CRDH = compound overlap days / (total drought days + total heatwave days) × 100

Special cases:
    - If a reconstructed year has no drought or no heatwave: CRDH = NaN
    - If both drought and heatwave exist but do not overlap: CRDH = 0

PMF_boot definition:
    PMF_boot = actual period mean CRDH / block bootstrap null hypothesis mean CRDH

Note:
    This program does not truncate PMF at 50. If color scale limits are needed for plotting,
    they should be handled at the plotting stage, not during statistical calculation.
"""

import os

# ============================================================
# Important: limit BLAS/OpenMP threads inside each parallel process
# ============================================================
# Python layer already uses multi-process parallelism. If each subprocess also starts
# multiple BLAS/OpenMP threads, severe thread oversubscription will occur. Therefore
# it must be limited to 1 before importing numpy.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import gc
import warnings
import numpy as np
import tifffile as tf
from joblib import Parallel, delayed, parallel_backend


# ============================================================
# 0. User parameter settings
# ============================================================

# ------------------------------------------------------------
# 12 "heatwave × drought" combinations and their common time ranges
# ------------------------------------------------------------
# Order strictly corresponds to:
# ERA5 × 4 drought datasets
# CPC  × 4 drought datasets
# BE   × 4 drought datasets
DATA_COMBINATIONS = [
    # ERA5 heatwave
    {'heat': 'ERA5', 'dry': '20CRV3_ERA5', 'common_start': 1951, 'common_end': 2013},
    {'heat': 'ERA5', 'dry': 'JRA55_ERA5',  'common_start': 1959, 'common_end': 2022},
    {'heat': 'ERA5', 'dry': 'MSWEP_GLEAM', 'common_start': 1981, 'common_end': 2019},
    {'heat': 'ERA5', 'dry': 'MSWEP_hPET',  'common_start': 1982, 'common_end': 2019},

    # CPC heatwave
    {'heat': 'CPC', 'dry': '20CRV3_ERA5', 'common_start': 1980, 'common_end': 2013},
    {'heat': 'CPC', 'dry': 'JRA55_ERA5',  'common_start': 1980, 'common_end': 2022},
    {'heat': 'CPC', 'dry': 'MSWEP_GLEAM', 'common_start': 1981, 'common_end': 2019},
    {'heat': 'CPC', 'dry': 'MSWEP_hPET',  'common_start': 1982, 'common_end': 2019},

    # BE heatwave
    {'heat': 'BE', 'dry': '20CRV3_ERA5', 'common_start': 1954, 'common_end': 2013},
    {'heat': 'BE', 'dry': 'JRA55_ERA5',  'common_start': 1959, 'common_end': 2022},
    {'heat': 'BE', 'dry': 'MSWEP_GLEAM', 'common_start': 1981, 'common_end': 2019},
    {'heat': 'BE', 'dry': 'MSWEP_hPET',  'common_start': 1982, 'common_end': 2019},
]

# Two target periods.
TARGET_PERIODS = {
    'P1_1960_1990': (1960, 1990),
    'P2_1991_2022': (1991, 2022),
}

# Warm season length, used for method description.
WARM_SEASON_LENGTH = 271

# Number of bootstrap repetitions.
# Suggested 200 for debugging; 2000–5000 for formal paper.
N_BOOT = 2000

# Random seed to ensure reproducibility.
RANDOM_SEED = 20260827

# ------------------------------------------------------------
# Parallel settings
# ------------------------------------------------------------
# 64-core machine uses 56 processes by default, leaving some cores for system and disk I/O.
N_JOBS = min(56, os.cpu_count() or 1)

# Bootstrap batch size.
# When set to None, it is automatically adjusted based on N_BOOT and N_JOBS so that
# the number of tasks is approximately twice the number of parallel processes.
# If a fixed batch is desired, it can be changed to an integer, e.g., 25.
BOOT_BATCH = None

# joblib uses read-only memory mapping for larger arrays to reduce array copying in multiprocessing.
JOBLIB_MAX_NBYTES = '10M'

# ------------------------------------------------------------
# Allowed actual sample lengths for each period to participate in multi-source averaging
# ------------------------------------------------------------
# P1 only uses combinations that fully cover 1960–1990, i.e., 31 years.
# P2 only uses combinations with actual coverage of 29 or 32 years.
# This avoids including combinations with only 9–11 years in P1 or 23 years in P2 directly
# into the average, thereby reducing the impact of sample length differences on PMF comparison.
ALLOWED_YEAR_COUNTS = {
    'P1_1960_1990': {31},
    'P2_1991_2022': {29, 32},
}

# ------------------------------------------------------------
# Date coordinate settings
# ------------------------------------------------------------
# Original NH_*_Start/End may be warm-season relative day-of-year or full-year day-of-year.
# Retain a unified 0–366 coordinate without changing the original date labels.
DAY_COORD_MAX = 366
DAY_AXIS_SIZE = DAY_COORD_MAX + 1

# ------------------------------------------------------------
# Input and output paths
# ------------------------------------------------------------
# The following are Windows paths; if running on a Linux server, just modify these 4 root paths.
REFERENCE_PATH = '.../reference_no_antarctica.tif'

DRY_ROOT = '.../Droughts/180days/3days_SPEI128/Observed'
HEAT_ROOT = '.../Heatwaves'

OUTPUT_ROOT = '.../Compound/180days/results/SI/bootstrap_PMF/All_obs_mean'


# ============================================================
# 1. Basic utility functions
# ============================================================
def get_actual_years(common_start, common_end, target_start, target_end):
    """
    Calculate the years that a data combination can actually use within the target period.

    Actual years = intersection of the combination common time range and the target period.
    """
    start_year = max(common_start, target_start)
    end_year = min(common_end, target_end)

    if start_year > end_year:
        return np.array([], dtype=int)

    return np.arange(start_year, end_year + 1, dtype=int)


def event_file_paths(dry_name, heat_name, event_type, year):
    """Return the start/end file paths for drought or heatwave events for a given combination and year."""
    if event_type == 'dry':
        folder = os.path.join(DRY_ROOT, dry_name, '3days_events')
        start_file = os.path.join(folder, f'NH_Dry_Start_{year}.npy')
        end_file = os.path.join(folder, f'NH_Dry_End_{year}.npy')

    elif event_type == 'heat':
        folder = os.path.join(HEAT_ROOT, heat_name, '3days_events')
        start_file = os.path.join(folder, f'NH_Heat_Start_{year}.npy')
        end_file = os.path.join(folder, f'NH_Heat_End_{year}.npy')

    else:
        raise ValueError("event_type must be 'dry' or 'heat'")

    return start_file, end_file


def check_required_files(dry_name, heat_name, years):
    """Check whether the files for the years actually participating in the current combination are complete."""
    missing = []

    for year in years:
        for event_type in ('dry', 'heat'):
            start_file, end_file = event_file_paths(
                dry_name,
                heat_name,
                event_type,
                year,
            )

            if not os.path.exists(start_file):
                missing.append(start_file)
            if not os.path.exists(end_file):
                missing.append(end_file)

    if missing:
        msg = '\n'.join(missing[:20])
        if len(missing) > 20:
            msg += f'\n……Additionally {len(missing) - 20} missing files.'

        raise FileNotFoundError(
            f'{heat_name} × {dry_name}: Data files for years actually participating are incomplete.\n'
            f'Missing files:\n{msg}'
        )


def validate_event_day_range(starts, ends, year, event_type, source_name):
    """Check whether event start/end dates lie within the allowed date coordinate range."""
    valid_start = starts[np.isfinite(starts)]
    valid_end = ends[np.isfinite(ends)]

    if valid_start.size == 0 and valid_end.size == 0:
        return

    values = []
    if valid_start.size:
        values.extend([valid_start.min(), valid_start.max()])
    if valid_end.size:
        values.extend([valid_end.min(), valid_end.max()])

    min_day = np.min(values)
    max_day = np.max(values)

    if min_day < 0 or max_day > DAY_COORD_MAX:
        raise ValueError(
            f'{source_name}, {event_type}, {year}: event date range is '
            f'{min_day}–{max_day}, outside allowed 0–{DAY_COORD_MAX}.'
        )


def annual_events_to_daily_mask(start_arr, end_arr, land_mask,
                                year, event_type, source_name):
    """
    Convert one year's event start/end arrays into a daily boolean array.

    Input shape:
        (lat, lon, event)

    Output shape:
        (DAY_AXIS_SIZE, lat, lon)

    Mathematically equivalent to the original code that expands dates per event using
    np.arange(start, end + 1).
    """
    if start_arr.shape != end_arr.shape:
        raise ValueError(
            f'{source_name}, {event_type}, {year}: start/end array shapes are inconsistent.'
        )

    if start_arr.ndim != 3:
        raise ValueError(
            f'{source_name}, {event_type}, {year}: expected 3D array '
            f'(lat, lon, event), got {start_arr.shape}'
        )

    validate_event_day_range(
        start_arr,
        end_arr,
        year,
        event_type,
        source_name,
    )

    nlat, nlon, _ = start_arr.shape
    daily = np.zeros((DAY_AXIS_SIZE, nlat, nlon), dtype=bool)

    # Vectorized judgement across all grid points and events for each date.
    for day in range(DAY_AXIS_SIZE):
        daily[day] = np.any(
            (start_arr <= day) & (end_arr >= day),
            axis=2,
        )

    # Set ocean and Antarctic regions to False.
    daily[:, ~land_mask] = False

    return daily


def load_period_daily_masks(dry_name, heat_name, years, land_mask):
    """
    Read all years of a data combination that actually exist within a target period,
    and convert them into daily drought/heatwave boolean sequences.

    The date order within each complete year block is completely unchanged.
    """
    check_required_files(dry_name, heat_name, years)

    nlat, nlon = land_mask.shape
    ny = len(years)

    dry_daily = np.zeros(
        (ny, DAY_AXIS_SIZE, nlat, nlon),
        dtype=bool,
    )
    heat_daily = np.zeros_like(dry_daily)

    for yi, year in enumerate(years):
        print(
            f'    Reading {heat_name} × {dry_name}: {year}'
        )

        dry_start_file, dry_end_file = event_file_paths(
            dry_name, heat_name, 'dry', year
        )
        heat_start_file, heat_end_file = event_file_paths(
            dry_name, heat_name, 'heat', year
        )

        dry_start = np.load(dry_start_file)
        dry_end = np.load(dry_end_file)
        heat_start = np.load(heat_start_file)
        heat_end = np.load(heat_end_file)

        dry_daily[yi] = annual_events_to_daily_mask(
            dry_start,
            dry_end,
            land_mask,
            year,
            'dry',
            dry_name,
        )

        heat_daily[yi] = annual_events_to_daily_mask(
            heat_start,
            heat_end,
            land_mask,
            year,
            'heat',
            heat_name,
        )

        del dry_start, dry_end, heat_start, heat_end

    return dry_daily, heat_daily


# Lookup table for the number of binary '1's in each uint8 value from 0 to 255.
# Used after np.packbits to quickly count overlapping drought and heatwave days.
POPCOUNT = np.array(
    [bin(i).count('1') for i in range(256)],
    dtype=np.uint8,
)


# ============================================================
# 2. Parallel construction of "drought year × heatwave year" CRDH pairing matrix
# ============================================================
def _pairwise_one_drought_year(a, dry_packed, heat_packed,
                               dry_days, heat_days, land_mask):
    """Compute CRDH for one fixed drought year a with all heatwave years in the current period."""
    ny = heat_packed.shape[0]
    nlat, nlon = land_mask.shape

    result_a = np.full(
        (ny, nlat, nlon),
        np.nan,
        dtype=np.float32,
    )

    d = dry_days[a]

    for b in range(ny):
        h = heat_days[b]

        # Bitwise AND of packed binary sequences to obtain days with both drought and heatwave.
        both = np.bitwise_and(dry_packed[a], heat_packed[b])

        # Count compound overlap days.
        overlap = POPCOUNT[both].sum(axis=0, dtype=np.int16)

        denom = d + h

        # Consistent with original CRDH code: CRDH defined only when both drought and heatwave exist.
        valid = (d > 0) & (h > 0) & land_mask

        out = np.full((nlat, nlon), np.nan, dtype=np.float32)

        # If both exist but no overlap, overlap=0, so CRDH=0.
        out[valid] = overlap[valid] / denom[valid] * 100.0

        result_a[b] = out

    return a, result_a


def build_pairwise_crdh(dry_daily, heat_daily, land_mask, n_jobs):
    """
    Parallel precomputation of annual CRDH for all "drought year × heatwave year" pairs in the current period.

    pair_crdh[a, b] represents:
        The annual CRDH obtained by pairing the a-th complete drought year block
        with the b-th complete heatwave year block.
    """
    ny, _, nlat, nlon = dry_daily.shape

    if heat_daily.shape != dry_daily.shape:
        raise ValueError('Shapes of dry_daily and heat_daily must be exactly the same.')

    dry_days = dry_daily.sum(axis=1, dtype=np.int16)
    heat_days = heat_daily.sum(axis=1, dtype=np.int16)

    # Pack daily boolean arrays into bit arrays to speed up year-to-year pairing.
    dry_packed = np.packbits(dry_daily, axis=1)
    heat_packed = np.packbits(heat_daily, axis=1)

    # At most 32 drought-year tasks in the current period, no need to start more than ny workers.
    pair_jobs = min(n_jobs, ny)

    print(
        f'    Start building year pairing matrix: {ny} × {ny}, '
        f'using {pair_jobs} parallel processes'
    )

    with parallel_backend('loky', inner_max_num_threads=1):
        results = Parallel(
            n_jobs=pair_jobs,
            verbose=5,
            max_nbytes=JOBLIB_MAX_NBYTES,
            mmap_mode='r',
        )(
            delayed(_pairwise_one_drought_year)(
                a,
                dry_packed,
                heat_packed,
                dry_days,
                heat_days,
                land_mask,
            )
            for a in range(ny)
        )

    pair_crdh = np.full(
        (ny, ny, nlat, nlon),
        np.nan,
        dtype=np.float32,
    )

    for a, result_a in results:
        pair_crdh[a] = result_a

    pair_crdh[:, :, ~land_mask] = np.nan

    del results, dry_days, heat_days, dry_packed, heat_packed
    gc.collect()

    return pair_crdh


def observed_period_crdh(pair_crdh):
    """
    Calculate the period-mean CRDH for actual same-year pairing.

    The main diagonal of pair_crdh represents:
        drought(y) × heatwave(y)
    i.e., actual same-year event pairing.
    """
    ny = pair_crdh.shape[0]
    idx = np.arange(ny)
    observed_annual = pair_crdh[idx, idx]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        observed = np.nanmean(observed_annual, axis=0)

    return observed.astype(np.float32)


# ============================================================
# 3. Parallel 271-day full-year block bootstrap
# ============================================================
def _bootstrap_batch(pair_crdh, dry_idx_batch, heat_idx_batch):
    """Compute period-mean CRDH for a batch of bootstrap repetitions."""
    # selected:
    # (bootstrap_batch, reconstructed_year, lat, lon)
    selected = pair_crdh[dry_idx_batch, heat_idx_batch]

    finite = np.isfinite(selected)

    numerator = np.where(finite, selected, 0.0).sum(
        axis=1,
        dtype=np.float32,
    )
    denominator = finite.sum(axis=1, dtype=np.int16)

    batch_mean = np.full(numerator.shape, np.nan, dtype=np.float32)

    np.divide(
        numerator,
        denominator,
        out=batch_mean,
        where=denominator > 0,
    )

    return batch_mean


def get_effective_boot_batch(n_boot, n_jobs):
    """
    Automatically determine bootstrap batch size.

    The goal is to make the number of tasks approximately twice the number of effective parallel processes,
    avoiding both idle CPUs due to too few tasks and excessive scheduling overhead from too fragmented tasks.
    """
    if BOOT_BATCH is not None:
        return int(BOOT_BATCH)

    target_tasks = max(1, min(n_boot, n_jobs * 2))
    return max(1, int(np.ceil(n_boot / target_tasks)))


def independent_year_block_bootstrap(pair_crdh, n_boot, rng, n_jobs):
    """
    Perform full-year block bootstrap for the current combination and period.

    Each bootstrap:
    1. Draw ny complete drought year blocks with replacement from the current period;
    2. Independently draw ny complete heatwave year blocks with replacement;
    3. Re-pair the k-th drawn drought year with the k-th drawn heatwave year;
    4. Average the reconstructed ny annual CRDH values to obtain a period mean.

    Because the order within each year is not rearranged, the following are preserved:
        - intra-warm-season seasonal structure;
        - event durations;
        - serial autocorrelation;
        - within-year spatial structure.

    Because drought and heatwave years are sampled independently, the actual same-year D-H pairing is broken.
    """
    ny, _, nlat, nlon = pair_crdh.shape

    dry_idx_all = rng.integers(
        0,
        ny,
        size=(n_boot, ny),
        endpoint=False,
        dtype=np.int16,
    )

    heat_idx_all = rng.integers(
        0,
        ny,
        size=(n_boot, ny),
        endpoint=False,
        dtype=np.int16,
    )

    batch_size = get_effective_boot_batch(n_boot, n_jobs)

    batch_ranges = [
        (b0, min(b0 + batch_size, n_boot))
        for b0 in range(0, n_boot, batch_size)
    ]

    boot_jobs = min(n_jobs, len(batch_ranges))

    print(
        f'    Start Bootstrap: {n_boot} repetitions, '
        f'{batch_size} per batch, {len(batch_ranges)} tasks total, '
        f'using {boot_jobs} parallel processes'
    )

    with parallel_backend('loky', inner_max_num_threads=1):
        batch_results = Parallel(
            n_jobs=boot_jobs,
            verbose=5,
            max_nbytes=JOBLIB_MAX_NBYTES,
            mmap_mode='r',
        )(
            delayed(_bootstrap_batch)(
                pair_crdh,
                dry_idx_all[b0:b1],
                heat_idx_all[b0:b1],
            )
            for b0, b1 in batch_ranges
        )

    bootstrap = np.concatenate(batch_results, axis=0).astype(
        np.float32,
        copy=False,
    )

    if bootstrap.shape != (n_boot, nlat, nlon):
        raise RuntimeError(
            f'Bootstrap output shape incorrect: {bootstrap.shape}, '
            f'expected {(n_boot, nlat, nlon)}'
        )

    del dry_idx_all, heat_idx_all, batch_results
    gc.collect()

    return bootstrap


# ============================================================
# 4. PMF calculation
# ============================================================
def safe_ratio(num, den):
    """Safely compute num / den; returns NaN when denominator <= 0 or either is NaN."""
    out = np.full(num.shape, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(num) &
        np.isfinite(den) &
        (den > 0)
    )

    out[valid] = num[valid] / den[valid]

    return out


def calculate_pmf_for_one_combination_period(
    dry_name,
    heat_name,
    years,
    land_mask,
    rng,
):
    """
    Calculate PMF_boot for one "heatwave × drought" combination in one target period.

    Returns:
        pmf_boot : (lat, lon)
    """
    print(
        f'    Actual years used: {years[0]}–{years[-1]}, total {len(years)} years'
    )

    # 1. Read daily event sequences.
    dry_daily, heat_daily = load_period_daily_masks(
        dry_name,
        heat_name,
        years,
        land_mask,
    )

    # 2. Precompute annual CRDH for all "drought year × heatwave year" pairs.
    pair_crdh = build_pairwise_crdh(
        dry_daily,
        heat_daily,
        land_mask,
        N_JOBS,
    )

    del dry_daily, heat_daily
    gc.collect()

    # 3. The main diagonal is the actual same-year pairing, compute period-mean CRDH.
    observed = observed_period_crdh(pair_crdh)

    # 4. Independent full-year block bootstrap to establish null hypothesis CRDH distribution.
    bootstrap = independent_year_block_bootstrap(
        pair_crdh,
        N_BOOT,
        rng,
        N_JOBS,
    )

    # 5. Compute null hypothesis mean CRDH.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        null_mean = np.nanmean(bootstrap, axis=0).astype(np.float32)

    # 6. PMF_boot = observed / null_mean.
    pmf_boot = safe_ratio(observed, null_mean)
    pmf_boot[~land_mask] = np.nan

    # This program ultimately only needs PMF, so release intermediate variables immediately.
    del pair_crdh, observed, bootstrap, null_mean
    gc.collect()

    return pmf_boot


# ============================================================
# 5. Main program: filter combinations by sample length, then compute multi-source mean PMF
# ============================================================
def main():
    print('=' * 90)
    print('Multi-source CRDH 271-day full-year block bootstrap parallel analysis')
    print(f'System detected CPU: {os.cpu_count()}')
    print(f'Maximum parallel processes actually used: {N_JOBS}')
    print(f'Bootstrap repetitions: {N_BOOT}')
    print(f'Number of candidate data combinations: {len(DATA_COMBINATIONS)}')
    print('P1 inclusion rule: actual year count must be 31 years')
    print('P2 inclusion rule: actual year count must be 29 or 32 years')
    print('=' * 90)

    # --------------------------------------------------------
    # 5.1 Read land mask
    # --------------------------------------------------------
    reference = tf.imread(REFERENCE_PATH).astype(float)

    # Remove Antarctica, consistent with original code.
    reference[62:, :] = np.nan
    land_mask = np.isfinite(reference)

    # Store PMF of data combinations meeting sample length requirements for each period.
    period_pmf_list = {
        label: []
        for label in TARGET_PERIODS
    }

    # Also store the names of data combinations actually participating in the average, for runtime checking.
    period_used_combinations = {
        label: []
        for label in TARGET_PERIODS
    }

    # --------------------------------------------------------
    # 5.2 Pre-determine which "combination × period" actually participate
    # --------------------------------------------------------
    selected_jobs = []

    for combo_index, combo in enumerate(DATA_COMBINATIONS, start=1):
        for period_label, (target_start, target_end) in TARGET_PERIODS.items():
            years = get_actual_years(
                combo['common_start'],
                combo['common_end'],
                target_start,
                target_end,
            )

            year_count = len(years)
            allowed_counts = ALLOWED_YEAR_COUNTS[period_label]

            if year_count in allowed_counts:
                selected_jobs.append(
                    (combo_index, combo, period_label, years)
                )

    # Generate independent and reproducible random streams for all actually participating "combination-period".
    seed_sequence = np.random.SeedSequence(RANDOM_SEED)
    child_seeds = seed_sequence.spawn(len(selected_jobs))

    # --------------------------------------------------------
    # 5.3 Print final selected combinations
    # --------------------------------------------------------
    print('\n' + '=' * 90)
    print('Data combinations after filtering by actual year count:')
    print('=' * 90)

    for period_label, (target_start, target_end) in TARGET_PERIODS.items():
        selected_for_period = [
            job for job in selected_jobs
            if job[2] == period_label
        ]

        print(
            f'\n{period_label} (target period {target_start}–{target_end}): '
            f'total {len(selected_for_period)} combinations'
        )

        for _, combo, _, years in selected_for_period:
            print(
                f"  {combo['heat']} × {combo['dry']}: "
                f'{years[0]}–{years[-1]}, {len(years)} years'
            )

    # --------------------------------------------------------
    # 5.4 Process each actually selected "combination × period"
    # --------------------------------------------------------
    for job_index, (combo_index, combo, period_label, years) in enumerate(
        selected_jobs,
        start=1,
    ):
        heat_name = combo['heat']
        dry_name = combo['dry']
        common_start = combo['common_start']
        common_end = combo['common_end']
        target_start, target_end = TARGET_PERIODS[period_label]

        print('\n' + '=' * 90)
        print(
            f'Task {job_index:02d}/{len(selected_jobs)}: '
            f'{heat_name} × {dry_name}, {period_label}'
        )
        print(f'Original common time range: {common_start}–{common_end}')
        print(
            f'Target period: {target_start}–{target_end}; '
            f'actual years: {years[0]}–{years[-1]}; total {len(years)} years'
        )
        print('=' * 90)

        rng = np.random.default_rng(child_seeds[job_index - 1])

        pmf_boot = calculate_pmf_for_one_combination_period(
            dry_name,
            heat_name,
            years,
            land_mask,
            rng,
        )

        # Keep only in memory; do not output individual combination files.
        period_pmf_list[period_label].append(pmf_boot)
        period_used_combinations[period_label].append(
            f'{heat_name} × {dry_name}'
        )

        # Print spatial mean as simple diagnostic.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            spatial_mean = np.nanmean(pmf_boot)

        print(
            f'    {heat_name} × {dry_name}, {period_label} completed; '
            f'spatial mean PMF = {spatial_mean:.4f}'
        )

    # --------------------------------------------------------
    # 5.5 Average selected combinations equally across grid points for P1 and P2 separately
    # --------------------------------------------------------
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for period_label in TARGET_PERIODS:
        expected_counts = {
            'P1_1960_1990': 4,
            'P2_1991_2022': 9,
        }

        actual_count = len(period_pmf_list[period_label])
        expected_count = expected_counts[period_label]

        if actual_count != expected_count:
            raise RuntimeError(
                f'{period_label} actually selected {actual_count} combinations, '
                f'but according to current common time configuration, {expected_count} were expected.'
            )

        pmf_stack = np.stack(
            period_pmf_list[period_label],
            axis=0,
        ).astype(np.float32)

        # Equal-weight average over all valid selected combinations at each grid point.
        # If a combination is NaN at that grid point, only remaining valid combinations are used.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_pmf = np.nanmean(pmf_stack, axis=0).astype(np.float32)

        mean_pmf[~land_mask] = np.nan

        # Finally output only two multi-source mean PMF files.
        output_file = os.path.join(
            OUTPUT_ROOT,
            f'{period_label}_pmf_boot.npy',
        )
        np.save(output_file, mean_pmf)

        # Count the number of data combinations actually contributing at each grid point, only for diagnostics, not saved.
        valid_dataset_count = np.isfinite(pmf_stack).sum(axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            global_mean = np.nanmean(mean_pmf)
            mean_valid_sources = np.nanmean(
                np.where(land_mask, valid_dataset_count, np.nan)
            )

        print('\n' + '=' * 90)
        print(f'{period_label} multi-source mean PMF saved:')
        print(output_file)
        print(f'Number of data combinations participating: {actual_count}')
        print('Participating combinations:')
        for name in period_used_combinations[period_label]:
            print(f'  {name}')
        print(f'Spatial mean PMF: {global_mean:.4f}')
        print(
            f'Mean valid data combination count over land grid points: '
            f'{mean_valid_sources:.2f} / {actual_count}'
        )
        print('=' * 90)

        del pmf_stack, mean_pmf, valid_dataset_count
        gc.collect()

    print('\nAll data combinations meeting sample length requirements have been processed.')
    print('Final outputs only:')
    print(os.path.join(OUTPUT_ROOT, 'P1_1960_1990_pmf_boot.npy'))
    print(os.path.join(OUTPUT_ROOT, 'P2_1991_2022_pmf_boot.npy'))


# ============================================================
# 6. Program entry point
# ============================================================
if __name__ == '__main__':
    main()