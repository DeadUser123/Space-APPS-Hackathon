import pandas as pd
import numpy as np

# Gravitational constant and conversions
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN = 1.98847e30  # kg
AU = 1.496e11  # m
DAY = 86400  # s

# ---------- Load CSVs ----------
toi = pd.read_csv("toi.csv", comment='#')
k2 = pd.read_csv("k2.csv", comment='#')
koi = pd.read_csv("koi.csv", comment='#')

# column renaming (cuz NASA didn't do this)
rename_map = {
    'pl_orbper': 'orbital_period', 'koi_period': 'orbital_period',
    'pl_trandurh': 'transit_duration', 'pl_trandur': 'transit_duration', 'koi_duration': 'transit_duration',
    'pl_trandep': 'transit_depth', 'koi_depth': 'transit_depth',
    'pl_rade': 'planet_radius', 'koi_prad': 'planet_radius',
    'pl_orbsmax': 'semi_major_axis', 'koi_sma': 'semi_major_axis',
    'pl_insol': 'insolation_flux', 'koi_insol': 'insolation_flux',
    'pl_eqt': 'equilibrium_temp', 'koi_teq': 'equilibrium_temp',

    'st_teff': 'stellar_teff', 'koi_steff': 'stellar_teff',
    'st_rad': 'stellar_radius', 'koi_srad': 'stellar_radius',
    'st_logg': 'stellar_logg', 'koi_slogg': 'stellar_logg',

    'tfopwg_disp': 'disposition', 'disposition': 'disposition', 'koi_disposition': 'disposition'
}

toi = toi.rename(columns=rename_map)
k2 = k2.rename(columns=rename_map)
koi = koi.rename(columns=rename_map)
print(toi.columns)
print(k2.columns)
print(koi.columns)

keep_cols = [
    'disposition',
    'orbital_period',
    'transit_duration',
    'transit_depth',
    'planet_radius',
    'semi_major_axis',
    'insolation_flux',
    'equilibrium_temp',
    'stellar_teff',
    'stellar_radius',
    'stellar_logg'
]

toi = toi[toi.columns.intersection(keep_cols)]
k2 = k2[k2.columns.intersection(keep_cols)]
koi = koi[koi.columns.intersection(keep_cols)]

k2['transit_depth'] = k2['transit_depth'] * 1e4 # k2 decided to be special for some reason and use percentages

# Convert missing semi-major axis in TOI using Kepler's Third Law
def derive_sma(period_days, logg, radius_solar):
    """
    Derive semi-major axis (in AU) for TOI planets.
    If stellar mass is unavailable, approximate from log(g) and radius.
    """
    try:
        if pd.isna(period_days) or pd.isna(logg) or pd.isna(radius_solar):
            return np.nan
        # g = 10^logg cm/s^2 = 10^(logg - 2) m/s^2
        g_m_s2 = 10 ** (logg - 2)
        R = radius_solar * 6.957e8  # solar radii → meters
        M = g_m_s2 * R**2 / G  # kg
        P = period_days * DAY
        a = ((G * M * P**2) / (4 * np.pi**2)) ** (1/3) / AU
        return a
    except Exception:
        return np.nan

if 'semi_major_axis' not in toi.columns or toi['semi_major_axis'].isna().all():
    toi['semi_major_axis'] = toi.apply(
        lambda row: derive_sma(row.get('orbital_period'), row.get('stellar_logg'), row.get('stellar_radius')), axis=1
    )

def disposition_to_binary(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip().lower()
    if any(x in val for x in ['confirmed', 'candidate', 'pc', 'cp', 'kp']):
        return 1
    elif any(x in val for x in ['false', 'fp', 'rejected', 'refuted', 'fa']):
        return 0
    else:
        print("Unknown Disposition: ", val) # debug
    return np.nan

for df in [toi, k2, koi]:
    df['disposition'] = df['disposition'].apply(disposition_to_binary)

toi['source'] = 'TOI'
k2['source'] = 'K2'
koi['source'] = 'KOI'

merged = pd.concat([toi, k2, koi], ignore_index=True)

merged = merged.dropna(subset=['orbital_period', 'planet_radius', 'stellar_teff'])

merged.to_csv("merged_exoplanets.csv", index=False)
print(f"✅ Saved {len(merged)} cleaned records to merged_exoplanets.csv")
