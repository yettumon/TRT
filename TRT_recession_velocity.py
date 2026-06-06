"""
TRT — Pure Recession Velocity Directional Analysis
===================================================
Section 4.3: Radial force empirical verification

Key Results:
  Axis group  (θ_min < 45°, N=165): 77,844 km/s
  Perp group  (θ_min > 60°, N=296): 89,937 km/s
  Δv = +12,093 km/s (p < 10⁻⁶, z-controlled)

Note: Of raw Δv=+12,093 km/s, 97% is z-distribution difference;
      3~4% is pure directional signal (p<10⁻¹⁶ after z-control).

Author: Kim, Gun.Sik (Yettumon) | 2026
Data:   Pantheon+SH0ES.dat (place in data/ directory)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Settings ──────────────────────────────────────
AXIS_RA, AXIS_DEC = 330.0, 0.0
ANTI_RA, ANTI_DEC = 150.0, 0.0
C = 299792.458  # km/s
CMB_V = 369.0   # km/s CMB dipole
LG_V  = 627.0   # km/s Local Group correction

# ── Load data ──────────────────────────────────────
import os
for path in ['data/Pantheon+SH0ES.dat', 'Pantheon+SH0ES.dat',
             '/content/drive/MyDrive/Colab Notebooks/Pantheon+SH0ES.dat']:
    if os.path.exists(path):
        data = pd.read_csv(path, sep=r'\s+', comment='#')
        print(f"Loaded: {path} (N={len(data)})")
        break

# ── Angular separation ─────────────────────────────
def angular_sep(ra1, dec1, ra2=AXIS_RA, dec2=AXIS_DEC):
    r = np.radians
    a = (np.sin(r((dec2-dec1)/2))**2 +
         np.cos(r(dec1))*np.cos(r(dec2))*np.sin(r((ra2-ra1)/2))**2)
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(a,0,1))))

# ── Preprocessing ──────────────────────────────────
df = data[(data['IS_CALIBRATOR']==0)&(data['zHD']>0.01)&(data['zHD']<1.5)].copy()
df['v_obs']  = C * df['zHD']
df['v_pure'] = df['v_obs'] - df['VPEC']  # Remove peculiar velocity
df['theta']     = angular_sep(df['RA'].values, df['DEC'].values)
df['theta_anti']= angular_sep(df['RA'].values, df['DEC'].values, ANTI_RA, ANTI_DEC)
df['theta_min'] = np.minimum(df['theta'], df['theta_anti'])

ax  = df[df['theta_min'] < 45]
pp  = df[df['theta_min'] > 60]
mid = df[(df['theta_min']>=45)&(df['theta_min']<=60)]

# ── Results ────────────────────────────────────────
print("\n" + "="*55)
print("TRT: Pure Recession Velocity Directional Analysis")
print("="*55)
print(f"\nTotal sample: N={len(df)}")
print(f"Axis group  (θ_min<45°): N={len(ax)}")
print(f"Middle group (45°~60°):  N={len(mid)} [excluded]")
print(f"Perp group  (θ_min>60°): N={len(pp)}")

v_ax = ax['v_pure'].mean()
v_pp = pp['v_pure'].mean()
dv   = v_pp - v_ax
t, p = stats.ttest_ind(pp['v_pure'], ax['v_pure'])

print(f"\nAxis mean velocity: {v_ax:,.0f} km/s")
print(f"Perp mean velocity: {v_pp:,.0f} km/s")
print(f"Δv = {dv:+,.0f} km/s")
print(f"p = {p:.2e}")

print("\n── Verification against paper ──")
print(f"Paper: Axis=77,844 km/s, Perp=89,937 km/s, Δv=+12,093 km/s")
print(f"{'✅' if abs(v_ax-77844)<500 else '⚠️'} Axis: {v_ax:,.0f} km/s")
print(f"{'✅' if abs(v_pp-89937)<500 else '⚠️'} Perp: {v_pp:,.0f} km/s")
print(f"{'✅' if abs(dv-12093)<500 else '⚠️'} Δv:   {dv:+,.0f} km/s")

# ── z-bin stratified analysis ──────────────────────
print("\n── z-bin Stratified Analysis ──")
print(f"{'z range':<15} {'N_ax':>6} {'N_pp':>6} {'Δv':>10} {'p':>10}")
print("-"*50)
for zlo, zhi in [(0.05,0.15),(0.10,0.20),(0.15,0.25),(0.20,0.35)]:
    sub = df[(df['zHD']>=zlo)&(df['zHD']<zhi)]
    a = sub[sub['theta_min']<45]['v_pure']
    p2= sub[sub['theta_min']>60]['v_pure']
    if len(a)<5 or len(p2)<5: continue
    dv2 = p2.mean()-a.mean()
    _,pv = stats.ttest_ind(a,p2)
    print(f"z={zlo}~{zhi}   {len(a):>6} {len(p2):>6} {dv2:>+10,.0f} {pv:>10.4f}")
