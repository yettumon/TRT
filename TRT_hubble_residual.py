"""
TRT — Pantheon+ Hubble Residual Directional Analysis
=====================================================
Section 4.2: Directional comparison of Hubble residuals

Key Results:
  Axis group (θ_min < 45°, N=603): μ = −0.0484 mag
  Perp group (θ_min > 60°, N=709): μ = −0.0668 mag
  Δμ = +0.0184 mag, p = 0.0295

Author: Kim, Gun.Sik (Yettumon) | 2026
Data:   Pantheon+SH0ES.dat (place in data/ directory)
"""

import numpy as np
import pandas as pd
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import warnings
warnings.filterwarnings('ignore')

# ── Settings ──────────────────────────────────────
AXIS_RA, AXIS_DEC = 330.0, 0.0
ANTI_RA, ANTI_DEC = 150.0, 0.0
H0, Om0 = 70.0, 0.3
AXIS_CUT, PERP_CUT = 45.0, 60.0

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
cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
df['mu_theory'] = cosmo.distmod(df['zHD'].values).value
df['residual']  = df['MU_SH0ES'] - df['mu_theory']
df['theta']     = angular_sep(df['RA'].values, df['DEC'].values)
df['theta_anti']= angular_sep(df['RA'].values, df['DEC'].values, ANTI_RA, ANTI_DEC)
df['theta_min'] = np.minimum(df['theta'], df['theta_anti'])

ax = df[df['theta_min'] < AXIS_CUT]
pp = df[df['theta_min'] > PERP_CUT]
mid= df[(df['theta_min']>=AXIS_CUT)&(df['theta_min']<=PERP_CUT)]

# ── Results ────────────────────────────────────────
print("\n" + "="*55)
print("TRT: Pantheon+ Hubble Residual Directional Analysis")
print("="*55)
print(f"\nCosmological sample: N={len(df)} (IS_CALIBRATOR=0, z>0.01)")
print(f"Axis group  (θ_min<{AXIS_CUT}°): N={len(ax)}")
print(f"Middle group ({AXIS_CUT}°~{PERP_CUT}°):  N={len(mid)} [excluded]")
print(f"Perp group  (θ_min>{PERP_CUT}°): N={len(pp)}")

print(f"\nAxis  mean residual: μ = {ax['residual'].mean():.4f} mag")
print(f"Perp  mean residual: μ = {pp['residual'].mean():.4f} mag")

dmu = ax['residual'].mean() - pp['residual'].mean()
t, p = stats.ttest_ind(ax['residual'], pp['residual'])
print(f"\nΔμ (axis − perp) = {dmu:+.4f} mag")
print(f"t = {t:.3f}, p = {p:.4f}")

# ── Verification ───────────────────────────────────
print("\n── Verification against paper ──")
print(f"Paper: Δμ = +0.0184 mag, p = 0.0295")
print(f"{'✅ MATCH' if abs(dmu-0.0184)<0.002 else '⚠️ CHECK'}: Δμ = {dmu:+.4f}")
print(f"{'✅ MATCH' if abs(p-0.0295)<0.01 else '⚠️ CHECK'}: p = {p:.4f}")

# ── Calibrator directional check ───────────────────
cal = data[data['IS_CALIBRATOR']==1].copy()
cal['theta']     = angular_sep(cal['RA'].values, cal['DEC'].values)
cal['theta_anti']= angular_sep(cal['RA'].values, cal['DEC'].values, ANTI_RA, ANTI_DEC)
cal['theta_min'] = np.minimum(cal['theta'], cal['theta_anti'])

cos_sample = df.copy()
pct_cal = (cal['theta_min'] > 90).mean() * 100
pct_cos = (cos_sample['theta'] > 90).mean() * 100

print(f"\n── Calibrator directional bias ──")
print(f"Calibrators θ>90°: {pct_cal:.1f}%")
print(f"Cosmological  θ>90°: {pct_cos:.1f}%")
print(f"Ratio: {pct_cal/pct_cos:.1f}×")
