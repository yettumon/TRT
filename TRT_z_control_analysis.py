"""
TRT — z-Controlled Directional Velocity Analysis
=================================================
Section 4.3: Separating pure directional signal from z-distribution effect

Key Results:
  Raw signal: Δv = +12,093 km/s (97% from z-distribution, 3~4% directional)

  Analysis 1 — Multivariate regression  v = a + b·z + c·sin(θ):
    c(sin θ) = +335.6 ± 37.7 km/s, p < 0.001
    R² = 0.99984, N = 665

  Analysis 2 — Distance matching  |Δz| < 0.01:
    Δv = +380.6 ± 39.5 km/s, p < 10⁻¹⁶
    N = 205 pairs, z-balance p = 0.93

Pipeline:
  CMB dipole correction:  369 km/s (RA=168°, Dec=-7°)
  Local Group correction: 627 km/s (RA=276°, Dec=30°)
  z filter: 0.15 < z < 0.50
  Axis group: θ < 45°           (N=205)
  Perp group: 45° ≤ θ ≤ 135°   (N=415)

Author: Kim, Gun.Sik (Yettumon) | 2026
Data:   Pantheon+SH0ES.dat (place in same directory)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import lstsq
import os, warnings
warnings.filterwarnings('ignore')

# ── Data load ─────────────────────────────────────────
for path in [
    'Pantheon_SH0ES.dat',
    'data/Pantheon_SH0ES.dat',
    '/mnt/user-data/uploads/Pantheon_SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon+SH0ES.dat',
]:
    if os.path.exists(path):
        df = pd.read_csv(path, sep=r'\s+')
        print(f"Loaded: {path} (N={len(df)})")
        break

c_light = 299792.458

# ── Angular separation (dot product method) ───────────
def dot_sky(ra_deg, dec_deg, ra0_deg, dec0_deg):
    r0  = np.radians(ra0_deg); d0  = np.radians(dec0_deg)
    ra  = np.radians(ra_deg);  dec = np.radians(dec_deg)
    return (np.cos(dec)*np.cos(d0)*np.cos(ra-r0)
            + np.sin(dec)*np.sin(d0))

# ── Pure recession velocity (CMB + LG corrected) ──────
df['v_pure'] = (c_light * df['zHD']
                - 369.0 * dot_sky(df['RA'], df['DEC'], 168.0, -7.0)
                - 627.0 * dot_sky(df['RA'], df['DEC'], 276.0, 30.0))
df['theta']  = np.degrees(np.arccos(np.clip(
                   dot_sky(df['RA'], df['DEC'], 330.0, 0.0), -1, 1)))
df['sin_th'] = np.sin(np.radians(df['theta']))

# ── Sample filter ─────────────────────────────────────
d = df[(df['IS_CALIBRATOR']==0)
       & (df['zHD'] > 0.15)
       & (df['zHD'] < 0.50)].copy()

y = d['v_pure'].values
n = len(d)

# ─────────────────────────────────────────────────────
# Analysis 1: Multivariate regression v = a + b·z + c·sin(θ)
# ─────────────────────────────────────────────────────
def ols(X, y):
    beta, _, _, _ = lstsq(X, y)
    resid = y - X @ beta
    k = X.shape[1]
    s2 = np.sum(resid**2) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t_vals = beta / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n-k))
    SS_tot = np.sum((y - y.mean())**2)
    R2 = 1 - np.sum(resid**2) / SS_tot
    R2_adj = 1 - (np.sum(resid**2)/(n-k)) / (SS_tot/(n-1))
    return beta, se, t_vals, p_vals, R2, R2_adj, resid

print(f"\nSample: N={n}, z=0.15~0.50, IS_CALIBRATOR=0")

X1A = np.column_stack([np.ones(n), d['zHD'].values, d['sin_th'].values])
b, se, t, p, R2, R2a, resid = ols(X1A, y)

print("\n" + "="*60)
print("Analysis 1: v = a + b·z + c·sin(θ)")
print("="*60)
for nm, bi, si, ti, pi in zip(['a (intercept)','b (z)','c (sin θ)'], b, se, t, p):
    sig = '***' if pi<0.001 else '**' if pi<0.01 else '*' if pi<0.05 else ''
    print(f"  {nm:<14}: {bi:>+12.2f} ± {si:.2f} km/s  t={ti:+.3f}  p={pi:.6f} {sig}")
print(f"  R²={R2:.8f},  adj-R²={R2a:.8f},  N={n}")

# Baseline (no z control)
X1B = np.column_stack([np.ones(n), d['sin_th'].values])
b2, se2, t2, p2, _, _, _ = ols(X1B, y)
print(f"\n  Baseline (no z control): c={b2[1]:+.1f} ± {se2[1]:.1f} km/s  p={p2[1]:.4f}")

# z-residual direction comparison
X1C = np.column_stack([np.ones(n), d['zHD'].values])
_, _, _, _, _, _, resid_z = ols(X1C, y)
d2 = d.copy(); d2['resid_z'] = resid_z

mask_ax = d2['theta'] < 45
mask_pp = (d2['theta'] >= 45) & (d2['theta'] <= 135)
r_ax = d2.loc[mask_ax, 'resid_z']
r_pp = d2.loc[mask_pp, 'resid_z']
t_r, p_r = stats.ttest_ind(r_ax, r_pp)

print(f"\n  z-residual comparison:")
print(f"    Axis residual: {r_ax.mean():+.1f} km/s (N={len(r_ax)})")
print(f"    Perp residual: {r_pp.mean():+.1f} km/s (N={len(r_pp)})")
print(f"    Δresidual:     {r_pp.mean()-r_ax.mean():+.1f} km/s  p={p_r:.6f}")

# z-bin stratified
print(f"\n  z-bin stratified Δresidual:")
for lo, hi in [(0.15,0.25),(0.25,0.35),(0.35,0.50)]:
    sub = d2[(d2['zHD']>=lo)&(d2['zHD']<hi)]
    ax_s = sub[sub['theta']<45]['resid_z']
    pp_s = sub[(sub['theta']>=45)&(sub['theta']<=135)]['resid_z']
    if len(ax_s)<5 or len(pp_s)<5: continue
    _, pb = stats.ttest_ind(ax_s, pp_s)
    sig = '***' if pb<0.001 else '**' if pb<0.01 else '*' if pb<0.05 else 'n.s.'
    print(f"    z=[{lo},{hi}): Δ={pp_s.mean()-ax_s.mean():+.0f} km/s  "
          f"N_ax={len(ax_s)} N_pp={len(pp_s)}  p={pb:.4f} {sig}")

# ─────────────────────────────────────────────────────
# Analysis 2: Distance matching |Δz| < 0.01
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Analysis 2: Distance Matching  |Δz| < 0.01")
print("="*60)

ax_df = d[d['theta'] < 45].reset_index(drop=True)
pp_df = d[(d['theta'] >= 45) & (d['theta'] <= 135)].reset_index(drop=True)
print(f"\n  Axis candidates: N={len(ax_df)}")
print(f"  Perp candidates: N={len(pp_df)}")

pairs_ax, pairs_pp = [], []
used_idx = set()
for _, row in ax_df.iterrows():
    diff  = np.abs(pp_df['zHD'].values - row['zHD'])
    valid = np.where((diff < 0.01) & (~pp_df.index.isin(used_idx)))[0]
    if len(valid) == 0: continue
    best = valid[np.argmin(diff[valid])]
    pairs_ax.append(row)
    pairs_pp.append(pp_df.iloc[best])
    used_idx.add(best)

p_ax = pd.DataFrame(pairs_ax).reset_index(drop=True)
p_pp = pd.DataFrame(pairs_pp).reset_index(drop=True)
dv   = p_pp['v_pure'].values - p_ax['v_pure'].values
N    = len(dv)

t_stat, p_t  = stats.ttest_1samp(dv, 0)
_, p_wilc    = stats.wilcoxon(dv)
_, p_mw      = stats.mannwhitneyu(p_pp['v_pure'], p_ax['v_pure'], alternative='two-sided')
_, p_z_bal   = stats.ttest_ind(p_ax['zHD'], p_pp['zHD'])

print(f"\n  Matched pairs: N={N}")
print(f"  mean |Δz|:     {np.abs(p_ax['zHD']-p_pp['zHD']).mean():.5f}")
print(f"\n  Δv = v(perp) - v(axis):")
print(f"    mean   = {dv.mean():+.1f} km/s")
print(f"    median = {np.median(dv):+.1f} km/s")
print(f"    std    = {dv.std():.1f} km/s")
print(f"    SE     = {dv.std()/np.sqrt(N):.1f} km/s")
print(f"\n  Statistical tests:")
print(f"    t-test:    t={t_stat:.4f},  p={p_t:.2e}")
print(f"    Wilcoxon:  p={p_wilc:.2e}")
print(f"    z-balance: p={p_z_bal:.4f} {'✅' if p_z_bal>0.05 else '❌'}")

print(f"\n  z-bin stratified Δv:")
for lo, hi in [(0.15,0.25),(0.25,0.35),(0.35,0.50)]:
    mask = (p_ax['zHD'].values>=lo) & (p_ax['zHD'].values<hi)
    dv_b = dv[mask]
    if len(dv_b)<5: continue
    _, pb = stats.ttest_1samp(dv_b, 0)
    sig = '***' if pb<0.001 else '**' if pb<0.01 else '*' if pb<0.05 else 'n.s.'
    print(f"    z=[{lo},{hi}): N={len(dv_b):3d}  "
          f"Δv={dv_b.mean():+.1f} ± {dv_b.std()/np.sqrt(len(dv_b)):.1f} km/s  "
          f"p={pb:.4f} {sig}")

# ─────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("Verification against paper values")
print("="*60)
checks = [
    ('Analysis 1 c(sinθ)',  b[2],                      335.6,   5.0),
    ('Analysis 1 R²',        R2,                         0.99984, 0.0001),
    ('Analysis 1 N',         float(n),                  665.0,   0.0),
    ('Analysis 2 N pairs',   float(N),                  205.0,   0.0),
    ('Analysis 2 mean|Δz|', np.abs(p_ax['zHD']-p_pp['zHD']).mean(), 0.00097, 0.0001),
    ('Analysis 2 Δv',        dv.mean(),                 380.6,   5.0),
    ('Analysis 2 SE',        dv.std()/np.sqrt(N),       39.5,    2.0),
    ('Analysis 2 z-bal p',   p_z_bal,                   0.93,    0.05),
]
for label, val, ref, tol in checks:
    ok = abs(val-ref) <= tol if tol > 0 else val == ref
    print(f"  {'✅' if ok else '❌'} {label}: {val:.5f} (paper: {ref})")
