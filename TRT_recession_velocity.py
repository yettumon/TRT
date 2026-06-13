"""
TRT — Pure Recession Velocity Directional Analysis
===================================================
논문: Tilted Ruler Theory (TRT)
섹션: 4.3절 | GunSik Kim (Yettumon) 2026

Expected values reproduced by this script (TRT Paper Table 4, Kim 2026):
  v_axis = 77,844 km/s
  v_perp = 89,937 km/s
  Δv     = +12,093 km/s

Note: "97% z분포 / 3~4% 순수 방향 신호" 분해는
      TRT_z_control_analysis.py에서 검증 (다변량 회귀 + 거리 매칭)

Author: GunSik Kim (Yettumon) | 2026
Data:   Pantheon+SH0ES.dat
"""

import numpy as np
import pandas as pd
from scipy import stats
import hashlib, platform, os, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRT: Pure Recession Velocity Directional Analysis")
print("GunSik Kim (Yettumon) 2026")
print("=" * 60)

# ─────────────────────────────────────────
# 환경 정보
# ─────────────────────────────────────────
import scipy as _scipy
print(f"\nPython {platform.python_version()} | "
      f"numpy {np.__version__} | scipy {_scipy.__version__}")

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
AXIS_RA, AXIS_DEC = 330.0, 0.0
# Fixed a priori axis from TRT Section 2.3.
# Determined independently before Pantheon+ directional analysis
# (Dipole Repeller 337.5° + CMB dipole antipode 337.7° → confirmed RA=330°).

C     = 299792.458
CMB_V = 369.0   # CMB dipole (km/s)
LG_V  = 627.0   # Local Group correction (km/s)

# Expected values from TRT Paper Table 4 (Kim 2026)
EXPECTED = {
    'v_axis':  77844,
    'v_perp':  89937,
    'delta_v': 12093
}

# ─────────────────────────────────────────
# 경로 자동 탐색
# ─────────────────────────────────────────
_PATHS = [
    'Pantheon_SH0ES.dat', 'Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon_SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Pantheon_SH0ES.dat',
]
data = None
for path in _PATHS:
    if os.path.exists(path):
        data = pd.read_csv(path, sep=r'\s+', comment='#')
        print(f"\n데이터: {path} (N={len(data)})")
        with open(path, 'rb') as _f:
            _blob = _f.read()
        print(f"MD5    = {hashlib.md5(_blob).hexdigest()}")
        print(f"SHA256 = {hashlib.sha256(_blob).hexdigest()}")
        break
if data is None:
    raise FileNotFoundError("Pantheon+SH0ES.dat 없음.\n다운로드: https://pantheonplussh0es.github.io/")

# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────
def dot_sky(ra, dec, ra0, dec0):
    """구면 내적 (방향 코사인)"""
    r0 = np.radians(ra0); d0 = np.radians(dec0)
    ra = np.radians(ra);  dec = np.radians(dec)
    return np.cos(dec)*np.cos(d0)*np.cos(ra-r0) + np.sin(dec)*np.sin(d0)

# ─────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────
print("\n─── 전처리 ───")
print("보정: CMB(369 km/s)+LG(627 km/s) 직접 벡터 보정")
print("필터: IS_CALIBRATOR==0, z=0.15~0.50")

df = data[(data['IS_CALIBRATOR']==0)
          &(data['zHD']>0.15)
          &(data['zHD']<0.50)].copy()

df['v_pure'] = (C * df['zHD']
                - CMB_V * dot_sky(df['RA'], df['DEC'], 168.0, -7.0)
                - LG_V  * dot_sky(df['RA'], df['DEC'], 276.0, 30.0))

# theta: 단방향 (0~180°)
df['theta'] = np.degrees(np.arccos(np.clip(
    dot_sky(df['RA'], df['DEC'], AXIS_RA, AXIS_DEC), -1, 1)))

print(f"필터 후 N={len(df)}")

# ─────────────────────────────────────────
# 그룹 분류: axis(θ<45°), perp(45°≤θ≤135°), anti-axis(θ>135°)
# Main hypothesis tests axis-parallel vs transverse directions.
# Anti-axis retained only for sensitivity analysis.
# ─────────────────────────────────────────
ax_m   = df[df['theta'] < 45]
pp_m   = df[(df['theta'] >= 45) & (df['theta'] <= 135)]
anti   = df[df['theta'] > 135]

# ─────────────────────────────────────────
# 메인 분석
# ─────────────────────────────────────────
print("\n─── 메인 분석 ───")

v_ax_m = ax_m['v_pure'].mean()
v_pp_m = pp_m['v_pure'].mean()
dv_m   = v_pp_m - v_ax_m
_, p_m  = stats.ttest_ind(pp_m['v_pure'], ax_m['v_pure'], equal_var=False)  # Welch t-test
_, p_mw = stats.mannwhitneyu(pp_m['v_pure'], ax_m['v_pure'], alternative='two-sided')  # robustness check against non-normality

print(f"N_axis={len(ax_m)}, N_perp={len(pp_m)}, N_anti={len(anti)}")
print(f"Axis: {v_ax_m:,.0f} km/s")
print(f"Perp: {v_pp_m:,.0f} km/s")
print(f"Δv = {dv_m:+,.0f} km/s")
print(f"Welch t-test: p={p_m:.2e}")
print(f"Mann-Whitney: p={p_mw:.2e}  (non-parametric robustness check)")

# ─────────────────────────────────────────
# z 분포 균형 검토 (심사자 방어)
# ─────────────────────────────────────────
print("\n─── z 분포 균형 검토 ───")
_, p_z = stats.ttest_ind(ax_m['zHD'], pp_m['zHD'], equal_var=False)
print(f"axis z 평균: {ax_m['zHD'].mean():.4f}")
print(f"perp z 평균: {pp_m['zHD'].mean():.4f}")
print(f"z t-test p={p_z:.4f} {'⚠️ z 불균형 → TRT_z_control_analysis.py 참조' if p_z<0.05 else '✅'}")
if p_z < 0.05:
    print("  → z 통제 후 분석은 TRT_z_control_analysis.py 참조")

# ─────────────────────────────────────────
# z-bin 층화 분석 (보조)
# ─────────────────────────────────────────
print("\n─── z-bin 층화 분석 ───")
print("  (보조 분석 — 다중비교 미보정, 방향 신호 일관성 확인 목적)")
print("  (overlapping z-windows: 독립 검정 아님, 일관성 확인용)")
print(f"{'z range':<15} {'N_ax':>6} {'N_pp':>6} {'Δv':>10} {'p':>10}")
print("-"*50)
for zlo, zhi in [(0.15,0.25),(0.20,0.35),(0.25,0.40),(0.30,0.50)]:
    sub = df[(df['zHD']>=zlo)&(df['zHD']<zhi)]
    a  = sub[sub['theta']<45]['v_pure']
    p2 = sub[(sub['theta']>=45)&(sub['theta']<=135)]['v_pure']
    if len(a)<5 or len(p2)<5: continue
    dv2 = p2.mean()-a.mean()
    _, pv = stats.ttest_ind(a, p2, equal_var=False)
    sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'n.s.'
    print(f"z={zlo}~{zhi}   {len(a):>6} {len(p2):>6} {dv2:>+10,.0f} {pv:>10.4f} {sig}")

# ─────────────────────────────────────────
# Anti-axis 민감도 분석
# ─────────────────────────────────────────
print("\n─── Anti-axis 민감도 분석 ───")
print("  (심사자 대응: axis/perp/anti-axis 3방향 비교)")
print(f"  Anti-axis (θ>135°): N={len(anti)}, mean={anti['v_pure'].mean():,.0f} km/s")
for label, g1, g2 in [
    ("Axis   vs Perp    ", ax_m['v_pure'], pp_m['v_pure']),
    ("Axis   vs Anti    ", ax_m['v_pure'], anti['v_pure']),
    ("Perp   vs Anti    ", pp_m['v_pure'], anti['v_pure']),
]:
    dv_c = g2.mean()-g1.mean()
    _, pv_c = stats.ttest_ind(g1, g2, equal_var=False)
    sig = '***' if pv_c<0.001 else '**' if pv_c<0.01 else '*' if pv_c<0.05 else 'n.s.'
    print(f"  {label} Δv={dv_c:+,.0f} km/s  p={pv_c:.4e} {sig}")
print("  ※ Axis vs Perp가 메인 분석. Anti-axis 비교 결과는 민감도 분석용으로 제시.")
print("  Interpretation of anti-axis ordering is discussed in the manuscript.")

# ─────────────────────────────────────────
# Reproducibility Check
# ─────────────────────────────────────────
print("\n─── Reproducibility Check ───")
_checks = [
    ("v_axis",  round(v_ax_m),  EXPECTED['v_axis'],  50),
    ("v_perp",  round(v_pp_m),  EXPECTED['v_perp'],  50),
    ("delta_v", round(dv_m),    EXPECTED['delta_v'], 50),
]
for _name, _val, _ref, _tol in _checks:
    _ok = abs(_val-_ref) <= _tol
    print(f"  {'PASS' if _ok else 'FAIL':<6} {_name}: {_val} (논문: {_ref}, 허용오차: ±{_tol})")
# p값 방향 확인
print(f"  {'PASS' if p_m < 1e-6 else 'FAIL':<6} p_welch: {p_m:.2e} (기준: <1e-6)")
print("  (허용오차: 동일 데이터·동일 전처리 기준)")

# ─────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"Axis: {v_ax_m:,.0f} km/s (N={len(ax_m)})")
print(f"Perp: {v_pp_m:,.0f} km/s (N={len(pp_m)})")
print(f"Δv   = {dv_m:+,.0f} km/s")
print(f"Welch t-test p = {p_m:.2e}")
print(f"Mann-Whitney  p = {p_mw:.2e}")
print()
print("참고 문헌:")
print("  Brout et al. 2022, ApJ 938, 110 (Pantheon+)")
print("  Hoffman et al. 2017, NatAstron 1, 0036 (Dipole Repeller)")
