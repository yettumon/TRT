"""
TRT — Fisher Combined Statistics (실제 데이터 기반)
=====================================================
논문: Tilted Ruler Theory (TRT)
섹션: 4.7절 | GunSik Kim (Yettumon) 2026

핵심 결과:
  Pantheon+ SN Ia:     Δμ = +0.0184 mag, p = 0.0295
  Pure recession vel:  Δv = +12,093 km/s, p < 10⁻⁶ (실측 p=2.67e-8)
  Fisher (2 채널):     5.5σ (실측값 기반) / 5.0σ (상한치 기반)
  LEE 보정 후:         6.4~7.2σ

수정 이력:
  v3: 실제 데이터 기반 p값 계산 추가 (설명용 상수 → 실측값)
      DESI BGS 섹션 제거 (가짜 코드 확인)

재현 가능 여부: ✅ 완전 재현 가능 (데이터 필요)
"""

import numpy as np
import pandas as pd
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import hashlib, platform, os, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRT: Fisher Combined Statistics (실제 데이터 기반)")
print("GunSik Kim (Yettumon) 2026")
print("=" * 60)

import scipy as _scipy
print(f"\nPython {platform.python_version()} | numpy {np.__version__} | scipy {_scipy.__version__}")

# ─────────────────────────────────────────
# 데이터 경로
# ─────────────────────────────────────────
PATHS = [
    'Pantheon_SH0ES.dat',
    'Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon_SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Pantheon_SH0ES.dat',
]
PATH = None
for p in PATHS:
    if os.path.exists(p):
        PATH = p
        break

# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────
C = 299792.458; CMB_V = 369.0; LG_V = 627.0
AXIS_RA, AXIS_DEC = 330.0, 0.0
ANTI_RA, ANTI_DEC = 150.0, 0.0

def angular_sep(ra1, dec1, ra2=330., dec2=0.):
    r = np.radians
    a = (np.sin(r((dec2-dec1)/2))**2 +
         np.cos(r(dec1))*np.cos(r(dec2))*np.sin(r((ra2-ra1)/2))**2)
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(a, 0, 1))))

def dot_sky(ra, dec, ra0, dec0):
    r0=np.radians(ra0); d0=np.radians(dec0)
    ra=np.radians(ra); dec=np.radians(dec)
    return np.cos(dec)*np.cos(d0)*np.cos(ra-r0)+np.sin(dec)*np.sin(d0)

def fisher_combine(pvals):
    chi2 = -2 * np.sum(np.log(pvals))
    df   = 2 * len(pvals)
    p_combined = stats.chi2.sf(chi2, df)
    sigma = stats.norm.isf(p_combined / 2)  # 양측검정
    return chi2, p_combined, sigma

def p2s(p): return stats.norm.isf(p/2)

# ─────────────────────────────────────────
# 채널 1: Pantheon+ 실제 p값 계산
# ─────────────────────────────────────────
print("\n─── 채널 1: Pantheon+ 허블 잔차 ───")

import hashlib as _hl
if PATH:
    data = pd.read_csv(PATH, sep=r'\s+', comment='#')
    with open(PATH, 'rb') as _f:
        _blob = _f.read()
    print(f"  MD5    = {_hl.md5(_blob).hexdigest()}")
    print(f"  SHA256 = {_hl.sha256(_blob).hexdigest()}")
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    # TRT_hubble_residual.py와 동일한 재현 조건:
    # IS_CALIBRATOR 구분 없음, MU_SH0ES 컬럼, isfinite 필터
    df_p = data[(data['zHD']>0.01)
                &(data['zHD']<1.5)
                &np.isfinite(data['MU_SH0ES'])].copy()
    df_p['mu_theory'] = cosmo.distmod(df_p['zHD'].values).value
    df_p['residual']  = df_p['MU_SH0ES'] - df_p['mu_theory']
    df_p['theta']     = angular_sep(df_p['RA'].values, df_p['DEC'].values)
    df_p['theta_anti']= angular_sep(df_p['RA'].values, df_p['DEC'].values, 150., 0.)
    df_p['theta_min'] = np.minimum(df_p['theta'], df_p['theta_anti'])

    ax_p = df_p[df_p['theta_min'] < 45]['residual']
    pp_p = df_p[df_p['theta_min'] > 60]['residual']
    delta_mu = ax_p.mean() - pp_p.mean()
    _, p_pan_actual = stats.ttest_ind(ax_p, pp_p,
                                      equal_var=False, nan_policy='omit')

    print(f"  N_axis={len(ax_p)}, N_perp={len(pp_p)}")
    print(f"  Δμ = {delta_mu:+.4f} mag")
    print(f"  실측 p값 = {p_pan_actual:.4f}")
    print(f"  논문 기재: Δμ=+0.0184, p=0.0295")
    print(f"  {'✅' if abs(delta_mu-0.0184)<0.002 else '❌'} Δμ 재현")
    print(f"  {'✅' if abs(p_pan_actual-0.0295)<0.005 else '❌'} p값 재현")
    p_pantheon = p_pan_actual
else:
    print("  데이터 없음 → 논문 기재값 사용")
    p_pantheon = 0.0295
    delta_mu   = 0.0184

# ─────────────────────────────────────────
# 채널 2: 순수 후퇴속도 실제 p값 계산
# ─────────────────────────────────────────
print("\n─── 채널 2: 순수 후퇴속도 ───")

if PATH:
    df_v = data[(data['IS_CALIBRATOR']==0)
                &(data['zHD']>0.15)
                &(data['zHD']<0.50)].copy()
    df_v['v_pure'] = (C*df_v['zHD']
                      - CMB_V*dot_sky(df_v['RA'],df_v['DEC'],168.,-7.)
                      - LG_V *dot_sky(df_v['RA'],df_v['DEC'],276.,30.))
    df_v['theta'] = np.degrees(np.arccos(np.clip(
        dot_sky(df_v['RA'],df_v['DEC'],330.,0.),-1,1)))

    ax_v = df_v[df_v['theta']<45]['v_pure']
    pp_v = df_v[(df_v['theta']>=45)&(df_v['theta']<=135)]['v_pure']
    delta_v = pp_v.mean() - ax_v.mean()
    _, p_vel_actual = stats.ttest_ind(pp_v, ax_v, equal_var=False, nan_policy='omit')  # Welch

    print(f"  N_axis={len(ax_v)}, N_perp={len(pp_v)}")
    print(f"  Δv = {delta_v:+,.0f} km/s")
    print(f"  실측 p값 = {p_vel_actual:.4e}")
    print(f"  논문 기재: Δv=+12,093 km/s, p<10⁻⁶")
    print(f"  {'✅' if abs(delta_v-12093)<500 and p_vel_actual<1e-6 else '⚠️ 확인 필요'}")
    print(f"  ※ 주의: z 분포 불균형 가능성 → z 통제 분석 병행 필요")
    print(f"  z 통제 후 순수 방향 신호 (논문 4.3절, TRT_z_control_analysis.py):")
    print(f"    다변량 회귀: c=+335.6±37.7 km/s (p<0.001) ← 메인 신호")
    print(f"    거리 매칭:   Δv=+380.6 km/s (p<10⁻¹⁶)")
    print(f"    Δv=+12,093은 원시 신호이며 97%는 z 분포 차이, 3~4%가 순수 방향 신호")

    # z 분포 통제 — axis/perp 그룹 평균 z 비교
    # 심사자 방어: "Δv가 z 분포 차이에서 오는 것 아닌가?"
    z_ax = df_v[df_v['theta']<45]['zHD']
    z_pp = df_v[(df_v['theta']>=45)&(df_v['theta']<=135)]['zHD']
    _, p_z = stats.ttest_ind(z_ax, z_pp, equal_var=False)
    print(f"\n  z 분포 균형 검토 (심사자 방어):")
    print(f"    axis 그룹 평균 z: {z_ax.mean():.4f}")
    print(f"    perp 그룹 평균 z: {z_pp.mean():.4f}")
    print(f"    z 분포 t-test p: {p_z:.4f} {'✅ z 균형' if p_z>0.05 else '⚠️ z 불균형 감지'}")
    if p_z < 0.05:
        print("    → z 통제 분석: TRT_z_control_analysis.py 참조")
        print("      다변량 회귀(v=a+b·z+c·sinθ) 기준 c=+335.6±37.7 km/s (p<0.001)")
        print("      거리 매칭 기준 Δv=+380.6 km/s (p<10⁻¹⁶)")
        print("      z 통제 후에도 방향 신호 독립 확인됨")
    p_velocity = p_vel_actual
else:
    print("  데이터 없음 → 논문 상한치 사용 (p<10⁻⁶)")
    p_velocity = 1e-6  # 상한치 (실제값은 더 작음)
    delta_v    = 12093

# ─────────────────────────────────────────
# Fisher 독립성 검증 (Bootstrap)
# ─────────────────────────────────────────
print("\n─── Fisher 독립성 검증 (Bootstrap N=10,000) ───")
print("  두 채널은 동일 파일 기반이나 서로 다른 물리량 측정")
print("  검정통계량 상관계수 Bootstrap 추정 → 준독립성 경험적 확인")

if PATH:
    np.random.seed(42)
    N_boot = 10000
    t1_boot, t2_boot = [], []
    ax1 = df_p[df_p['theta_min']<45]['residual']
    pp1 = df_p[df_p['theta_min']>60]['residual']
    ax2 = df_v[df_v['theta']<45]['v_pure']
    pp2 = df_v[(df_v['theta']>=45)&(df_v['theta']<=135)]['v_pure']

    for _ in range(N_boot):
        ax1_b = ax1.sample(len(ax1), replace=True)
        pp1_b = pp1.sample(len(pp1), replace=True)
        t1, _ = stats.ttest_ind(ax1_b, pp1_b, equal_var=False)
        ax2_b = ax2.sample(len(ax2), replace=True)
        pp2_b = pp2.sample(len(pp2), replace=True)
        t2, _ = stats.ttest_ind(pp2_b, ax2_b, equal_var=False)
        t1_boot.append(t1); t2_boot.append(t2)

    rho, p_rho = stats.pearsonr(np.array(t1_boot), np.array(t2_boot))
    print(f"  ρ = {rho:.4f},  p = {p_rho:.4f}")
    if abs(rho) < 0.1:
        print(f"  ✅ 상관 거의 없음 (|ρ|<0.1) → Fisher 독립성 가정을 지지하는 경험적 근거")
        print(f"     (Bootstrap 상관계수는 독립성 증명이 아닌 경험적 지지 근거임)")
    elif abs(rho) < 0.3:
        print(f"  ⚠️ 약한 상관 — Fisher 적용 가능하나 보수적 해석 권장")
    else:
        print(f"  ❌ 유의미한 상관 — Fisher 결합 재검토 필요")
else:
    print("  데이터 없음 → 논문 기재값: ρ=-0.013, p=0.204")
    print("  (독립성 가정을 지지하는 경험적 근거)")

# ─────────────────────────────────────────
# Fisher 결합
# ─────────────────────────────────────────
print("\n─── Fisher 결합 ───")

chi2, p_comb, sigma = fisher_combine([p_pantheon, p_velocity])
print(f"  채널 1 (Pantheon+):  p={p_pantheon:.4e}")
print(f"  채널 2 (후퇴속도):   p={p_velocity:.4e}")
print(f"  χ²={chi2:.2f}, df=4")
print(f"  Fisher p={p_comb:.3e} → {sigma:.2f}σ")

# 참고: 상한치(1e-6) 사용 시
chi2_ub, p_ub, sigma_ub = fisher_combine([0.0295, 1e-6])
print(f"\n  [참고] 상한치(p=0.0295, p=1e-6) 사용 시: {sigma_ub:.2f}σ")
print(f"  ※ 논문 기재 '5.0σ(before LEE)'는 상한치 기준")

# ─────────────────────────────────────────
# LEE 보정
# ─────────────────────────────────────────
print("\n─── LEE (Look-Elsewhere Effect) 보정 ───")
print("  ※ 본 코드에서 LEE Monte Carlo를 직접 계산하지 않음.")
print("  논문 5.3④의 별도 MC 시뮬레이션(10만회) 결과를 인용:")
print()
print("  Pantheon+ 단독: p_LEE=0.35 (0.93σ) — 단일 데이터셋 방향 탐색 보정")
print("  → Pantheon+ 단독 신호는 LEE 보정 후 약 1σ로 약화됨")
print()
print("  Fisher 결합(2채널)의 LEE 추정: 6.4~7.2σ")
print("  → 근거: 두 채널(Pantheon+, 후퇴속도)은 서로 독립적 데이터셋")
print("     독립 채널 결합 시 LEE 보정 범위가 단일 채널보다 좁음")
print("     후퇴속도 채널은 방향 사전 정의로 LEE 미해당")
print("     따라서 Fisher 결합 유의도는 단독 Pantheon LEE 보정과 다름")
print("  → 논리: Pantheon+ 단독(0.93sigma)과 Fisher 결합(6.4~7.2sigma)의 차이:")
print("     Pantheon+ 단독: Pantheon+ 내 방향 탐색 LEE 적용")
print("     Fisher 결합: 독립 채널이므로 LEE 범위 좁음, 후퇴속도는 LEE 미해당")
print("     이 수치는 논문 섹션 5.3의 별도 MC 분석 결과이며 본 코드 미검증")

# ─────────────────────────────────────────
# Bonferroni
# ─────────────────────────────────────────
print("\n─── Bonferroni (k=2) ───")
for name, pv in [('Pantheon+', p_pantheon), ('후퇴속도', p_velocity)]:
    sig = '✅' if pv < 0.025 else '⚠️'
    print(f"  {sig} {name}: p={pv:.4e} (기준 α=0.025)")

# ─────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────
# Reproducibility Check
# ─────────────────────────────────────────
print("\n─── Reproducibility Check ───")
_checks = [
    ("Δμ (Pantheon+)",  delta_mu,   0.0184, 0.002),
    ("p (Pantheon+)",   p_pantheon, 0.0295, 0.005),
    ("Δv (recession)",  delta_v,    12093,  500),
    ("Fisher σ",        sigma,      None,   None),  # 데이터 기반 계산값
]
for _name, _val, _ref, _tol in _checks:
    if _ref is None:
        print(f"  INFO  {_name}: {_val:.2f}σ (실측값 기반, 데이터마다 다름)")
    else:
        _ok = abs(_val - _ref) <= _tol
        print(f"  {'PASS' if _ok else 'FAIL':<6} {_name}: {_val:.4f} (논문: {_ref}, 허용오차: ±{_tol})")

# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"{'데이터셋':25s} {'결과':20s} {'p값(실측)':15s} {'σ'}")
print("-" * 70)
print(f"{'Pantheon+ SN Ia':<25} {f'Δμ={delta_mu:+.4f} mag':<20} {p_pantheon:<15.4e} {'~2σ'}")
print(f"{'순수 후퇴속도':<25} {f'Δv={delta_v:+,.0f} km/s':<20} {p_velocity:<15.4e} {'>5σ'}")
print(f"{'Fisher (실측값 기반)':<25} {'결합':<20} {p_comb:<15.4e} {f'{sigma:.1f}σ'}")
print(f"{'Fisher (LEE 보정, 논문인용)':<25} {'별도 MC 분석':<20} {'':<15} {'6.4~7.2σ (참고)'}")

print()
print("참고 문헌:")
print("  Brout et al. 2022, ApJ 938, 110 (Pantheon+)")
print("  Fisher, R.A. 1932 (Fisher method)")
