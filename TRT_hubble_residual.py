"""
TRT — Pantheon+ Hubble Residual Directional Analysis
=====================================================
논문: Tilted Ruler Theory (TRT)
섹션: 4.2절 | GunSik Kim (Yettumon) 2026

핵심 결과:
  전체 샘플 (IS_CALIBRATOR 구분 없음, z>0.01, isfinite):
  주축 그룹 (θ_min<45°, N=603): μ_residual 평균
  수직 그룹 (θ_min>60°, N=709): μ_residual 평균
  Δμ (axis − perp) = +0.0184 mag, p = 0.0295

분석 기준:
  데이터: Pantheon+SH0ES 전체 N=1,701
  필터: zHD>0.01, zHD<1.5, isfinite(MU_SH0ES)
  IS_CALIBRATOR 구분 없음 (캘리브레이터 포함)
  거리 모듈: FlatLambdaCDM(H0=70, Om0=0.3)
  잔차: MU_SH0ES - mu_theory

재현 가능 여부:
  본 스크립트와 동일한 Pantheon+SH0ES 데이터 파일(MD5 명시)을
  사용할 경우 논문 수치의 독립 재현이 가능함.
  검증 환경: astropy >= 6.0 (validated with astropy 7.x)
  FlatLambdaCDM.distmod() 결과는 astropy 버전에 따라 미세 차이 가능.

예상 재현값 (논문 기재):
  N_ax≈603, N_mid≈271, N_pp≈709
  Δμ≈+0.0184 mag, p≈0.0295
  (실제 실행 로그 첨부 시 "확인" 표현 사용 가능)
  

필요 데이터:
  Pantheon+SH0ES.dat (공개 데이터)
  https://pantheonplussh0es.github.io/
"""

import numpy as np
import pandas as pd
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import warnings, os
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRT: Pantheon+ Hubble Residual Directional Analysis")
print("GunSik Kim (Yettumon) 2026")
print("=" * 60)

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
AXIS_RA,  AXIS_DEC  = 330.0, 0.0
# Fixed a priori axis from TRT Section 2.3.
# Determined independently before Pantheon+ directional analysis.
# (Dipole Repeller 337.5° + CMB dipole antipode 337.7° → confirmed RA=330°)
ANTI_RA,  ANTI_DEC  = 150.0, 0.0   # LA-axis 사자자리 끝
AXIS_CUT = 45.0   # 주축 그룹: θ_min < 45°
PERP_CUT = 60.0   # 수직 그룹: θ_min > 60°
Z_MIN    = 0.01
Z_MAX    = 1.5
H0, Om0  = 70.0, 0.3   # 논문 기준 ΛCDM

# ─────────────────────────────────────────
# 경로 자동 탐색
# ─────────────────────────────────────────
PATHS = [
    'Pantheon+SH0ES.dat',
    'Pantheon_SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Colab Notebooks/Pantheon_SH0ES.dat',
    '/content/drive/MyDrive/Pantheon+SH0ES.dat',
    '/content/drive/MyDrive/Pantheon_SH0ES.dat',
]
PATH = None
for p in PATHS:
    if os.path.exists(p):
        PATH = p
        break
if PATH is None:
    raise FileNotFoundError("Pantheon+SH0ES.dat 없음.\n다운로드: https://pantheonplussh0es.github.io/")

print(f"\n데이터 파일: {PATH}")

# ④ 환경 정보
import platform, hashlib
import scipy as _scipy
import astropy as _astropy
print(f"Python {platform.python_version()} | "
      f"numpy {np.__version__} | pandas {pd.__version__} | "
      f"scipy {_scipy.__version__} | astropy {_astropy.__version__}")

data = pd.read_csv(PATH, sep=r'\s+', comment='#')
print(f"전체 N = {len(data)}")

# ② 파일 버전 식별 (MD5)
with open(PATH, 'rb') as _f:
    _blob = _f.read()
_md5    = hashlib.md5(_blob).hexdigest()
_sha256 = hashlib.sha256(_blob).hexdigest()
print(f"MD5    = {_md5}")
print(f"SHA256 = {_sha256}")

# ③ 필수 컬럼 검증
_required = ['RA', 'DEC', 'zHD', 'MU_SH0ES', 'IS_CALIBRATOR']
_missing = [c for c in _required if c not in data.columns]
if _missing:
    raise ValueError(f"Missing columns: {_missing}")
print(f"필수 컬럼: ✅")
print(f"전체 컬럼: {data.columns.tolist()}")
print(f"  IS_CALIBRATOR==1 (캘리브레이터): {(data['IS_CALIBRATOR']==1).sum()}")
print(f"  IS_CALIBRATOR==0 (우주론 샘플):  {(data['IS_CALIBRATOR']==0).sum()}")

# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────
# θ 정의: great-circle angular separation (구면 삼각법)
# 단위: degrees (0~180°)
# theta_min = min(θ_axis, θ_anti): 주축/반주축 중 가까운 쪽 (0~90°)
def angular_sep(ra1, dec1, ra2=AXIS_RA, dec2=AXIS_DEC):
    r = np.radians
    a = (np.sin(r((dec2-dec1)/2))**2 +
         np.cos(r(dec1))*np.cos(r(dec2))*np.sin(r((ra2-ra1)/2))**2)
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(a, 0, 1))))

# ─────────────────────────────────────────
# 전처리
# 논문 재현 조건:
#   - IS_CALIBRATOR 구분 없이 전체 사용
#   - zHD>0.01, zHD<1.5
#   - MU_SH0ES isfinite
#   → N=603(주축), N=709(수직) 재현
# ─────────────────────────────────────────
print("\n─── 전처리 ───")
print(f"필터: zHD>{Z_MIN}, zHD<{Z_MAX}, isfinite(MU_SH0ES)")
print(f"IS_CALIBRATOR 구분 없음 (전체 포함)")

cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

df = data[(data['zHD'] > Z_MIN) &
          (data['zHD'] < Z_MAX) &
          np.isfinite(data['MU_SH0ES'])].copy()

df['mu_theory'] = cosmo.distmod(df['zHD'].values).value
df['residual']  = df['MU_SH0ES'] - df['mu_theory']

# theta: 단방향 (주축 방향, 0~180°) — 캘리브레이터 편중 검정용
# theta_min: 양방향 min(주축, 반주축, 0~90°) — 그룹 분류용
df['theta']      = angular_sep(df['RA'].values, df['DEC'].values)
df['theta_anti'] = angular_sep(df['RA'].values, df['DEC'].values, ANTI_RA, ANTI_DEC)
df['theta_min']  = np.minimum(df['theta'], df['theta_anti'])

print(f"필터 후 N = {len(df)}")

# ─────────────────────────────────────────
# 그룹 분류
# ─────────────────────────────────────────
ax  = df[df['theta_min'] < AXIS_CUT]
pp  = df[df['theta_min'] > PERP_CUT]
mid = df[(df['theta_min'] >= AXIS_CUT) & (df['theta_min'] <= PERP_CUT)]

print(f"주축 그룹 (θ_min<{AXIS_CUT}°): N={len(ax)}")
print(f"중간 구간 ({AXIS_CUT}~{PERP_CUT}°):       N={len(mid)} [제외]")
print(f"수직 그룹 (θ_min>{PERP_CUT}°): N={len(pp)}")
print(f"논문 기재: N_ax=603, N_mid=271, N_pp=709")

# ─────────────────────────────────────────
# 주요 결과: Hubble 잔차 방향별 비교
# ─────────────────────────────────────────
print("\n─── Hubble 잔차 방향별 비교 ───")
print(f"잔차 = MU_SH0ES - μ_ΛCDM(z; H0={H0}, Om0={Om0})")

dmu = ax['residual'].mean() - pp['residual'].mean()
t, p_val = stats.ttest_ind(ax['residual'], pp['residual'],
                           equal_var=False, nan_policy='omit')  # Welch t-test

print(f"\n주축 평균 잔차: {ax['residual'].mean():+.4f} mag")
print(f"수직 평균 잔차: {pp['residual'].mean():+.4f} mag")
print(f"Δμ (axis−perp): {dmu:+.4f} mag")
print(f"t통계: {t:.3f},  p = {p_val:.4f}")

print(f"\n─── 논문 기재값 비교 ───")
print(f"논문: Δμ=+0.0184 mag, p=0.0295")
print(f"재현: {'✅ MATCH' if abs(dmu-0.0184)<0.002 else '❌ CHECK'}: Δμ={dmu:+.4f}")
print(f"재현: {'✅ MATCH' if abs(p_val-0.0295)<=0.005 else '❌ CHECK'}: p={p_val:.4f} (기준: 0.0295±0.005)")

# ─────────────────────────────────────────
# 캘리브레이터 방향 편중 검정
# 별도 분석: IS_CALIBRATOR 기준으로 재분류
# theta (단방향, 0~180°) 사용 — theta_min과 구분
# ─────────────────────────────────────────
print("\n─── 캘리브레이터 방향 편중 검정 ───")
print("(IS_CALIBRATOR==1 vs ==0 별도 비교)")
print("기준: theta(단방향, 0~180°) > 90° = GCOD 주축으로부터 90° 이상")

# 캘리브레이터: z 필터 없이 전체 77개 사용 (대부분 z<0.01)
# 우주론 샘플: df 내 IS_CALIBRATOR==0
cal = data[data['IS_CALIBRATOR'] == 1].copy()   # 전체 캘리브레이터 N=77
cos = df[df['IS_CALIBRATOR'] == 0].copy()        # 우주론 샘플 (z>0.01 필터 적용)

cal['theta'] = angular_sep(cal['RA'].values, cal['DEC'].values)

# theta (단방향) 기준 — theta_min 아님
pct_cal = (cal['theta'] > 90).mean() * 100
pct_cos = (cos['theta'] > 90).mean() * 100
mean_ang_cal = cal['theta'].mean()

print(f"\n캘리브레이터 N={len(cal)}, θ>90°: {pct_cal:.1f}%")
print(f"우주론 샘플  N={len(cos)}, θ>90°: {pct_cos:.1f}%")
print(f"캘리브레이터 평균 θ: {mean_ang_cal:.1f}°")

print(f"\n논문: 캘리브레이터 72.7%, 우주론 33.7%, 평균 104.4°")
print(f"{'✅ MATCH' if abs(pct_cal-72.7)<1.0 else '❌ CHECK'}: pct_cal={pct_cal:.1f}%")
print(f"{'✅ MATCH' if abs(pct_cos-33.7)<1.0 else '❌ CHECK'}: pct_cos={pct_cos:.1f}%")
print(f"{'✅ MATCH' if abs(mean_ang_cal-104.4)<2.0 else '❌ CHECK'}: mean_ang={mean_ang_cal:.1f}°")

# ─────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────
print("\n─── Reproducibility Check ───")
# (name, 실측값, 논문기재값, 허용오차) — 허용오차=0은 정확 일치 필요
_checks = [
    ("N_axis",  len(ax),             603,    0),
    ("N_mid",   len(mid),            271,    0),
    ("N_perp",  len(pp),             709,    0),
    ("delta_mu", round(dmu,4),       0.0184, 0.002),
    ("p_value",  round(p_val,4),     0.0295, 0.005),
    ("pct_cal",  round(pct_cal,1),   72.7,   1.0),
    ("mean_ang", round(mean_ang_cal,1), 104.4, 2.0),
]
for _name, _val, _ref, _tol in _checks:
    _ok = (_val == _ref) if _tol == 0 else (abs(_val - _ref) <= _tol)
    print(f"  {'PASS' if _ok else 'FAIL':<6} {_name}: {_val} (논문: {_ref}, 허용오차: {_tol})")

print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"전체 분석 샘플: N={len(df)} (z>{Z_MIN}, isfinite, IS_CAL 무관)")
print(f"주축 그룹: N={len(ax)},  수직 그룹: N={len(pp)}")
print(f"Δμ = {dmu:+.4f} mag,  p = {p_val:.4f}")
print(f"캘리브레이터 θ>90°: {pct_cal:.1f}% (우주론: {pct_cos:.1f}%)")
print()
print("참고 문헌:")
print("  Brout et al. 2022, ApJ 938, 110 (Pantheon+)")
print("  Riess et al. 2022, ApJL 934, L7 (SH0ES)")
print("  데이터: https://pantheonplussh0es.github.io/")
