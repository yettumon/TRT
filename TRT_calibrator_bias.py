"""
TRT — SH0ES 캘리브레이터 방향 편중 분석
==========================================
논문: Tilted Ruler Theory (TRT)
섹션: 4.1절 | GunSik Kim (Yettumon) 2026

핵심 결과:
  캘리브레이터(N=77) θ>90° 비율: 72.7%
  우주론 샘플(N=1,624) θ>90° 비율: 33.7%
  평균 각도: 104.4°
  이항검정 (귀무=50%):   p = 4.11×10⁻⁵  (3.9σ)
  이항검정 (귀무=33.7%): p = 3.09×10⁻¹² (6.9σ)

분석 기준:
  theta (단방향, 0~180°): 주축(RA=330°) 기준 각거리
  θ>90°: 방사 분력 vᵣ·sin(θ) 최대 방향 (f(θ)<1 구간)
  IS_CALIBRATOR==1: 캘리브레이터 N=77
  IS_CALIBRATOR==0: 우주론 샘플 N=1,624

재현 가능 여부:
  본 스크립트와 동일한 Pantheon+SH0ES 데이터 파일(MD5 명시)을
  사용할 경우 논문 수치의 독립 재현이 가능함.

예상 재현값 (논문 기재):
  N_cal≈77, pct_cal≈72.7%, pct_cos≈33.7%, mean_ang≈104.4°
  p_50≈4.11×10⁻⁵, p_337≈3.09×10⁻¹²
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom
import hashlib, platform, os, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TRT: SH0ES 캘리브레이터 방향 편중 분석")
print("GunSik Kim (Yettumon) 2026")
print("=" * 60)

# ─────────────────────────────────────────
# 환경 정보
# ─────────────────────────────────────────
import scipy as _scipy
print(f"\nPython {platform.python_version()} | "
      f"numpy {np.__version__} | "
      f"scipy {_scipy.__version__}")

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
AXIS_RA,  AXIS_DEC  = 330.0, 0.0   # 사전 정의된 물리적 기준 (논문 2.3절, 본 분석 이전 독립 도출)
ANTI_RA,  ANTI_DEC  = 150.0, 0.0

# ─────────────────────────────────────────
# 경로 자동 탐색
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
if PATH is None:
    raise FileNotFoundError("Pantheon+SH0ES.dat 없음.\n다운로드: https://pantheonplussh0es.github.io/")

print(f"\n데이터 파일: {PATH}")
with open(PATH, 'rb') as f:
    _blob = f.read()
print(f"MD5    = {hashlib.md5(_blob).hexdigest()}")
print(f"SHA256 = {hashlib.sha256(_blob).hexdigest()}")

# ─────────────────────────────────────────
# 데이터 로드 및 컬럼 검증
# ─────────────────────────────────────────
data = pd.read_csv(PATH, sep=r'\s+', comment='#')
print(f"전체 N = {len(data)}")

_required = ['RA', 'DEC', 'zHD', 'MU_SH0ES', 'IS_CALIBRATOR']
_missing = [c for c in _required if c not in data.columns]
if _missing:
    raise ValueError(f"Missing columns: {_missing}")
print(f"필수 컬럼: ✅")

# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────
def angle_from_axis(ra, dec, ax_ra=AXIS_RA, ax_dec=AXIS_DEC):
    """단방향 각거리 (0~180°) — theta 기준"""
    ra_r  = np.radians(ra);   dec_r = np.radians(dec)
    ax_r  = np.radians(ax_ra); ax_d = np.radians(ax_dec)
    gx = np.cos(dec_r)*np.cos(ra_r)
    gy = np.cos(dec_r)*np.sin(ra_r)
    gz = np.sin(dec_r)
    ax = np.cos(ax_d)*np.cos(ax_r)
    ay = np.cos(ax_d)*np.sin(ax_r)
    az = np.sin(ax_d)
    dot = np.clip(gx*ax + gy*ay + gz*az, -1, 1)
    return np.degrees(np.arccos(dot))

def f_theta(theta_deg, A1=0.004, A2=0.007):
    """f(θ) 보정계수 (z≈0이므로 exp 항 생략)
    주의: θ>90°에서는 f(θ)<1 (f>1은 θ<90° 및 θ>150° 구간)
    f(θ)>1 비율은 참고용이며 핵심 지표는 θ>90° 비율임"""
    t = np.radians(theta_deg)
    return 1 + A1*np.cos(t) + A2*(3*np.cos(t)**2 - 1)/2

# ─────────────────────────────────────────
# 샘플 분리
# ─────────────────────────────────────────
cal = data[data['IS_CALIBRATOR'] == 1].copy()   # N=77
cos = data[data['IS_CALIBRATOR'] == 0].copy()   # N=1,624

print(f"\n캘리브레이터 N={len(cal)}, 우주론 샘플 N={len(cos)}")

# theta 계산 (단방향, 0~180°)
cal['theta'] = angle_from_axis(cal['RA'].values, cal['DEC'].values)
cos['theta'] = angle_from_axis(cos['RA'].values, cos['DEC'].values)

# ─────────────────────────────────────────
# 핵심 분석: θ>90° 방향 비율
# ─────────────────────────────────────────
print("\n─── θ>90° 방향 비율 (핵심 지표) ───")
print("θ>90°: GCOD 주축(RA=330°)으로부터 90° 이상 = 방사 분력 최대 방향")

pct_cal  = (cal['theta'] > 90).mean() * 100
pct_cos  = (cos['theta'] > 90).mean() * 100
mean_ang = cal['theta'].mean()
N_cal    = len(cal)
k_cal    = int((cal['theta'] > 90).sum())

print(f"\n캘리브레이터: {k_cal}/{N_cal} = {pct_cal:.1f}%  (θ>90°)")
print(f"우주론 샘플:  {(cos['theta']>90).sum()}/{len(cos)} = {pct_cos:.1f}%  (θ>90°)")
print(f"캘리브레이터 평균 θ: {mean_ang:.1f}°")
print(f"비율 차이: +{pct_cal-pct_cos:.1f}%p ({pct_cal/pct_cos:.1f}배)")

# ─────────────────────────────────────────
# 이항검정
# ─────────────────────────────────────────
print("\n─── 이항검정 ───")

# 귀무=50% (보수적)
p_50  = 1 - binom.cdf(k_cal - 1, N_cal, 0.5)
s_50  = stats.norm.isf(p_50)

# 귀무=33.7% (물리적: 우주론 샘플 실측값)
null_337 = pct_cos / 100
p_337 = 1 - binom.cdf(k_cal - 1, N_cal, null_337)
s_337 = stats.norm.isf(p_337)

print(f"\n귀무=50%  (보수적): p={p_50:.4e}  ({s_50:.2f}σ)")
print(f"귀무=33.7%(물리적): p={p_337:.4e}  ({s_337:.2f}σ)")
print(f"\n논문 기재: p=4.11×10⁻⁵ (3.9σ) / p=3.09×10⁻¹² (6.9σ)")

# Monte Carlo 교차검증
np.random.seed(42)
N_mc = 100_000
mc_50  = np.mean(np.random.binomial(N_cal, 0.5,      N_mc) >= k_cal)
mc_337 = np.mean(np.random.binomial(N_cal, null_337, N_mc) >= k_cal)
print(f"\nMonte Carlo 교차검증 (N={N_mc:,}):")
print(f"  귀무=50%:    p={mc_50:.6f}  {'✅' if abs(mc_50-p_50)<0.001 else '⚠️'}")
print(f"  귀무=33.7%:  p={mc_337:.6f}  ※ 참고만 (N=10만회로는 p~10⁻¹² 수준 검증 불가)")
print(f"  ※ MC는 귀무=50% 교차검증 전용. 귀무=33.7%는 이론 이항분포 계산값(위 p_337)이 정확한 값.")

# ─────────────────────────────────────────
# 참고: f(θ)>1 비율
# ─────────────────────────────────────────
print("\n─── f(θ)>1 비율 (참고용) ───")
print("주의: θ>90°에서는 f(θ)<1. 핵심 지표는 θ>90° 비율임.")
f_c = f_theta(cal['theta'].values)
f_k = f_theta(cos['theta'].values)
print(f"캘리브레이터 f>1 비율: {np.mean(f_c>1)*100:.1f}%")
print(f"우주론 샘플  f>1 비율: {np.mean(f_k>1)*100:.1f}%")

# ─────────────────────────────────────────
# 관측 선택 효과 검토
# ─────────────────────────────────────────
print("\n─── 관측 선택 효과 검토 ───")
t_z, p_z = stats.ttest_ind(cal['zHD'].values, cos['zHD'].values)
print(f"z 분포 비교 (Malmquist bias 검토):")
print(f"  캘리브레이터 z 평균: {cal['zHD'].mean():.4f}")
print(f"  우주론 샘플  z 평균: {cos['zHD'].mean():.4f}")
print(f"  t-test p: {p_z:.4e}")
if p_z < 0.05:
    print(f"  ⚠️ z 분포 차이 유의 — 캘리브레이터는 z<0.01에 집중됨 (정상)")
    print(f"  → 캘리브레이터는 본질적으로 저z 샘플 (z̄<0.01, 코세페이드 측정 한계)")
else:
    print(f"  ✅ z 분포 유사")

print(f"\n은하면 오염 검토:")
try:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _c = SkyCoord(ra=330*u.deg, dec=0*u.deg, frame='icrs')
    _g = _c.galactic
    _b = abs(_g.b.deg)
    print(f"  LA-axis 은하좌표: l={_g.l.deg:.1f}°, b={_g.b.deg:.1f}°")
    print(f"  |b|={_b:.1f}° → 은하면(|b|<10°) 기준 {'오염 범위 외 ✅' if _b > 10 else '⚠️ 근접, 추가 검토 필요'}")
    print(f"  (은하면과 직접 일치하지 않는 방향; 세부 정량 오염 분석은 별도 수행 필요)")
except ImportError:
    print(f"  LA-axis(Dec=0°)는 은하면과 직접 일치하지 않는 방향 (정량 검증은 별도 분석 필요)")

# ─────────────────────────────────────────
# Reproducibility Check
# ─────────────────────────────────────────
print("\n─── Reproducibility Check ───")
_checks = [
    ("N_cal",     len(cal),          77,    0),
    ("k_cal",     k_cal,             56,    1),  # 허용오차 1: 데이터 버전 차이 대응
    ("pct_cal",   round(pct_cal,1),  72.7,  0.5),
    ("pct_cos",   round(pct_cos,1),  33.7,  1.0),
    ("mean_ang",  round(mean_ang,1), 104.4, 2.0),
    ("p_50",      round(p_50,6),     4.11e-5, 0.5e-5),
]
for _name, _val, _ref, _tol in _checks:
    _ok = (_val == _ref) if _tol == 0 else (abs(_val - _ref) <= _tol)
    print(f"  {'PASS' if _ok else 'FAIL':<6} {_name}: {_val} (논문: {_ref}, 허용오차: {_tol})")

# ─────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"캘리브레이터: {k_cal}/{N_cal} = {pct_cal:.1f}%  (θ>90°)")
print(f"우주론 샘플:  {pct_cos:.1f}%  (θ>90°)")
print(f"평균 θ:        {mean_ang:.1f}°")
print(f"이항검정 귀무=50%:   p={p_50:.4e}  ({s_50:.1f}σ)")
print(f"이항검정 귀무=33.7%: p={p_337:.4e}  ({s_337:.1f}σ)")
print()
print("참고 문헌:")
print("  Brout et al. 2022, ApJ 938, 110 (Pantheon+)")
print("  Riess et al. 2022, ApJL 934, L7 (SH0ES)")
print("  데이터: https://pantheonplussh0es.github.io/")
