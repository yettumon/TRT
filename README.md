# TRT (Tilted Ruler Theory) — Analysis Code

**논문**: Tilted Ruler Theory (TRT): SH0ES 캘리브레이터의 방향 편향과 허블 텐션  
**저자**: Kim, Gun.Sik (Yettumon) | 독립 연구자  
**이메일**: nearcop3@gmail.com  
**날짜**: 2026.06  
**Zenodo DOI**: https://doi.org/10.5281/zenodo.20572072  
**GitHub**: https://github.com/yettumon/TRT  

---

## 개요

TRT(Tilted Ruler Theory)는 SH0ES 캘리브레이터 샘플의 방향 편향이 허블 상수를 과대 추정하게 만든다는 이론이다. 두 가지 독립 메커니즘을 Pantheon+SH0ES 공개 데이터로 정량 검증한다.

- **경로 1**: f(θ) 광도 감쇄 → 캘리브레이터 방향 편중 → H₀ 과대 추정
- **경로 2**: 방사 분력 vᵣ·sin(θ) → GCOD 주축 실재의 독립 증거

**분석 기준축**: RA=330°, Dec=0° (GCOD 주축, LA-axis)  
본 분석 이전 독립 도출 — Dipole Repeller(337.5°) + CMB dipole 반대방향(337.7°) 정합.

---

## 코드 파일 및 역할

| 파일 | 섹션 | 역할 | 상태 |
|------|------|------|------|
| `TRT_calibrator_bias.py` | 4.1절 | 캘리브레이터 θ>90° 편중 정량화 (이항검정) | ✅ |
| `TRT_hubble_residual.py` | 4.2절 | Pantheon+ 허블 잔차 방향별 분석 | ✅ |
| `TRT_recession_velocity.py` | 4.3절 | 순수 후퇴속도 방향별 분석 (보조 코드) | ✅ |
| `TRT_z_control_analysis.py` | 4.3절 | z 통제 정밀 분석 (메인 코드) | ✅ |
| `TRT_fisher_statistics.py` | 4.7절 | Fisher 결합 통계 (실측값 기반) | ✅ |

### 코드 역할 구조

```
TRT_recession_velocity.py  ← "방향별 속도 차이 존재" 확인 (보조)
        ↓
TRT_z_control_analysis.py  ← "z 분포 효과 제거 후 순수 방향 신호" 증명 (메인)
        ↓
TRT_fisher_statistics.py   ← Pantheon+ + 후퇴속도 2채널 Fisher 결합
```

---

## 핵심 결과

| 검증 항목 | 결과 | 유의도 |
|-----------|------|--------|
| 캘리브레이터 θ>90° 편중 | 72.7% vs 우주론 33.7% | p=4.11×10⁻⁵ (귀무=50%) |
| 캘리브레이터 θ>90° 편중 | 72.7% vs 우주론 33.7% | p=3.09×10⁻¹² (귀무=33.7%) |
| Pantheon+ 허블 잔차 | Δμ = +0.0184 mag | p = 0.0295 |
| 순수 후퇴속도 원시 신호 | Δv = +12,093 km/s | p < 10⁻⁶ |
| z 통제 후 방향 신호 (회귀) | c = +335.6 ± 37.7 km/s | p < 0.001 |
| z 통제 후 방향 신호 (매칭) | Δv = +380.6 km/s | p < 10⁻¹⁶ |
| Fisher 결합 (2채널, before LEE) | ~5σ | 실측값 기반 |
| Fisher 결합 (2채널, after LEE) | 6.4 ~ 7.2σ | 별도 MC 분석 |

> **Note**: Δv=+12,093 km/s의 97%는 z 분포 차이에서 기인. 순수 방향 신호는 z 통제 후 +335~381 km/s. 상세 분해는 `TRT_z_control_analysis.py` 참조.

> **Note**: Fisher 6.4~7.2σ는 LEE(Look-Elsewhere Effect) Monte Carlo 10만회 보정 후 수치. DESI BGS는 데이터 무결성 문제로 Fisher 결합에서 제외, 정성적 참고만 사용.

---

## 필요 데이터

```
Pantheon+SH0ES.dat
출처: https://github.com/PantheonPlusSH0ES/DataRelease
      https://pantheonplussh0es.github.io/
```

**재현성 확인**: 각 스크립트 실행 시 MD5 및 SHA256 해시가 자동 출력됩니다.  
동일 논문 결과를 재현하려면 **동일한 해시값의 파일**을 사용해야 합니다.

> 논문에서 사용한 Pantheon+SH0ES.dat MD5: 각 스크립트 실행 시 자동 출력되는 값으로 확인

---

## 실행 방법

### 환경 설정

```bash
pip install numpy scipy astropy matplotlib pandas
```

### 실행 순서 (권장)

```bash
# 1. 캘리브레이터 편중 분석
python TRT_calibrator_bias.py

# 2. 허블 잔차 방향 분석
python TRT_hubble_residual.py

# 3. 순수 후퇴속도 분석 (보조)
python TRT_recession_velocity.py

# 4. z 통제 정밀 분석 (메인)
python TRT_z_control_analysis.py

# 5. Fisher 결합 통계 (데이터 필요)
python TRT_fisher_statistics.py
```

### 재현 확인

각 스크립트는 실행 시 다음을 자동 출력합니다:
- Python / NumPy / SciPy 버전
- 데이터 파일 MD5 및 SHA256
- 논문 수치와의 차이값 또는 PASS/FAIL

---

## 통계 방법론 요약

| 분석 | 방법 | 비고 |
|------|------|------|
| 방향 편중 검정 | 이항검정 (귀무=50%, 33.7%) | Monte Carlo 교차검증 포함 |
| 허블 잔차 비교 | Welch t-test | 등분산 가정 없음 |
| 후퇴속도 비교 | Welch t-test + Mann-Whitney U | 모수/비모수 병행 |
| z 통제 (회귀) | OLS 다변량 회귀 v=a+b·z+c·sin(θ) | sin(θ): TRT 물리적 근거 |
| z 통제 (매칭) | Greedy nearest-neighbor \|Δz\|<0.01 | Wilcoxon + Mann-Whitney 검증 |
| Fisher 결합 | Fisher χ² method (2채널) | LEE 보정 별도 MC |

---

## 분석 기준축 근거

GCOD 주축 RA=330°, Dec=0°는 **본 Pantheon+ 분석 이전에 독립적으로 도출**됨:

1. Dipole Repeller (Hoffman et al. 2017): RA=337.5°
2. CMB dipole 반대방향 (Kogut et al. 1993): RA=337.7°
3. 물리적 정의: 방사 분력 vᵣ·sin(θ)가 최소인 방향

세 독립 기준의 정합으로 RA=330° 확정. 사후적 축 선택(post-hoc) 아님.

---

## 참고 문헌

- Brout, D., et al. 2022, ApJ, 938, 110 (Pantheon+)
- Riess, A.G., et al. 2022, ApJL, 934, L7 (SH0ES)
- Hoffman, Y., et al. 2017, Nature Astronomy, 1, 0036 (Dipole Repeller)
- Kogut, A., et al. 1993, ApJ, 419, 1 (CMB dipole)
- Planck Collaboration 2018, A&A, 641, A6
- Colin, J., et al. 2019, A&A, 631, L13
- Bengaly & Alcaniz 2024, MNRAS, 530, 4667

---

## 라이선스

CC BY 4.0 — Kim, Gun.Sik (Yettumon), 2026
