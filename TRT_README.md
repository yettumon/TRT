# TRT (Tilted Ruler Theory) 분석 코드

**논문**: Tilted Ruler Theory (TRT): SH0ES 캘리브레이터의 방향 편향과 허블 텐션  
**저자**: Kim, Gun.Sik (Yettumon) | 독립 연구자  
**이메일**: nearcop3@gmail.com  
**날짜**: 2026.06  
**Zenodo DOI**: (업로드 후 기재)

---

## 개요

TRT는 SH0ES 캘리브레이터 샘플의 방향 편향이 잣대를 기울게 만들어
허블 상수를 과대 추정하게 한다는 이론이다.
두 가지 독립 메커니즘(경로 1: f(θ) 광도 감쇄, 경로 2: 방사 분력)을
Pantheon+SH0ES 공개 데이터로 정량적으로 검증한다.

---

## 코드 파일

| 파일 | 섹션 | 내용 | 검증 |
|------|------|------|------|
| TRT_calibrator_bias.py | 4.1항 | 캘리브레이터 방향 편중 정량화 | ✅ 재현 완료 |
| TRT_hubble_residual.py | 4.2항 | Pantheon+ 허블 잔차 방향별 분석 | ✅ 재현 완료 |
| TRT_recession_velocity.py | 4.3항 | 순수 후퇴속도 방향별 분석 | ✅ 재현 완료 |
| TRT_z_control_analysis.py | 4.3항 | z 통제 정밀 분석 | ✅ 재현 완료 |
| TRT_fisher_statistics.py | 4.7항 | Fisher 결합 통계 | ✅ 재현 완료 |

---

## 필요 데이터

```
Pantheon+SH0ES.dat
출처: https://github.com/PantheonPlusSH0ES/DataRelease
```

---

## 실행 환경

```bash
pip install numpy scipy astropy matplotlib pandas
python TRT_calibrator_bias.py
```

---

## 핵심 결과

| 검증 항목 | 결과 | 유의도 |
|-----------|------|--------|
| 캘리브레이터 방향 편중 | θ>90° 방향 72.7% | p<10⁻¹⁰ |
| Hubble 잔차 방향 의존성 | Δμ=+0.0184 mag | p=0.0295 |
| 순수 후퇴속도 비등방성 | Δv=+12,093 km/s | p<10⁻⁶ |
| Fisher 결합 통계 | 6.4~7.2σ | LEE 보정 후 |

---

## 인용

Kim, G.S. (Yettumon), 2026. Tilted Ruler Theory (TRT).
Zenodo. DOI: (업로드 후 기재)
