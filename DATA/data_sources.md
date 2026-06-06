# Data Sources

This directory is intentionally empty. The analysis requires the following
publicly available dataset. Due to licensing, data files are not included
in this repository.

---

## Pantheon+SH0ES

**Used in:** TRT_hubble_residual.py, TRT_calibrator_bias.py,
             TRT_recession_velocity.py, TRT_z_control_analysis.py  
**File needed:** `Pantheon+SH0ES.dat`  
**Download:** https://pantheonplussh0es.github.io/  
**Reference:** Brout et al. 2022, ApJ 938, 110 / Scolnic et al. 2022, ApJ 938, 113

Place the downloaded file in the same directory as the scripts, or in `data/`.

```
data/
└── Pantheon+SH0ES.dat   ← place here
```

The scripts will automatically search the following paths:
- `Pantheon+SH0ES.dat` (same directory)
- `data/Pantheon+SH0ES.dat`
- `/content/drive/MyDrive/Colab Notebooks/Pantheon+SH0ES.dat` (Google Colab)

---

## Notes

- TRT_fisher_statistics.py does not require any data file (uses p-values directly).
- All data files are publicly available at no cost.
- No proprietary or restricted data were used in this analysis.
- For reproducibility, the exact version used is cited in the paper.
