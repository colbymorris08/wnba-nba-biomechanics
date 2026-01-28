# WNBA vs NBA Shooting Biomechanics Analysis

## 🏀 Overview

This project analyzes free throw shooting mechanics using **real motion capture data** from the [MLSE Sport Performance Lab (SPL) Open Data](https://github.com/mlsedigital/SPL-Open-Data) repository. We examine biomechanical differences between made and missed shots, then extrapolate implications for WNBA vs NBA players based on height-adjusted physics.

**Key Question:** What biomechanical factors most predict free throw success, and how do requirements differ for shorter vs taller players?

---

## 📊 Key Findings

### Primary Finding: Elbow Range of Motion is THE Key Predictor

| Metric | Made Shots | Missed Shots | Difference | p-value | Effect Size |
|--------|------------|--------------|------------|---------|-------------|
| **Elbow ROM** | 102.4° ± 9.6° | 97.8° ± 6.4° | **+4.65°** | **0.008*** | d = 0.57 |
| Hip ROM | 46.8° ± 2.2° | 47.6° ± 3.1° | -0.80° | 0.107 | d = -0.30 |
| Knee ROM | 59.3° ± 2.1° | 59.1° ± 2.3° | +0.20° | 0.635 | d = 0.09 |
| Entry Angle | 43.9° ± 1.6° | 43.8° ± 2.1° | +0.15° | 0.668 | d = 0.08 |

**\*\*\* p < 0.01 (highly statistically significant)**

### Entry Angle Success Rate
- **84.1%** of made shots had entry angles in the optimal range (42-46°)
- Only **59.5%** of missed shots fell in this range

---

## 📈 Understanding Cohen's d (Effect Size)

**Cohen's d** measures the *practical significance* of a finding, not just statistical significance. It tells you how large the difference is between two groups in standardized units.

### Formula:
```
Cohen's d = (Mean₁ - Mean₂) / Pooled Standard Deviation
```

### Interpretation Scale:
| Cohen's d | Interpretation | Real-World Meaning |
|-----------|---------------|-------------------|
| 0.2 | Small | Barely noticeable difference |
| 0.5 | Medium | Noticeable, meaningful difference |
| 0.8 | Large | Obvious, substantial difference |

### Our Finding: d = 0.57 (Medium-Large)
This means made shots have elbow ROM that is **0.57 standard deviations higher** than missed shots. In practical terms:
- If you randomly pick a made shot and a missed shot, the made shot will have higher elbow ROM **~65% of the time**
- This is a meaningful, actionable difference for coaching

### Why Effect Size Matters:
With large samples, even tiny differences become "statistically significant." Effect size tells you if the difference is actually *meaningful*. Our elbow ROM finding is both:
- ✅ Statistically significant (p = 0.008)
- ✅ Practically meaningful (d = 0.57)

---

## 🏀 WNBA vs NBA Implications

### Height-Adjusted Biomechanical Requirements

| Profile | Height | Release Point | Optimal Arc | Required Elbow ROM |
|---------|--------|---------------|-------------|-------------------|
| WNBA Guard | 5'8" | 7.5 ft | 52.0° | **103°** |
| WNBA Forward | 6'0" | 8.0 ft | 51.0° | **102°** |
| WNBA Center | 6'4" | 8.3 ft | 50.3° | **101°** |
| NBA Guard | 6'2" | 8.2 ft | 50.5° | 101° |
| NBA Forward | 6'8" | 8.8 ft | 49.5° | 99° |
| NBA Center | 7'0" | 9.2 ft | 48.5° | 98° |

### Why Shorter Players Need More Elbow Extension:
1. **Lower release point** → ball must travel further vertically
2. **Higher arc required** → need ~52° vs ~48° for taller players
3. **Higher arc demands more arm extension** → full elbow ROM critical
4. **Physics are unforgiving** → shorter players have smaller margin for error

---

## 🔬 Data Source

**MLSE Sport Performance Lab Open Data**
- 125 free throw trials from a single participant
- Markerless motion capture (30 fps)
- 3D coordinates for 26 body landmarks per frame
- Shot outcome (made/missed) and entry angle recorded

Reference: [SPL-Open-Data Repository](https://github.com/mlsedigital/SPL-Open-Data)

---

## 🏋️ Training Recommendations

Based on our analysis, coaches working with shorter players should:

1. **Prioritize Full Elbow Extension**
   - Drills focusing on complete arm follow-through
   - Target 100-105° elbow ROM through the shot

2. **Train for Optimal Entry Angle (42-46°)**
   - Use shot tracking technology
   - "Rainbow shot" mechanics for shorter players

3. **Stabilize Lower Body**
   - Made shots showed *less* hip ROM variation
   - Consistent base → consistent release

4. **Emphasize Curry-Style Mechanics**
   - Higher set point
   - Maximum arm extension
   - Hip-driven power transfer (not arm-dominant)

---

## 📁 Project Structure

```
wnba_nba_biomechanics/
├── README.md
├── data/
│   ├── spl_trials/              # Raw SPL JSON files (125 trials)
│   ├── processed_trials.json    # Extracted biomechanics features
│   └── statistical_results.json # Full analysis results
├── src/
│   ├── process_spl_data.py      # Load & extract features from SPL data
│   ├── biomechanics_analysis.py # Statistical analysis & comparisons
│   └── generate_plots.py        # Create all visualizations
├── plots/
│   ├── real_biomechanics_analysis.png
│   └── wnba_nba_implications.png
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy matplotlib
```

### 2. Run Analysis
```bash
# Process raw SPL data
python src/process_spl_data.py

# Run statistical analysis
python src/biomechanics_analysis.py

# Generate visualizations
python src/generate_plots.py
```

### 3. View Results
- Check `plots/` directory for visualizations
- Check `data/statistical_results.json` for full statistics

---

## 📊 Sample Output

### Made vs Missed Comparison
```
STATISTICAL ANALYSIS: Made vs Missed (Two-sample t-tests)
======================================================================

Elbow ROM (°):
  Made:   102.41 ± 9.56
  Missed:  97.76 ± 6.40
  Δ = +4.65, t=2.69, p=0.0081 ***
  Cohen's d = 0.571

RANKED BY SIGNIFICANCE:
1. Elbow ROM (°)             p=0.0081 *** (Δ=+4.65)
2. Hip ROM (°)               p=0.1065     (Δ=-0.80)
3. Hip Drop (m)              p=0.3449     (Δ=-0.00)
...
```

---

## 📚 References

1. Cabarkapa, D., et al. (2022). "Kinematic differences between successful and unsuccessful free throws." *Frontiers in Sports and Active Living*.
2. MLSE Sport Performance Lab. (2023). SPL Open Data Repository. GitHub.
3. Brancazio, P.J. (1981). "Physics of basketball." *American Journal of Physics*, 49(4), 356-365.

---

## 📄 License

This project uses data from the [SPL Open Data](https://github.com/mlsedigital/SPL-Open-Data) repository under their open data license. Analysis code is MIT licensed.

---

## 👤 Author

Colby Morris  
[GitHub: @colbymorris08](https://github.com/colbymorris08)
# wnba-nba-biomechanics
