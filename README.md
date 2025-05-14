
# 🧠 Homeostatic Dynamic Mean Field Model (DMF_ISP)

This repository contains Python code for simulating a **homeostatic dynamic mean field (DMF) model** with synaptic plasticity and generating synthetic BOLD signals. The model implements biologically inspired excitatory-inhibitory dynamics using a structural connectivity (SC) matrix derived from human brain data. It supports noise, homeostatic inhibitory plasticity, and simulation of functional connectivity (FC).

---

## 📂 Repository Contents

```
DMF_ISP/
├── BOLDModel.py              # Module for simulating BOLD signals from firing rates
├── DMF_ISP_numba.py          # Core DMF model with inhibitory synaptic plasticity (Numba-accelerated)
├── run_DMF_ISP.py            # Example script to run DMF simulation and generate FC matrix
├── SCmatrices88healthy.mat   # Structural connectivity matrix averaged across 88 healthy participants
```

---

## 🧬 Scientific References

### Structural Connectivity Source

Škoch, A., Rehák Bučková, B., Mareš, J., et al. (2022).  
*Human brain structural connectivity matrices–ready for modelling*.  
Scientific Data, 9, 486. https://doi.org/10.1038/s41597-022-01596-9

### DMF Model (Preprint)

Mindlin, I., Coronel-Oliveros, C., Sitt, J. D., Cofré, R., Luppi, A., Andrillon, T., & Herzog, R. (*in preparation*).  
*A homeostatic dynamic mean field model: enhanced stability and state repertoire*.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/DMF_ISP.git
cd DMF_ISP
```

2. Install the required Python dependencies:

```bash
pip install numpy scipy matplotlib numba
```

---

## 🚀 Usage

To run a basic simulation:

```bash
python run_DMF_ISP.py
```

This will:

- Load the structural connectivity matrix
- Run the DMF model with or without plasticity
- Generate synthetic BOLD signals
- Compute and display a functional connectivity matrix

---

## 🧠 Model Description

- **Nodes**: 90 brain regions
- **Inputs**: Structural connectivity (SC), noise
- **Outputs**: Firing rates, BOLD signals, FC matrix
- **Plasticity**: Homeostatic inhibitory plasticity targeting a fixed mean firing rate
- **Numerical Integration**: Euler-Maruyama with Gaussian noise
- **Optimization**: Numba-accelerated for performance

---

## 📊 Outputs

- **BOLD signals**: simulated low-frequency hemodynamic response
- **FC matrix**: pairwise Pearson correlations between regions
- **Plots**: connectivity matrix visualization

---

## 📦 Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `numba`

---


