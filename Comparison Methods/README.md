## Overview

This Jupyter Notebook provides a framework to **compare the performance** of multiple network influence and robustness methods on your graph data. It loads a single edgelist input, computes seeds or rankings via several methods, simulates diffusion (SIR) to evaluate robustness, and visualizes results side-by-side.

**Included Methods**:

- K-Shell
- CIM (Community Influence Maximization)
- CBIM (Centrality-based IM)
- MIM Reasoner
- MA IMMULTI
- Our Method (pre-trained model inference)

---

## Prerequisites

- **Python 3.7+**
- **Jupyter Notebook** or **JupyterLab**

### Required Packages

```bash
pip install networkx pandas matplotlib plotly termcolor tqdm scikit-learn
```

> If a `requirements.txt` is provided, install via:
>
> ```bash
> pip install -r requirements.txt
> ```

---

## Directory Structure

Arrange your project directory as follows:

```
project-root/
├── Classes/                          # Custom utility classes
│   ├── Files_Handler_Class.py
│   ├── Network_Node_Centrality_Class_Old.py
│   ├── K_Shell_Calculate_Class.py
│   ├── SIR_Diffusion_Model_Class.py
│   ├── Layers_Ranking_Class_Old.py
│   └── ...                           # Other helper classes
├── Comparetion Methods/                        # Comparetion notebooks
│   └── Results_Compare_Source_Code.ipynb  # This comparison pipeline
├── data/                             # Input graphs and outputs
│   └── <network_name>/
│       └── [network name].edgelist.            # Four-column edge list
└── README_Results_Compare.md         # ← This file
```

---

## Input Format

The pipeline expects a **Network edgelist file** in `data/<network_name>/[Network name].edgeslist` with **four space- or tab-separated columns** (no header):

```
Source_Node_Id  Source_Node_Layer  Destination_Node_Id  Destination_Node_Layer
```

> Note: *Layer** labels annotate multiple layers, not directionality.*

---

## Configuration

At the top of the notebook, adjust these variables:

| Variable                    | Description                                                        | Example                    |
| --------------------------- | ------------------------------------------------------------------ | -------------------------- |
| `file_path`                 | Path to your edgelist (folder or file)                             | `"../data/MyNet/"`         |
| `methods_list`              | List of method names to compare                                    | `['K-Shell','Our Method']` |
| `robustness_centrality_str` | Key for centrality used in robustness (`'b'`, `'d'`, `'c'`, `'e'`) | `'d'` (degree)             |
| `robustness_beta`           | Infection probability for SIR evaluation (0 < β ≤ 1)               | `0.5`                      |
| `draw_network`              | Toggle network drawing (True/False)                                | `False`                    |

---

## Running the Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```
2. **Open** `Notebooks/Results_Compare_Source_Code.ipynb`.
3. **Execute cells sequentially** (Shift + Enter):
   1. **Imports & Setup** — loads dependencies and `Files_Handler`.
   2. **File Selection** — choose your network folder or file.
   3. **Feature & Seed Extraction** — runs each method in `methods_list` to select seed nodes.
   4. **Robustness Evaluation** — simulates SIR diffusion (β = `robustness_beta`) for each seed set.
   5. **Visualization** — generates comparative plots (Plotly & Matplotlib) of infection vs. seed set size.

---

## Outputs

- **Interactive Plots**: Side-by-side comparison of each method’s diffusion curves.Stored under `data/<network_name>/Compare_Methods_.../`

All outputs are human-readable and organized by network name.

---

## Module Breakdown

| Module/Class                             | Purpose                                                |
| ---------------------------------------- | ------------------------------------------------------ |
| `Files_Handler_Class`                    | Parse input paths and manage file selection            |
| `Resd_Network_Infos_Class`		       | Resd required infos     							    |
| Plot Functions (`plot_comparison`, etc.) | Create bar charts, line plots, and interactive figures |

