**README for `Oure_Method.ipynb`**

---

## Overview

This Jupyter Notebook implements the **Oure Method**: a pipeline for extracting multilayer network features and **inferring** nodes’ potential influence using a pre-trained neural network model. Instead of training from scratch, the notebook **loads an existing model** and applies it to predict each node’s diffusion strength.

The goal of this README is to walk you through preparing your data, running the inference pipeline, and interpreting the outputs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Preparing Your Network Data](#preparing-your-network-data)
5. [Configuring the Pipeline](#configuring-the-pipeline)
6. [Running the Notebook](#running-the-notebook)
7. [Understanding Outputs](#understanding-outputs)
8. [Detailed Module Descriptions](#detailed-module-descriptions)
9. [Contact & License](#contact--license)

---

## Prerequisites

- **Python 3.7+**
- **Jupyter Notebook** or **JupyterLab**

### Required Python Packages

```bash
pip install torch networkx pandas tqdm termcolor
``` 

> **Tip**: If you have a `requirements.txt`, run `pip install -r requirements.txt`.

---

## Directory Structure

```
project-root/
├── Classes/                   # Utility Python classes
│   ├── Bcolors_Class.py
│   ├── CSV_Files_Class.py
│   ├── Files_Handler_Class.py
│   ├── Get_Past_Results_Class.py
│   ├── K_Shell_Calculate_Class.py
│   ├── Layers_Ranking_Class_Old.py
│   ├── Network_Infos_Writer_Class.py
│   ├── Network_Node_Centrality_Class_Old.py
│   ├── Resd_Network_Infos_Class.py
│   └── SIR_Diffusion_Model_Class.py
├── Main Code/                # Notebook and related code
│   └── Oure_Method.ipynb      # Main inference pipeline
├── data/                     # Input & generated outputs by network name
│   └── <network_name>/        # e.g. "KarateClub/"
│       ├── [network_name].edgeslist     # Input file
│       ├── <network_name>_SIR_p/  # Model predictions (CSV outputs)
│       └── Our Method/       # Final JSON summary
│       └── ...
└── README.md
```

---


## Preparing Your Network Data

The pipeline expects **one input file**: an edge list with **four columns** (no header):

```
Source_Node_Id Source_Node_Layer Destination_Node_Id Destination_Node_Layer
```

- **Source_Node_Layer** and **Destination_Node_Layer** label each node’s layer (not a directed graph).  

---

## Running the Notebook

1. **Launch Jupyter**:
   ```bash
jupyter notebook
   ```
2. **Open** `Main Code/Oure_Method.ipynb`.
3. **Run cells in order**:
   - **Load Classes & Data**
   - **Compute Multilayer Features**  
     - *All features used by the neural model* are extracted here (layer density, degree distributions, k-shell values, etc.).  
   - **Load Pre-trained Model**  
     - The PyTorch model is instantiated and weights loaded from `model_path`.
   - **Inference Phase**  
     - The loaded model predicts each node’s **potential diffusion strength**.
	- **Finding Seedset Nodes**  
     - Find seedset nodes using model predicted potential diffusion.
   - **Export Results**  
     - Predictions and feature summaries are saved to CSV/JSON.

---

## Understanding Outputs

After running, the pipeline writes into `data/<network_name>/`:

1. **`<network_name>_SIR_p/`**
   - Contains two CSV files with **model-predicted diffusion** for each node.
2. **`Our Method/`**
   - A JSON file summarizing:
     - The **seed set size** (chosen nodes)
     - The **simulated diffusion quality** (SIR evaluation on that seed set)

All output files are human-readable and organized under your network’s data folder.

---

## Detailed Module Descriptions

| Module/Class                         | Functionality                                                     |
|--------------------------------------|-------------------------------------------------------------------|
| `Files_Handler_Class`                | Manages input, directories and paths                        |
| `Get_Past_Results_Class`             | Loads previous outputs for ignore recompution                              |
| `Network_Infos_Writer_Class`         | Exports features and predictions to CSV, JSON, or other formats    |
| `Network_Node_Centrality_Class_Old`  | Computes per-node centrality metrics (degree, closeness, etc.)     |
| `Layers_Ranking_Class_Old`           | **Extracts** per-layer feature vectors                              |
| `K_Shell_Calculate_Class`            | Computes k-shell decomposition                      |
| `SIR_Diffusion_Model_Class`          | Evaluates diffusion quality of a seed set via the SIR model         |
| `Multilayer_Full_Model` (PyTorch)    | Neural network model architecture for loading pretrained model  |

---


## Contact & License

- **Author**: Ali Seyfi (ali.seyfi.n@gmail.com)  
- **License**: MIT

Contributions and issue reports are welcome—feel free to open a GitHub issue or pull request!

