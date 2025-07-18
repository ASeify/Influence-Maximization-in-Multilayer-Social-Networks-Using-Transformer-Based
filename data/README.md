# 📁 Data Folder

This folder contains all input and output data related to the multilayer network analysis project.It is structured to include:

- **Input Networks** (edgeslist format)
- **Generated Results** (SIR simulation, node/layer features)
- **Intermediate Caches** (for heavy computations)

---

## 📂 Structure


data/
├── [network_1]/
│   └── network.edgeslist
├── [network_2]/
│   └── network.edgeslist
│	└── Network/              # Auto-generated output folder
│	   └── [timestamp]/       # e.g., 2025_07_18_14
│	       ├── [network]_graph temp.csv
│	       ├── [network]_layers.csv
│	       ├── [network]_beta=0.01_landa=0.7_epoch=1000 temp SIR.csv
│	       ├── [network]_fn=... graph.csv
│	       └── [network]_fn=..._beta=0.01_landa=0.7_epoch=1000.csv

## Files Description


Each subfolder under data/ (e.g., data/Facebook/) contains one or more edgelist files used as the source for network construction.

File format: .edgeslist

Usage: Loaded via GUI (easygui) for network building.

Generated Output
Upon running the notebook (Extract Network Nodes and Layers Features/Multilayer_Network.ipynb), a folder [Network_Name]/ is created. Inside it, a subfolder named by execution timestamp is generated (e.g., 2025_07_18_14/) to store results. The following files may be generated:

1. [Network_Name]_graph temp.csv
Purpose: Temporary cache of node features and SIR values.

Reason: Avoids recomputation if the process is interrupted due to heavy processing.

Usage: Loaded during reruns if available.

2. [Network_Name]_layers.csv
Purpose: Matrix of extracted layer features.

Format: First column = layer ID; other columns = layer-level features.

3. [Network_Name]_beta=..._landa=..._epoch=... temp SIR.csv
Purpose: Temporary cache of SIR simulation results.

Usage: Accelerates re-execution by skipping recomputation.

4. [Network_Name]_fn=... graph.csv
Purpose: Final processed node features and SIR values.

Status: Considered as the validated/canonical output.

Usage: If this file exists and can be read, it is loaded directly in subsequent runs.

5. [Network_Name]_fn=..._beta=..._landa=..._epoch=....csv
Purpose: Final output of the SIR diffusion simulation.

Usage: Loaded instead of re-running the simulation.


