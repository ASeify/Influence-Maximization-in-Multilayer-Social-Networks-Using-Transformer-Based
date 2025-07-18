# 📁 Data Folder

This folder contains all input and output data related to the multilayer network analysis project.It is structured to include:

- **Input Networks** (edgeslist format)
- **Generated Results** (SIR simulation, node/layer features)
- **Intermediate Caches** (for heavy computations)

---

## 📂 Structure


data/
├── [network_1]/\n
│   └── network.edgeslist\n
├── [network_2]/\n
│   └── network.edgeslist\n
│	└── Network/              # Auto-generated output folder
│	   └── [timestamp]/       # e.g., 2025_07_18_14
│	       ├── [network]_graph temp.csv
│	       ├── [network]_layers.csv
│	       ├── [network]_beta=0.01_landa=0.7_epoch=1000 temp SIR.csv
│	       ├── [network]_fn=... graph.csv
│	       └── [network]_fn=..._beta=0.01_landa=0.7_epoch=1000.csv
