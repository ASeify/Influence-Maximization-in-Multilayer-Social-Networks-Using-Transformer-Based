# Multilayer Network Analyzer with SIR Simulation

This project provides tools to analyze and simulate multilayer networks using the NetworkX library. It includes several custom classes for handling real-world datasets, computing node centralities, ranking network layers, and simulating SIR-based diffusion models.

## Features

- Construction and handling of multilayer networks
- Node centrality analysis (degree, k-shell, etc.)
- SIR diffusion simulation on complex networks
- Ranking and evaluation of layers
- Visual feedback with `termcolor`
- GUI file selection with `easygui`

## Folder Structure

├── Extract Network Nodes and Layers Features/
│ ├──Multilayer_Network.ipynb
├── Classes/
│ ├── Files_Handler_Class.py
│ ├── K_Shell_Calculate_Class.py
│ ├── SIR_Diffusion_Model_Class.py
│ └── ... (other class files)
├── requirements.txt
└── README.md