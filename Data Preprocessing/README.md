# Data Aggregation & Outlier Detection Toolkit

This repository contains two Jupyter notebooks implementing core data‑processing workflows:

1. **Data_Aggregation.ipynb**  
   – Methods to merge, clean, and summarize multi‑source datasets.  
2. **Outlier_Detection.ipynb**  
   – Techniques to identify and visualize outliers in numerical data.

These notebooks were developed as part of the project  
> **Influence maximization in multilayer social networks using transformer‑based node embeddings and deep neural networks**  
> (Neurocomputing, DOI: https://doi.org/10.1016/j.neucom.2025.130939)

---

## 📋 Repository Structure

├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── environment.yml
├── notebooks/
│ ├── Data_Aggregation.ipynb
│ └── Outlier_Detection.ipynb
└── data/
└── sample_data.csv

---


- **notebooks/** – your two analysis scripts  
- **data/** – place small sample datasets here (e.g. `sample_data.csv`)  
- **requirements.txt** – pip installable dependencies  
- **environment.yml** – conda environment specification  
- **LICENSE** – MIT open‑source license  

---

## How to Run

### Data Aggregation

Place your raw CSV files into data/.

Open notebooks/Data_Aggregation.ipynb.

Follow each cell to load, merge, and output aggregated_data.csv.

### Outlier Detection

Ensure aggregated_data.csv (or any numeric dataset) is in data/.

Open notebooks/Outlier_Detection.ipynb.

Run cells to perform z‑score and IQR‑based outlier detection, with visualizations.

---

## Dependencies

All major libraries are listed in requirements.txt and environment.yml. Key packages:

pandas

NumPy

scikit‑learn

matplotlib

seaborn