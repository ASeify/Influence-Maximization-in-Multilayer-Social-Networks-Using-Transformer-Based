# Multilayer Influence Maximization Full Model

This repository contains the complete implementation for:

> **Influence maximization in multilayer social networks using transformer-based node embeddings and deep neural networks**  
> Neurocomputing (2025), DOI: https://doi.org/10.1016/j.neucom.2025.130939

---

## 📂 Repository Structure

── .gitignore
├── LICENSE
├── README.md
├── Multilayer_Full_Model\
	└── 2024_12_13_9_0 model lr=0.0001 wd=1e-05 	#It is e.g.
		└── epoch train loss hist list.txt
		└── epoch valid loss hist list.txt
		└── epoch=204 loss_valid=9.8249 loss_train=9.6183.png
		└── highest_epoch_loss_train_hist.txt
		└── highest_epoch_loss_valid_hist.txt
		└── highest_epoch_train Adam lr=0.0001 wd=1e-05 epochs=204  loss_valid=9.8249 loss_train=9.6183.optim
		└── highest_epoch_train model lr=0.0001 wd=1e-05 epochs=204  loss_valid=9.8249 loss_train=9.6183.pt
└── Multilayer_Full_Model.ipynb