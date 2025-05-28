# 🧠 Cortical Dyslamination Evaluation with MEA Electrode Classification

Welcome to a neuroscience-powered toolkit designed to evaluate **cortical delamination** through **multi-electrode array (MEA)** signal analysis. This repository contains all the code used in our study to assess signal variability, oscillatory patterns, and electrode behavior, offering a modular, flexible framework for brain signal exploration.

---

## 🔍 What’s This About?

We believe that understanding baseline electrical activity is key to characterizing **oscillatory heterogeneity**, which can signal early markers of pathological responses like seizures. While this work focuses on cortical dysplasia and delamination, the analysis pipeline can be generalized to other models and signal processing tasks.

So dive in, try it out, and feel free to reach out if you hit any roadblocks or have questions! 💬

---

## ⚙️ Quick Start: How to Use This Repository

Here’s a quick overview of the workflow:

### 1️⃣ Set Up Your Environment
Make sure to install the required Python libraries in a fresh environment.  
👉 You can use a `requirements.txt` or a virtual environment setup (coming soon).

### 2️⃣ Convert Raw Data  
Convert your acquisition software output to **HDF5** format using the [MCS Py Data Tools](https://github.com/multichannelsystems/McsPyDataTools).

### 3️⃣ Preprocess Signals  
Run [`Preprocessing.py`](./Preprocessing.py) to clean and prepare your raw LFP data. This step is essential 🧼

### 4️⃣ Decompose the Signal  
Use [`SlepianTapers.py`](./SlepianTapers.py) for robust time-frequency decomposition using multitaper methods.

### 5️⃣ Classify Electrodes  
Run [`ElectrodeClustering.py`](./ElectrodeClustering.py) to cluster electrodes based on extracted features.

---

## 🧪 Explore Signal Dynamics (Bonus Tools)

Want to go deeper? Here are some optional — but powerful — scripts:

- [`SignalClassification.py`](./SignalClassification.py): Explore how signal patterns evolve over time using the electrode classes.
- [`MI_signalModules.py`](./MI_signalModules.py): Analyze **mutual information** between features in your signals.
- [`DirectMI.py`](./DirectMI.py): Estimate **directed transfer entropy** to uncover temporal dependencies and potential causal flow 🔁

---

## 🤝 Contributions & Questions

If you encounter any issues, ideas, or questions, please feel free to open an issue or contact me directly. Collaboration and curiosity are always welcome!
You can reach me at: anaaquiles@ciencias.unam.mx/ana.aquiles@igf.cnrs.fr

---
