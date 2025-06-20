# 🧠 Cortical Dyslamination Evaluation with MEA Electrode Classification

Welcome to a neuroscience-powered toolkit designed to evaluate **cortical delamination** through **multi-electrode array (MEA)** signal analysis. This repository contains all the code used in our study to [assess signal variability](https://www.biorxiv.org/content/10.1101/2025.06.16.659942v1), oscillatory patterns, and electrode behavior, offering a modular, flexible framework for brain signal exploration.

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

Before diving deep, I highly encourage you to explore [`CommonBand.py`](./CommonBand.py) — a method for extracting **common oscillatory activity** across your recordings. This is especially useful when comparing conditions like **control vs. disease**, as it reveals consistent rhythms that might otherwise be hidden 🧭.

Want to go further? Check out these optional **but powerful** scripts:

- [`SignalClassification.py`](./SignalClassification.py): Track how signal dynamics evolve over time using the electrode class labels.
- [`MI_signalModules.py`](./MI_signalModules.py): Quantify **mutual information** between different signal features.
- [`DirectMI.py`](./DirectMI.py): Reveal directed relationships in your data using **transfer entropy** 🔁

---

## 🤝 Contributions & Questions

Found a bug? Have an idea? Curious about applying this to your own dataset? Don’t hesitate to get in touch — collaboration and curiosity are always welcome! 🌱

📫 **Contact:**  
- anaaquiles@ciencias.unam.mx  
- ana.aquiles@igf.cnrs.fr  

---
