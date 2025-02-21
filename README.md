# **Gene Expression-Based PE & OE Prediction Model**

## **Overview**
This repository provides a computational **gene expression-based model** designed to distinguish between **PE (Patient Endometrium) and normal samples, as well as OE (Ovarian Endometriosis) and normal samples** using **TensorFlow (Deep Neural Networks)** and **XGBoost** models.  

This tool allows researchers and bioinformaticians to analyze **gene expression profiles** for PE and OE conditions, process datasets, and apply machine learning techniques for prediction.

The model supports both **TensorFlow-based deep learning** and **XGBoost-based machine learning**, enabling users to choose the best-suited approach for their data.

## **Features**
✅ Supports **Deep Learning (TensorFlow) and Machine Learning (XGBoost) models**  
✅ **Command-line interface (CLI) for easy execution**  
✅ **Automatic preprocessing**, including:
   - Handling **duplicate genes** (selecting highest IQR)
   - **Cleaning gene symbols** (resolving aliases and separators)
   - **Ensuring proper gene alignment** for model input  
✅ Works with **custom gene expression datasets**  
✅ **No pre-trained models** – users train their own models  
✅ **Expandable architecture** for future updates  

---

## **Usage**
To run the model, use the following command:
```bash
python predict.py -d data/GSE7305_OE.csv -m tf -t OE
```
### **Arguments:**
- `-d / --data`: Path to the input gene expression CSV file  
- `-m / --model`: Model format (`tf` for TensorFlow, `xgb` for XGBoost)  
- `-t / --type`: Model type (`PE` for Patient Endometrium, `OE` for Ovarian Endometriosis)

### **Example:**
```bash
python predict.py -d data/GSE7305_OE.csv -m xgb -t OE
```
This will **process the samples in `GSE7305_OE.csv`** using the **XGBoost model for OE**.

---

## **File Structure**
```
📂 ems_oe_pe/
│── models/                # User's trained models
│   ├── tf_model_PE.keras  # TensorFlow model for PE
│   ├── tf_model_OE.keras  # TensorFlow model for OE
│   ├── xgb_model_PE.model # XGBoost model for PE
│   ├── xgb_model_OE.model # XGBoost model for OE
│── data/                  # Required files and datasets
│   ├── dataset_OE.csv         # Training dataset for OE
│   ├── dataset_PE.csv         # Training dataset for PE
│   ├── E-MTAB-694_PE.csv      # Test dataset for PE
│   ├── GSE7305_OE.csv         # Test dataset for OE
│   ├── metaData_OE.csv        # Metadata for OE
│   ├── metaData_PE.csv        # Metadata for PE
│   ├── tf_valid_genes_PE.pkl  # Gene list for TF PE model
│   ├── tf_valid_genes_OE.pkl  # Gene list for TF OE model
│   ├── xgb_valid_genes_PE.pkl # Gene list for XGB PE model
│   ├── xgb_valid_genes_OE.pkl # Gene list for XGB OE model
│── predict.py             # Main prediction script
│── README.md              # Documentation
│── requirements.txt       # Dependencies
```

---

## **Installation**
### **Clone the Repository**
```bash
git clone https://github.com/msisakht/ems_oe_pe.git
cd ems_oe_pe
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Data Format**
The input **CSV file** must follow this structure:

| Symbols | Sample1  | Sample2  | Sample3  |
|---------|---------|---------|---------|
| GUCA1A  | 3.104   | 3.204   | 3.304   |
| MIR5193 | 6.469   | 6.569   | 6.669   |
| CCL5    | 8.908   | 8.978   | 9.008   |

**Important Notes:**
- The **first column (`Symbols`) contains gene names**.
- Samples have unique **custom identifiers** (e.g., `GSM12345`).
- The model will automatically **clean and align genes** for compatibility.
- If a dataset has missing gene expression values, this may affect prediction performance.

---

## **PE & OE Test Data**
This repository provides the framework to analyze gene expression profiles for **Patient Endometrium (PE) and Ovarian Endometriosis (OE)**.

- **E-MTAB-694 (PE)** and **GSE7305 (OE)** were used **only as external test datasets**, not for model training.
- Users can explore these datasets on **ArrayExpress and GEO (Gene Expression Omnibus)** for further details:
  - 📌 **[E-MTAB-694 on ArrayExpress](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-694?query=e-mtab-694)**  
  - 📌 **[GSE7305 on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE7305)**

🔹 **Users must train their own models** using the provided datasets (`dataset_PE.csv`, `dataset_OE.csv`).

---

## **License**
This project is released under the **MIT License** – feel free to use and modify it!
