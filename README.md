<p align="center">

# 🛡️ <span style="color:#b30000;">Iron Shield</span>

### <span style="color:#cc0000;">AI-Powered Non-Invasive Anemia Detection & Risk Awareness System</span>

</p>

<p align="center">

<img src="https://img.shields.io/badge/AI-Healthcare-darkred?style=for-the-badge"/>
<img src="https://img.shields.io/badge/DeepLearning-PyTorch-red?style=for-the-badge"/>
<img src="https://img.shields.io/badge/ComputerVision-MedicalAI-crimson?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Research%20Project-bloodred?style=for-the-badge"/>

</p>

---

# 🩸 Overview

**Iron Shield** is an AI system designed to **detect anemia non-invasively using eye conjunctiva images**.
Instead of blood tests, the system uses **deep learning to estimate hemoglobin levels from eye images**.

The project focuses on:

* Early anemia screening
* Risk awareness
* Monitoring hemoglobin improvement over time

Iron Shield uses **Computer Vision + Deep Learning (DenseNet121)** to predict hemoglobin levels and classify anemia severity.

---

# 🔬 How It Works

```
Eye Image
   │
   ▼
Conjunctiva Extraction
   │
   ▼
Image Preprocessing
   │
   ▼
Deep Learning Model (DenseNet121)
   │
   ▼
Hemoglobin Prediction
   │
   ▼
Anemia Risk Classification
```

---

# ❤️ Key Features

🩸 **Non-Invasive Hemoglobin Estimation**
🧠 **Deep Learning Model (DenseNet121)**
⚡ **GPU Accelerated Training (PyTorch)**
📊 **Evaluation Metrics for Medical AI**
📈 **Hemoglobin Progress Tracking**
🛡️ **Anemia Risk Classification System**
🔧 **Modular ML Architecture**

---

# 📂 Project Structure

```
IronShield
│
├── dataset
│   ├── raw
│   └── processed
│       ├── images
│       └── labels.csv
│
├── models
│   ├── hb_model.py
│   └── risk_model.py
│
├── training
│   └── train_hb_model.py
│
├── inference
│   └── predict_hb.py
│
├── evaluation
│   └── evaluate_model.py
│
├── utils
│   ├── dataset_loader.py
│   ├── extract_conjunctiva.py
│   └── preprocessing.py
│
├── api
│   └── app.py
│
├── dashboard
│
├── hb_model.pth
├── requirements.txt
└── README.md
```

---

# 🧬 Dataset

Iron Shield is trained on the **Eyes-Defy-Anemia Dataset**.

Dataset characteristics:

🩸 ~218 conjunctiva images
🩸 Pixel-level segmentation masks
🩸 Laboratory hemoglobin measurements
🩸 Patient metadata (age, gender)
🩸 Standardized lighting conditions

Dataset Source:

https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia

---

# 🧠 Model Architecture

The system uses **DenseNet121**, a powerful convolutional neural network known for efficient feature reuse.

Model configuration:

* Transfer learning with ImageNet weights
* Input resolution **224 × 224**
* CNN feature extractor
* Regression head for hemoglobin prediction

---

# 🏋️ Training

Run training with:

```
python training/train_hb_model.py
```

Training includes:

* GPU acceleration
* Train / validation split
* Progress bar in terminal
* Loss monitoring
* MAE & RMSE tracking

---

# 📊 Model Evaluation

Evaluate the trained model:

```
python evaluation/evaluate_model.py
```

Evaluation metrics used:

| Metric | Description             |
| ------ | ----------------------- |
| MAE    | Mean Absolute Error     |
| RMSE   | Root Mean Squared Error |
| R²     | Model explanatory power |
| MAPE   | Percentage error        |

Example output:

```
MAE  : 0.672 g/dL
RMSE : 0.989 g/dL
R²   : 0.825
MAPE : 5.46%
```

---

# 🔎 Prediction

Predict hemoglobin from a new eye image:

```
python inference/predict_hb.py
```

Example output:

```
Predicted Hemoglobin: 11.42 g/dL
Anemia Risk: Mild Anemia
```

---

# ⚠️ Anemia Risk Levels

Based on WHO hemoglobin thresholds.

| Hemoglobin   | Condition          |
| ------------ | ------------------ |
| ≥ 13 g/dL    | 🟢 Normal          |
| 11 – 13 g/dL | 🟡 Mild Anemia     |
| 8 – 11 g/dL  | 🟠 Moderate Anemia |
| < 8 g/dL     | 🔴 Severe Anemia   |

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/IndronathBasu/IronShield.git
cd IronShield
```

Install dependencies

```
pip install -r requirements.txt
```

---

# 🌍 Applications

Iron Shield can be used for:

🩸 Rural healthcare screening
🩸 Telemedicine systems
🩸 Preventive healthcare
🩸 Remote anemia monitoring
🩸 Low-cost diagnostics

---

# 🔮 Future Improvements

Planned upgrades:

* Automatic conjunctiva detection model
* Medical explainability using Grad-CAM
* FastAPI deployment
* Web dashboard for Hb progression
* Mobile screening application

---

# ⚠️ Disclaimer

Iron Shield is intended **for research and educational purposes only**
and should not replace professional medical diagnosis.

---

# 👨‍💻 Author

**Indronath Basu**
SRM Institute of Science and Technology

AI/ML • Computer Vision • Healthcare AI

---

<p align="center">

### 🩸 <span style="color:#b30000;">Iron Shield — Protecting Health with AI</span>

</p>
