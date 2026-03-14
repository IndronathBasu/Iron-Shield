# 🛡️ Iron Shield

### AI-Powered Non-Invasive Anemia Risk Detection

<p align="center">

<img src="https://img.shields.io/badge/AI-Medical%20Imaging-darkred">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red">
<img src="https://img.shields.io/badge/Computer%20Vision-Anemia%20Detection-bloodred">
<img src="https://img.shields.io/badge/Status-Research%20Prototype-darkred">

</p>

---

# 🩸 Overview

**Iron Shield** is an AI-based medical imaging system designed to **estimate Hemoglobin (Hb) levels and detect anemia risk using eye images**.

Instead of traditional blood tests, the system analyzes the **palpebral conjunctiva** (inner eyelid) to estimate anemia severity using deep learning.

The system integrates **computer vision, medical image segmentation, and regression models** to provide a **non-invasive anemia screening tool**.

---

# 🧠 AI System Architecture

```
Eye Image
   │
   ▼
👁 Eye Detection (YOLOv8)
   │
   ▼
✂ Eye Cropping
   │
   ▼
🧬 Conjunctiva Segmentation (U-Net)
   │
   ▼
🔬 Conjunctiva Extraction
   │
   ▼
🩸 Hemoglobin Prediction (DenseNet121)
   │
   ▼
⚠ Anemia Risk Classification
```

---

# ⚙️ Technologies Used

| Component        | Technology  |
| ---------------- | ----------- |
| Deep Learning    | PyTorch     |
| Object Detection | YOLOv8      |
| Segmentation     | U-Net       |
| Feature Learning | DenseNet121 |
| Image Processing | OpenCV      |
| Data Handling    | Pandas      |
| Visualization    | Matplotlib  |

---

# 📊 Model Components

## 👁 Eye Detection

Detects eye region from full image.

Model:

```
YOLOv8
```

Dataset:

```
Eye Detection Dataset (~2000 annotated images)
```

---

## 🧬 Conjunctiva Segmentation

Extracts the **palpebral conjunctiva region**, which is used for anemia detection.

Model:

```
U-Net (ResNet34 Encoder)
```

Dataset:

```
Eyes-Defy-Anemia Dataset
211 annotated images
```

---

## 🩸 Hemoglobin Prediction

Predicts hemoglobin level from conjunctiva region.

Model:

```
DenseNet121 Regression Model
```

Output:

```
Predicted Hb Level (g/dL)
```

---

## ⚠ Risk Classification

Based on WHO guidelines:

| Hb Level | Risk          |
| -------- | ------------- |
| < 8      | Severe Anemia |
| 8 – 10   | Moderate      |
| 10 – 12  | Mild          |
| > 12     | Normal        |

---

# 📂 Project Structure

```
anemia-ai
│
├── dataset
│   ├── raw
│   ├── segmentation
│   └── processed
│
├── detection
│   └── detect_eye.py
│
├── segmentation
│   ├── dataset_loader.py
│   ├── train_unet.py
│   └── predict_mask.py
│
├── models
│   ├── hb_model.py
│   ├── risk_model.py
│   └── *.pth
│
├── inference
│   └── predict_hb.py
│
├── pipeline
│   └── predict_pipeline.py
│
├── utils
│   ├── preprocessing.py
│   └── build_segmentation_dataset.py
│
├── outputs
│
├── README.md
└── .gitignore
```

---

# 🚀 Running the System

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 2️⃣ Run the Full AI Pipeline

```
python pipeline/predict_pipeline.py
```

Input:

```
Enter image path: eye_image.jpg
```

Example Output:

```
RESULT
--------------------
Hemoglobin : 11.4 g/dL
Risk Level : Mild Anemia
```

---

# 📈 Model Performance

### Hemoglobin Prediction

| Metric   | Score     |
| -------- | --------- |
| MAE      | 0.67 g/dL |
| RMSE     | 0.98 g/dL |
| R² Score | 0.82      |
| MAPE     | 5.46%     |

These results show promising performance for **non-invasive anemia screening**.

---

# 🧪 Dataset

### Eyes-Defy-Anemia Dataset

Contains:

```
218 eye images
Hb levels
age
gender
manual conjunctiva segmentation
```

Collected from:

```
India
Italy
```

---

# 🛡️ Key Features

✔ Non-invasive anemia screening
✔ End-to-end AI pipeline
✔ Automated conjunctiva detection
✔ Hemoglobin estimation
✔ Anemia risk classification

---

# 🔬 Future Improvements

* Attention U-Net for improved segmentation
* Larger clinical datasets
* Mobile application integration
* Real-time anemia screening
* Telemedicine deployment

---

# 👨‍💻 Author

**Indronath Basu**

AI/ML Developer
SRM Institute of Science and Technology

Focus Areas:

```
Computer Vision
Medical AI
Deep Learning
AI Systems
```

---

# ⚠ Disclaimer

This project is a **research prototype** and is **not intended for clinical diagnosis**.
Medical decisions should always be made by qualified healthcare professionals.

---

# ❤️ Contributions

Pull requests and improvements are welcome.

If you find this project useful, consider ⭐ starring the repository.
