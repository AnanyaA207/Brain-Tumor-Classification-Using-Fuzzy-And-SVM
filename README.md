# 🧠 Brain Tumor Classification using Fuzzy Logic & SVM

A machine learning project that enhances MRI-based brain tumor classification by combining **fuzzy logic** (for uncertainty handling) and **Support Vector Machines (SVM)** (for robust classification). This hybrid approach improves detection accuracy, even with small datasets.

---

## 🚀 Project Overview

Brain tumors are complex and vary in size, shape, and texture—making manual detection difficult and error-prone. Our system automates tumor classification using:

- 🔁 **Fuzzy Logic** to model uncertainty in MRI images  
- 🧮 **Defuzzification** to extract meaningful crisp data  
- 🔍 **SVM Classifier** for precise tumor detection

---

## 🧪 Methodology

**1. Preprocessing**  
- Load MRI images  
- Normalize and enhance contrast

**2. Fuzzification**  
- Apply sigmoid membership function to pixel intensity values  
- Represent uncertainty in tumor boundaries

**3. Defuzzification**  
- Convert fuzzy values into crisp intensity using multiple methods:  
  - Center of Gravity (COG)  
  - Mean/Last/First of Maxima (MoM, LoM, FoM)  
  - Weighted Average  
  - Height Method

**4. Classification**  
- Flatten images  
- Train SVM using RBF kernel  
- Evaluate performance metrics (Accuracy, Precision, Recall, F1-score)

---

## 📊 Results

| Defuzzification Method | Accuracy (%) |
|------------------------|--------------|
| Center of Gravity (COG) | **95.65%**   |
| Weighted Average        | 92.30%       |
| Last of Maxima (LOM)    | 92.10%       |
| Height Method           | 92.00%       |
| First of Maxima (FOM)   | 91.70%       |
| Mean of Maxima (MOM)    | 91.80%       |

📌 **COG performed best** due to superior feature preservation and noise resistance.

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **Libraries:** OpenCV, NumPy, Matplotlib, scikit-fuzzy, scikit-learn

---


