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

## 📊 Outputs

## 🔍 Preprocessed MRI using Fuzzy Logic

Brain tumor boundaries are often blurred or vague in raw scans, making them tough to classify accurately.

Fuzzification assigns pixel intensities a membership value between 0 and 1, handling the ambiguity in a smart way. This step improves contrast and highlights potential tumor regions more clearly.

![Screenshot 2025-06-13 222348](https://github.com/user-attachments/assets/6cbd19c6-f247-4129-892f-150fef00dd0f)

![Screenshot 2025-06-13 222505](https://github.com/user-attachments/assets/9295dbb5-86e1-43e0-a93c-c0c0efb6e372)

*The fuzzified MRI image showing enhanced boundary clarity. This processed image is used as input to the SVM classifier.*

---

## 🧠 Classification Results using Fuzzy + SVM

Once the MRI images are preprocessed with fuzzy logic, we pass them to a **Support Vector Machine (SVM)** for classification.

SVM is a powerful supervised learning model that works well with small, high-dimensional datasets—making it a great fit for medical imaging.

SVM classification was used to distinguish between tumor and non-tumor images after fuzzy preprocessing.
The best results were obtained using the Center of Gravity (COG) method for defuzzification.

![Screenshot 2025-06-13 222547](https://github.com/user-attachments/assets/16cddd5f-af75-4eb1-8328-c13dadda36fb)

![Screenshot 2025-06-13 222614](https://github.com/user-attachments/assets/9f854a2d-2abb-415c-92e4-fe96e1bef00b)


By combining fuzzy logic and SVM, we created a lightweight yet accurate pipeline for brain tumor detection using MRI images—balancing interpretability, performance, and speed.

---

## 🔮 Future Enhancements

- Implement CNN-based classification for deeper feature learning
- Expand dataset with real-world MRI images
- Build a GUI/web interface for easy use in clinical settings
- Compare SVM performance with newer ML models (Random Forest, XGBoost, etc.)

---

## 🤝 Let's Connect  
Always up for ideas or feedback!








