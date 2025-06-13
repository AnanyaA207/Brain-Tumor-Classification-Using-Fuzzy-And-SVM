import os
import cv2
import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
def load_data(data_dir):
    images, fuzzy_images, labels = [], [], []
    
    for label, category in enumerate(["no_tumor", "tumor"]):  
        category_path = os.path.join(data_dir, category)
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            img = cv2.resize(img, (128, 128))  

            # Normalize image to [0,1]
            img = img.astype(np.float64) / 255.0  

            # Apply fuzzy sigmoid membership function
            fuzzy_img = 1 / (1 + np.exp(-10 * (img - 0.5)))  

            images.append(img)  
            fuzzy_images.append(fuzzy_img)
            labels.append(label)
    
    return np.array(images), np.array(fuzzy_images), np.array(labels)

# Defuzzification Methods
def defuzzify(fuzzy_img, img, method="cog"):
    """Apply different defuzzification techniques."""
    defuzzified_img = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_membership = fuzzy_img[i, j]
            
            if method == "cog":  # Center of Gravity
                defuzzified_img[i, j] = np.sum(pixel_membership * img[i, j]) / (np.sum(pixel_membership) + 1e-6)
            elif method == "mom":  # Mean of Maxima
                max_val = np.max(pixel_membership)
                maxima_indices = np.where(pixel_membership == max_val)
                defuzzified_img[i, j] = np.mean(img[maxima_indices])
            elif method == "lom":  # Last of Maxima
                max_val = np.max(pixel_membership)
                maxima_indices = np.where(pixel_membership == max_val)
                defuzzified_img[i, j] = img[maxima_indices][-1]
            elif method == "fom":  # First of Maxima
                max_val = np.max(pixel_membership)
                maxima_indices = np.where(pixel_membership == max_val)
                defuzzified_img[i, j] = img[maxima_indices][0]
            elif method == "weighted_avg":  # Weighted Average
                defuzzified_img[i, j] = np.average(img, weights=pixel_membership)
    
    return defuzzified_img

# Path to the dataset
data_dir = "brain_tumor_dataset"
X_original, X_fuzzy, y = load_data(data_dir)

# Choose defuzzification method
selected_method = "cog"  # Change this to "mom", "lom", "fom", or "weighted_avg" to test other methods

print(f"\n Applying Defuzzification Method: {selected_method.upper()}")

# Apply defuzzification
X_defuzzified = np.array([defuzzify(fuzzy, img, selected_method) for img, fuzzy in zip(X_original, X_fuzzy)])

# Flatten images for SVM
X_flattened = X_defuzzified.reshape(X_defuzzified.shape[0], -1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', class_weight='balanced')  
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred) * 100  
print(f" Accuracy using {selected_method.upper()}: {accuracy:.2f}%")
print(" Classification Report:")
print(classification_report(y_test, y_pred))

# Display original, fuzzified, and defuzzified images with classification labels
fig, axes = plt.subplots(3, 5, figsize=(12, 8))

for i in range(5):
    # Get classification result (Tumor/No Tumor)
    label_text = "Tumor" if y_pred[i] == 1 else "No Tumor"

    # Original Image
    axes[0, i].imshow(X_original[i], cmap='gray')
    axes[0, i].set_title(f"Original ({label_text})")
    axes[0, i].axis('off')

    # Fuzzified Image
    axes[1, i].imshow(X_fuzzy[i], cmap='gray')
    axes[1, i].set_title("Fuzzified Image")
    axes[1, i].axis('off')

    # Defuzzified Image
    axes[2, i].imshow(X_defuzzified[i], cmap='gray')
    axes[2, i].set_title(f"Defuzzified ({selected_method.upper()})")
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()

    




 

