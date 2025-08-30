# 🧠 Breast Cancer Classification using Deep Learning

This project implements a **Neural Network model** using **TensorFlow/Keras** to classify breast tumors as **Malignant (cancerous)** or **Benign (non-cancerous)** based on the **Breast Cancer Wisconsin dataset** from Scikit-Learn.

---

## 📌 Project Overview
- Preprocessed **569 patient records** with **30 medical features** (mean radius, texture, perimeter, area, smoothness, etc.).  
- Built and trained a **Deep Learning model** with 2 hidden layers using **ReLU** and **Sigmoid** activations.  
- Achieved **~95% test accuracy** in tumor classification.  
- Visualized **training accuracy/loss curves** for better performance monitoring.  
- Deployed model predictions on a **single patient input** for real-time classification.  

---

## ⚙️ Tech Stack
- **Languages & Libraries**: Python, NumPy, Pandas, Matplotlib, Scikit-Learn, TensorFlow, Keras  
- **ML Techniques**: Data preprocessing, Feature scaling, Train-test split, Deep Learning (ANN)  
- **Tools**: Jupyter Notebook / Python script  

---

## 📂 Dataset
The dataset is the **Breast Cancer Wisconsin dataset**, available directly from **Scikit-Learn**:  
```python
from sklearn.datasets import load_breast_cancer
Samples: 569

Features: 30

Classes:

0 → Malignant

1 → Benign

🚀 Model Architecture
Input Layer → 30 neurons (features)

Hidden Layer 1 → 20 neurons, ReLU activation

Hidden Layer 2 → 2 neurons, Sigmoid activation

Loss Function → Sparse Categorical Crossentropy

Optimizer → Adam

Metrics → Accuracy

📊 Results
Training Accuracy: ~96%

Validation Accuracy: ~94%

Test Accuracy: ~95%

📉 Example Accuracy & Loss curves:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
🧪 Prediction Example

# Sample input data (patient record)
input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,
              0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,
              12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# Preprocess input and predict
input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
input_data_std = scaler.transform(input_data_as_numpy_array)
prediction = model.predict(input_data_std)

if np.argmax(prediction) == 0:
    print("🔴 The tumor is Malignant")
else:
    print("🟢 The tumor is Benign")
📌 How to Run
Clone the repository:


git clone https://github.com/Om20An00/breast-cancer-classification.git
Install dependencies:


pip install -r requirements.txt
Run the script:


python breast_cancer_classification.py
🏆 Key Takeaways
✔️ Applied Deep Learning (ANN) on healthcare dataset.
✔️ Demonstrated 95%+ accuracy in medical classification.
✔️ Hands-on with data preprocessing, scaling, training & evaluation.
✔️ Built a predictive system for real-world tumor diagnosis support.
