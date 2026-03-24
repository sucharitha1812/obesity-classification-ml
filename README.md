# 🧠 Obesity Level Classification using Machine Learning

<p>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Multi--Class%20Classification-green"/>
  <img src="https://img.shields.io/badge/Data-Health%20%26%20Lifestyle-orange"/>
  <img src="https://img.shields.io/badge/Visualization-Matplotlib-11557C"/>
</p>

---

## 🚀 Predicting Obesity Levels from Lifestyle Data

This project builds a machine learning system to classify individuals into different obesity levels based on **eating habits, physical activity, and demographic attributes**.

The objective is to demonstrate how machine learning can support **early obesity detection** and enable **data-driven health interventions**.

---

## 🎯 Why This Project Matters

Obesity is a major global health issue linked to lifestyle patterns and behavioral factors. Early identification of obesity risk can help:

- Prevent severe health complications  
- Enable personalized lifestyle recommendations  
- Support healthcare decision-making  

This project demonstrates how machine learning can transform **behavioral and health data into actionable insights**.

---

## 📊 Project Snapshot

- **Task:** Multi-class classification  
- **Dataset:** Lifestyle & obesity dataset  
- **Models Used:** Decision Tree, Support Vector Machine (SVM), Random Forest  
- **Best Model:** Support Vector Machine (SVM)  
- **Best Accuracy:** **91.8%**

---

## ⚙️ Approach

### 1️⃣ Data Processing
- Handled mixed categorical and numerical features  
- Encoded categorical variables for model compatibility  
- Prepared dataset for training and evaluation  

---

### 2️⃣ Feature Selection

Applied multiple techniques to identify influential variables:

- Variance Threshold  
- Recursive Feature Elimination (RFE)  
- SelectKBest  

These methods helped identify key lifestyle factors such as **physical activity and dietary habits**.

---

### 3️⃣ Model Development

Three machine learning models were implemented and compared:

#### 🌳 Decision Tree
- Accuracy: ~84%  
- Strength: High interpretability  
- Limitation: Confusion in mid-level categories  

#### ⚡ Support Vector Machine (SVM)
- Accuracy: **91.8% (Best Model)**  
- Strength: Strong class separation and robust performance  
- Limitation: Slight overlap in similar categories  

#### 🌲 Random Forest
- Accuracy: ~73.8%  
- Strength: Handles complex feature interactions  
- Limitation: Performance impacted due to reduced feature set  

---

## 📈 Results & Insights

- SVM achieved the highest accuracy (**91.8%**) and best overall performance  
- Extreme obesity categories are easier to classify due to distinct patterns  
- Middle categories (Normal Weight, Overweight) show overlap, leading to misclassifications  
- Feature selection significantly influences model performance  

---

## 🧠 Key Learnings

- Feature selection techniques strongly impact model behavior and accuracy  
- Visualization tools (confusion matrices, decision boundaries) improve interpretability  
- Different models perform differently depending on data distribution  
- Trade-offs exist between interpretability (Decision Tree) and performance (SVM)  

---

## 📂 Repository Structure

```
obesity-classification-ml/
├── data/
│   └── ObesityDataSet.csv
├── notebooks/
│   └── obesity_classification_models.ipynb
├── src/
│   └── obesity_classification_models.py
├── reports/
│   └── obesity_classification_report.pdf
├── README.md
├── requirements.txt
```

---

## ▶️ How to Run

```
git clone https://github.com/sucharitha1812/obesity-classification-ml.git
cd obesity-classification-ml
jupyter notebook
```

Run:
```
notebooks/obesity_classification_models.ipynb
```

---

## 💼 Real-World Impact

- Supports early obesity detection systems  
- Can be integrated into healthcare monitoring platforms  
- Helps identify key lifestyle factors affecting obesity  
- Enables data-driven health recommendations  

---

## ⚠️ Limitations

- Reduced feature set impacts Random Forest performance  
- Overlapping classes affect classification accuracy  
- Dataset may not generalize to all populations  

---

## 🔮 Future Improvements

- Hyperparameter tuning using GridSearchCV  
- Use full feature set for ensemble models  
- Experiment with deep learning approaches  
- Deploy as a web-based health prediction application  

---

## ✅ Conclusion

This project demonstrates how machine learning can classify obesity levels using lifestyle data. Among the evaluated models, **Support Vector Machine (SVM)** achieved the best performance with **91.8% accuracy**, making it the most effective model for this task.

---

## 🎯 Project Highlights

✔ Multi-class Classification  
✔ Feature Selection Techniques  
✔ Model Comparison & Evaluation  
✔ Real-world Healthcare Application  
