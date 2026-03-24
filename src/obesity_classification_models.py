#Import necessary libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

"""#Load the dataset"""

data_path = '/content/ObesityDataSet.csv'
data = pd.read_csv(data_path)

"""# Inspect the first few rows"""

print(data.head())

"""# Drop rows with NaN values for simplicity"""

data.dropna(inplace=True)

"""# Separate features and target"""

X = data.drop('Obesity Level', axis=1)
y = data['Obesity Level']

"""# Convert categorical columns to numeric using one-hot encoding"""

X = pd.get_dummies(X)

"""# Encode target labels to numeric values


"""

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

"""# Split data into training and testing sets"""

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

"""### 1. Decision Tree with Variance Threshold Feature Selection ###"""

# Apply Variance Threshold
vt_selector = VarianceThreshold(threshold=0.01)
X_train_vt = vt_selector.fit_transform(X_train)
X_test_vt = vt_selector.transform(X_test)

# Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_vt, y_train)
y_pred_dt = dt_model.predict(X_test_vt)


# Print evaluation metrics
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt, target_names=label_encoder.classes_))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(dt_model, X_test_vt, y_test, display_labels=label_encoder.classes_)
plt.title("Decision Tree Confusion Matrix")
plt.show()

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns[vt_selector.get_support()])
plt.title("Decision Tree Visualization")
plt.show()

"""2. Support Vector Machine (SVM) with Recursive Feature Elimination (RFE)"""

# Scale the data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply RFE with Logistic Regression as the base estimator to select top 2 features
rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=2)
X_train_rfe = rfe_selector.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe_selector.transform(X_test_scaled)

# Train and evaluate SVM model
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_rfe, y_train)
y_pred_svm = svm_model.predict(X_test_rfe)

# Print evaluation metrics
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(svm_model, X_test_rfe, y_test, display_labels=label_encoder.classes_)
plt.title("SVM Confusion Matrix")
plt.show()

# Decision Boundary Plot for SVM with RFE-selected Features
plt.figure(figsize=(10, 6))
x_min, x_max = X_test_rfe[:, 0].min() - 1, X_test_rfe[:, 0].max() + 1
y_min, y_max = X_test_rfe[:, 1].min() - 1, X_test_rfe[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict on the mesh grid and ensure labels are numeric
xy_mesh = np.c_[xx.ravel(), yy.ravel()]
Z = svm_model.predict(xy_mesh)
Z = Z.reshape(xx.shape)

# Plot decision boundary and scatter plot of actual data points
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_test_rfe[:, 0], X_test_rfe[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("RFE Feature 1")
plt.ylabel("RFE Feature 2")
plt.title("SVM Decision Boundary with RFE-selected Features")
plt.colorbar(scatter, label="Obesity Levels")
plt.show()

"""3. Random Forest with SelectKBest Feature Selection



"""

# Apply SelectKBest to select the top 2 features
kbest_selector = SelectKBest(score_func=f_classif, k=2)
X_train_kbest = kbest_selector.fit_transform(X_train, y_train)
X_test_kbest = kbest_selector.transform(X_test)

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_kbest, y_train)
y_pred_rf = rf_model.predict(X_test_kbest)

# Print evaluation metrics
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(rf_model, X_test_kbest, y_test, display_labels=label_encoder.classes_)
plt.title("Random Forest Confusion Matrix")
plt.show()

# Decision Boundary Plot for Random Forest with SelectKBest-selected Features
plt.figure(figsize=(10, 6))
x_min, x_max = X_test_kbest[:, 0].min() - 1, X_test_kbest[:, 0].max() + 1
y_min, y_max = X_test_kbest[:, 1].min() - 1, X_test_kbest[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict on the mesh grid for Random Forest
Z = rf_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot decision boundary and scatter plot of actual data points
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_test_kbest[:, 0], X_test_kbest[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("SelectKBest Feature 1")
plt.ylabel("SelectKBest Feature 2")
plt.title("Random Forest Decision Boundary with SelectKBest-selected Features")
plt.colorbar(scatter, label="Obesity Levels")
plt.show()