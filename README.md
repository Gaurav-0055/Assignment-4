# Assignment-4
# Breast Cancer Classification using K-Nearest Neighbors (KNN)

**AI-ML Assignment 4**

| | |
|---|---|
| **Name** | Gaurav Gour |
| **Registration Number** | 23BSA10096 |
| **Application Number** | IN26011516 |
| **Batch Number** | 1A |
| **Assignment Number** | Assignment-4 |

---

## Objective

To develop a K-Nearest Neighbors (KNN) classification model that predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements, supporting healthcare organizations in accurate and early cancer diagnosis.

---

## Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**

Kaggle Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Note: The dataset is not included in this repository. Please download it directly from the Kaggle link above and place `data.csv` in the same directory as the notebook.

---

## Libraries Used

- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib**, **seaborn** — data visualization
- **scikit-learn** — preprocessing, model development, and evaluation
  - `train_test_split`
  - `StandardScaler`, `LabelEncoder`
  - `KNeighborsClassifier`
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`

---

## Methodology

1. **Data Understanding** — Loaded the dataset using Pandas, inspected the first five records, identified numerical features and the target variable (`diagnosis`), and reviewed dataset information and summary statistics.

2. **Data Preprocessing** — Checked for missing values, removed unnecessary columns (`id` and an empty unnamed column), label-encoded the target variable (M → 1, B → 0), standardized all numerical features using `StandardScaler`, and split the dataset into 80% training and 20% testing sets.

3. **Model Development** — Trained a K-Nearest Neighbors classifier with K = 5 on the standardized training data and generated predictions on the test dataset.

4. **Model Evaluation** — Assessed model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9744 |
| Recall | 0.9048 |
| F1-Score | 0.9383 |

**Observations**

1. The model achieved an accuracy of 95.61%, indicating that KNN performs well on this dataset once features are properly scaled.
2. Precision (97.44%) is notably higher than recall (90.48%), meaning the model is very reliable when it predicts malignancy but still misses a small proportion of actual malignant cases — a gap worth noting given the clinical cost of false negatives.
3. The F1-score of 0.9383 reflects a strong overall balance between precision and recall, though the recall gap suggests that tuning K or exploring alternative distance metrics could further improve sensitivity to malignant cases.

---

## Conclusion

This project applied the K-Nearest Neighbors algorithm to classify breast tumors as Malignant or Benign using the Breast Cancer Wisconsin Diagnostic Dataset. Following preprocessing — including encoding of the target variable and standardization of all numerical features — the model achieved strong classification performance at K = 5, with an accuracy of 95.61%, precision of 97.44%, recall of 90.48%, and an F1-score of 0.9383.

Feature scaling proved essential for KNN, as the algorithm relies on Euclidean distance between data points; without normalization, features with larger numeric ranges, such as area or perimeter, would disproportionately influence the distance calculation and distort classification outcomes.

A key limitation of KNN is its computational cost at prediction time, as it requires calculating distances to all training points for every new prediction, which becomes inefficient on large datasets. The algorithm is also sensitive to irrelevant features and the choice of K, necessitating careful parameter tuning for optimal performance.
