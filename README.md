# Logistic Regression on Iris Dataset (Multiclass Classification)

This project demonstrates the implementation of **Multinomial Logistic Regression** on the famous **Iris Dataset** using scikit-learn.  
We trained the model, evaluated its accuracy, visualized decision boundaries, and explored class probabilities.  

---

## Libraries Used
- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn (`LabelEncoder`, `train_test_split`, `LogisticRegression`, `accuracy_score`, `confusion_matrix`)  
- mlxtend (for decision region plotting)  

---

## Steps of the Project

### 1. Loading Dataset
- Loaded the **Iris dataset** into a pandas DataFrame.  
- Selected the following columns:  
  - `sepal length`  
  - `petal length`  
  - `species` (target column)  

---

### 2. Encoding Target Column
- Used `LabelEncoder` to convert the categorical target (`species`) into numeric values.  
- This allows the Logistic Regression model to process the classes.  

---

### 3. Train-Test Split
- Split the dataset into **80% training** and **20% testing** using `train_test_split`.  
- Features (`X`) = [sepal length, petal length]  
- Target (`y`) = species  

---

### 4. Logistic Regression Model
- Used scikit-learn’s `LogisticRegression` with:  
  - `multi_class='multinomial'` (by default `auto`)  
- Fitted the model on training data.  

---

### 5. Accuracy & Confusion Matrix
- Achieved **96% accuracy** on test data.  
- Confusion matrix showed only **1 misclassified point**.  

📷 <img width="640" height="547" alt="download (3)" src="https://github.com/user-attachments/assets/574acb14-4461-479f-a50e-4740b6100596" />
📷 <img width="756" height="590" alt="download (4)" src="https://github.com/user-attachments/assets/df70f5a9-87ee-44d7-ba81-7a4011b231a7" />
 

---

### 6. Class Probability Prediction
- Checked probability of all 3 classes for a custom query:  
  ```python
  query = np.array([[3.4, 2.7]])
  clf.predict_proba(query)   # gives probability for each class
  clf.predict(query)         # gives final class prediction

### 7. Decision Boundary Visualization

Used mlxtend to plot decision regions.

The model divided the dataset with 2 boundary lines (since it’s 3-class classification).

In higher dimensions, more boundaries are formed similarly.

📷 <img width="534" height="455" alt="download (2)" src="https://github.com/user-attachments/assets/0b796c05-b761-4804-95a7-7a9570522ded" />

### Results

Logistic Regression (multinomial) performed with 96% accuracy.

Misclassified only one sample in the test set.

Successfully visualized how Logistic Regression separates classes with clear boundaries

### Learning Outcome

Logistic Regression is powerful for multiclass problems using the softmax function.

Visualization with mlxtend provides intuition about decision boundaries.

Probability outputs give insight into model confidence for each class.
