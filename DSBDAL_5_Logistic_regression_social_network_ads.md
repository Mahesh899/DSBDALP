# Logistic Regression for Social Network Ads Classification

## Data Loading and Exploration

First, we import necessary libraries and load the dataset:

```python
import numpy as np
import pandas as pd

data = pd.read_csv("social_network_ads.csv")
data.head()
```

## Data Analysis

We examine the dataset statistics and check for missing values:

```python
print(data.describe())
print("-"*70)
print(data.isnull().sum())
```

## Data Preprocessing

We prepare the data for modeling by converting categorical variables and splitting into features/target:

```python
x = pd.get_dummies(data.drop(columns=['Purchased']), drop_first=True)
y = data['Purchased']
```

## Train-Test Split

We split the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
```

## Model Training

We train a logistic regression model:

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs')
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
```

## Model Evaluation

We evaluate the model's performance:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, pred, labels=lr.classes_)
print("Confusion Matrix:\n", cm)
```

```python
# Visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
display_matrix.plot(cmap=plt.cm.Blues)
plt.show()
```

## Performance Metrics

We calculate various performance metrics:

```python
accuracy = accuracy_score(y_test,pred)
error_rate = 1 - accuracy
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
print("Accuracy: ",accuracy,"\nAccuracy(%): ",accuracy*100)
print("Error Rate: ",error_rate,"\nError Rate(%): ",error_rate*100)
print("Recall: ",recall,"\nRecall(%): ",recall*100)
print("Precision: ",precision,"\nPrecision(%): ",precision*100)
```

## Visualization

We visualize the purchase patterns by gender:

```python
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Purchased', data=data, palette='viridis')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of Males and Females Who Purchased')
plt.legend(title='Purchased', labels=['Not Purchased', 'Purchased'])
plt.show()
```

```python
plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Purchased', data=data, palette='viridis')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of Males and Females Who Purchased')
plt.legend(title='Purchased', labels=['Not Purchased', 'Purchased'])
plt.show()
```
