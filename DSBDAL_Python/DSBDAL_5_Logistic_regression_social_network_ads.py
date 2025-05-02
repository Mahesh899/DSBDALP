import numpy as np
import pandas as pd

data = pd.read_csv("social_network_ads.csv")
data.head()

print(data.describe())
print("-"*70)
print(data.isnull().sum())

x = pd.get_dummies(data.drop(columns=['Purchased']), drop_first=True)
y = data['Purchased']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0, solver='lbfgs')
lr.fit(x_train, y_train)
pred = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, pred, labels=lr.classes_)
print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
display_matrix = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
display_matrix.plot(cmap=plt.cm.Blues)
plt.show()

accuracy = accuracy_score(y_test,pred)
error_rate = 1 - accuracy
recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
print("Accuracy: ",accuracy,"\nAccuracy(%): ",accuracy*100)
print("Error Rate: ",error_rate,"\nError Rate(%): ",error_rate*100)
print("Recall: ",recall,"\nRecall(%): ",recall*100)
print("Precision: ",precision,"\nPrecision(%): ",precision*100)

plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Purchased', data=data, palette='viridis')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of Males and Females Who Purchased')
plt.legend(title='Purchased', labels=['Not Purchased', 'Purchased'])
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', hue='Purchased', data=data, palette='viridis')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Number of Males and Females Who Purchased')
plt.legend(title='Purchased', labels=['Not Purchased', 'Purchased'])
plt.show()

