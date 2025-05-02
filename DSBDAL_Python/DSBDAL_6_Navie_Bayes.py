import pandas as pd
import numpy as np

data = pd.read_csv("diabetes.csv")
data.head()

data.isnull().sum()

X = data.drop(columns=['Outcome'])
y = data['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score
accuracy = accuracy_score(y_test,Y_pred)
print("Accuracy: ",accuracy,"\nAccuracy(%): ",accuracy*100)
error=1-accuracy
print("Error Rate: ",error,"\nError rate (%):",error*100)

precision =precision_score(y_test, Y_pred,average='micro')
print("Precision: ", precision,"\nPrecision(%): ", precision%100)

recall = recall_score(y_test, Y_pred,average='micro')
print("Recall: ", recall,"\nRecall(%): ", recall\*100)

cm = confusion_matrix(y_test, Y_pred)
print("Confusion Matrix: \n",cm)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

