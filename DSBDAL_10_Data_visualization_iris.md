# Data Visualization Iris

```python
import pandas as pd
from sklearn.datasets import load_iris
```

```python
# Load the Iris dataset
iris = load_iris()

# Convert the dataset into a pandas DataFrame
data = pd.DataFrame(iris.data, columns=iris.feature_names)
```

```python
# Add the target variable 'species' to the DataFrame
data['species'] = iris.target_names[iris.target]

# Display the first few rows to check the dataset
data.head()
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
# Create a histogram for each numeric feature in the dataset
numeric_features = data.drop(columns='species')

plt.figure(figsize=(15, 10))

# Loop over each numeric feature and plot its histogram
for i, feature in enumerate(numeric_features.columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data[feature], kde=True, bins=15)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(15, 10))

# Loop over each numeric feature and plot its box plot
for i, feature in enumerate(numeric_features.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='species', y=feature, data=data)
    plt.title(f'Box plot of {feature}')

plt.tight_layout()
plt.show()
```

```python
# Create box plots for each numeric feature
plt.figure(figsize=(12, 6))

for i, column in enumerate(data.select_dtypes(include='number').columns):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(y=data[column], color='lightgreen')
    plt.title(column)
```
