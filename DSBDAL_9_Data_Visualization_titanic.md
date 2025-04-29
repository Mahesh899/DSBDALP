#Data Visualization Titanic

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data= sns.load_dataset('titanic')
```

```python
data.head()
data.info()
```

```python
data.describe()
```

```python
data.isnull().sum()
```

```python
data['age'].fillna(data['age'].mean(), inplace=True)
data.dropna(subset = ['embarked'], inplace=True)
data.drop(columns='deck', inplace=True)
```

```python
data.isnull().sum()
```

```python
data.duplicated().sum()
```

```python
data.drop(columns=data.loc[:, 'class':'alone'].columns, inplace=True)
data.head()
```

```python
data.nunique()
```

```python
data['survived'].value_counts()
```

```python
data['pclass'].value_counts()
```

```python
data['sex'].value_counts()
```

```python
data['embarked'].value_counts()
```

```python
data.groupby('sex')['survived'].sum().plot(kind='bar')
plt.xlabel('Sex')
plt.ylabel('No. of people')
plt.title('Survived vs Sex')
plt.show()
```

```python
data.groupby('pclass')['survived'].sum().plot(kind='bar')
plt.xlabel('Pclass')
plt.ylabel('No. of people')
plt.title('Survived vs Pclass')
plt.show()
```

```python
data.groupby('embarked')['survived'].sum().plot(kind='bar')
plt.xlabel('Embarked')
plt.ylabel('No. of people')
plt.title('Survived vs Embarked')
plt.show()
```

```python
sns.histplot(data=data, x='age', hue='survived', kde=True)
plt.show()
```

```python
sns.scatterplot(data=data, x='age', y='fare', hue='survived')
plt.show()
```

```python
sns.boxplot(data=data, x='survived', y='fare')
plt.show()
```

```python
sns.heatmap(data.corr(), annot=True)
plt.show()
```

```python
data.groupby('sex')['survived'].value_counts()
```

```python
# Create a figure with two subplots side by side

plt.figure(figsize=(12, 6))

# Pie chart for males

plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
male_data = data[data['sex'] == 'male']['survived'].value_counts()
plt.pie(male_data, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90)
plt.title('Male Survival')
```

```python
# Pie chart for females

plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
female_data = data[data['sex'] == 'female']['survived'].value_counts()
plt.pie(female_data, labels=['Survived', 'Not Survived'], autopct='%1.1f%%', startangle=90)
plt.title('Female Survival')

plt.tight_layout() # Adjust layout to prevent overlapping
plt.show()

```
