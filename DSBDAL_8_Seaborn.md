# Seaborn

```python
import seaborn as sns
```

```python
data=sns.load_dataset('tips')
data.head()
```

```python
nData=data[['total_bill','tip','size']]
```

```python
nData.corr()
```

```python
sns.heatmap(nData.corr())
```

```python
sns.displot(data['tip'])
```

```python
sns.jointplot(x="tip",y='total_bill',data=data,kind='hex')
```

```python
sns.pairplot(data)
```

```python
sns.pairplot(data,hue='sex')
```

```python
sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", data=data)
```
