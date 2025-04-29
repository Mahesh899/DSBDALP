```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
```

```python
data=pd.read_csv('StudentPerformance1.csv')
data
```

```python
fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(data['Placement_Score'], data['Placement_Offer_Count'])
plt.show()
```

```python
print(np.where((data['Placement_Score']<50) & (data['Placement_Offer_Count']>1)))
print(np.where((data['Placement_Score']>85) & (data['Placement_Offer_Count']<3)))
```

```python
z = np.abs(stats.zscore(data['Math_Score']))
z
```

```python
col = ['Math_Score', 'Reading_Score' ,'Writing_Score','Placement_Score']
data.boxplot(column=col)
plt.show()
```

```python
sortedReadingScore= sorted(data['Reading_Score'])
print(sortedReadingScore)
```

```python
q1 = np.percentile(sortedReadingScore, 25)
q3 = np.percentile(sortedReadingScore, 75)
print("Q1: ",q1,"Q3: ",q3)
```

```python
IQR = q3-q1
lowerBound = q1-(1.5*IQR)
upperBound = q3+(1.5*IQR)
print("Lower Bound: ",lowerBound,"Upper Bound: ", upperBound)
```

```python
readingOutliers = []
for i in sortedReadingScore:

    if (i<lowerBound or i>upperBound):
        print(i)
        readingOutliers.append(i)
print(readingOutliers)
```
