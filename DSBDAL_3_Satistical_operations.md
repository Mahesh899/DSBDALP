#Statistic Operations

Dataset Link: [StudentPerformanceG.csv](https://github.com/Mahesh899/DSBDALP/blob/main/StudentPerformanceG.csv)

```python
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import preprocessing
```

```python
data=pd.read_csv('StudentPerformanceG.csv')
data.describe()
```

```python
numericColumns= ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score', 'Placement_Offer_Count']
#Mean of each column
print("Mean:")
data[numericColumns].mean()
```

```python
#Mode
print("Mode:")
data[numericColumns].mode().iloc[0]
```

```python
#Median
print("Median:")
data[numericColumns].median()
```

```python
#Standard Deviation
print("Standard Deviation:")
data[numericColumns].std()
```

```python
#Maximum
print("Maximum:")
data[numericColumns].max()
```

```python
#Minimum
print("Minimum:")
data[numericColumns].min()
```

```python
#list that contains a numeric value for each response to the categorical variable.
e = preprocessing.OneHotEncoder()
enc_data = pd.DataFrame(e.fit_transform(data[['Gender']]).toarray())
enc_data
```

```python
encodeData =data.join(enc_data)
encodeData
```

```python
encodeData.describe()
```
