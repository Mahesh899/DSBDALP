# Linear Regression

Dataset Link: [Housing.csv](https://github.com/Mahesh899/DSBDALP/blob/main/Housing.csv)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

```python
data = pd.read_csv("housing.csv")
data.head()
```

```python
data.isnull().sum()
data = pd.get_dummies(data, drop_first=True)
data.head()
```

```python
x = data.drop(['price'], axis=1)
y = data['price']
```

```python
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
```

```python
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xtrain, ytrain)
```

```python
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)
```

```python
df=pd.DataFrame(ytrain_pred,ytrain)
df=pd.DataFrame(ytest_pred,ytest)
```

```python

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(ytest, ytest_pred)
print(mse)
mse = mean_squared_error(ytrain_pred,ytrain)
mse
```

```python

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(ytrain ,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred ,c='lightgreen',marker='s',label='Test data')
plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title("True value vs Predicted value")
plt.legend(loc= 'upper left')
plt.plot()
plt.show()
```
