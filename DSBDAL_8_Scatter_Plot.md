```python
import matplotlib.pyplot as plt
import numpy as np
```

```python

x=np.arange(0,10)
y=np.arange(0,10)
```

```python

plt.scatter(x,y,c='g')
plt.xlabel('X Axis')
plt.xlabel('Y Axis')
plt.title('Scatter Plot')
plt.savefig('scatterplot.png')
```

```python

x=np.arange(0,10)
y=x\*x
plt.scatter(x,y,c='r')
plt.xlabel('X')
plt.xlabel('Y=X^2')
plt.title('Square')
plt.plot(x,y,'r\*',linestyle='dashed')
plt.savefig('Square.png')
```

```python

plt.subplot(2,2,1)
plt.plot(x,y,'r--')
plt.subplot(2,2,2)
plt.plot(x,y,'g--')
plt.subplot(2,2,3)
plt.plot(x,y,'bo')
plt.subplot(2,2,4)
plt.plot(x,y,'go')
```

```python


a=np.arange(1,12)
b=(3\*a)+5
plt.plot(a,b,c='g')
plt.xlabel('X')
plt.xlabel('Y=3X+5')
plt.title('3X+5')

plt.savefig('Equation.png')
```

```python

c=np.arange(0,4\*np.pi,0.1)
d=np.sin(c)
plt.plot(c,d,c='r')
plt.title('Sin Wave')
plt.savefig('SinWave.png')
```

```python

x1=np.arange(0,4\*np.pi,0.1)
y_sin=np.sin(x1)
y_cos=np.cos(x1)
plt.subplot(2,2,1)
plt.plot(x1,y_sin,'r')
plt.title("Sin")
plt.subplot(2,2,2)
plt.plot(x1,y_cos,'m')
plt.title("Cos")
plt.savefig('Sin_Cos.png')
```

```python

x=[2,8,10]
y=[11,16,9]
x1=[3,9,11]
y1=[6,15,17]
plt.bar(x,y)
plt.bar(x1,y1,color='r')
```

```python

x=np.array([22,11,47,52,12,60,8,27,44,42])
plt.hist(x,bins=30, color='m')
plt.title("Histogram")
```

```python

data=[np.random.normal(0,std,100)for std in range(1,4)]
plt.boxplot(data,vert=True,patch_artist=False)
plt.title('Boxplot')
plt.savefig('Boxplot.png')
```

```python
label=['python','cpp','Ruby','Java',]
size=[215,15,245,210]
color=['gold','yellowgreen','lightcoral','lightskyblue']
explode=[0.04,0,0,0]
plt.pie(size,explode=explode,labels=label,colors=color)
plt.title("Pie Chart")
plt.savefig('PieChart.png')
```
