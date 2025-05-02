import seaborn as sns

data=sns.load_dataset('tips')
data.head()

nData=data[['total_bill','tip','size']]

nData.corr()

sns.heatmap(nData.corr())

sns.displot(data['tip'])

sns.jointplot(x="tip",y='total_bill',data=data,kind='hex')

sns.pairplot(data)

sns.pairplot(data,hue='sex')

sns.catplot(x="day", y="total_bill", hue="sex", kind="violin", data=data)

