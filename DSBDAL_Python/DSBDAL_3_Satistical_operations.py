import pandas as pd
import numpy as np
import pandas as pd
from sklearn import preprocessing

data=pd.read_csv('StudentPerformanceG.csv')
data.describe()

numericColumns= ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score', 'Placement_Offer_Count']
#Mean of each column
print("Mean:")
data[numericColumns].mean()

#Mode
print("Mode:")
data[numericColumns].mode().iloc[0]

#Median
print("Median:")
data[numericColumns].median()

#Standard Deviation
print("Standard Deviation:")
data[numericColumns].std()

#Maximum
print("Maximum:")
data[numericColumns].max()

#Minimum
print("Minimum:")
data[numericColumns].min()

#list that contains a numeric value for each response to the categorical variable.
e = preprocessing.OneHotEncoder()
enc_data = pd.DataFrame(e.fit_transform(data[['Gender']]).toarray())
enc_data

encodeData =data.join(enc_data)
encodeData

encodeData.describe()

