import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


data = pd.read_csv("/Users/aseelali/Desktop/machine learning python/1.01. Simple linear regression.csv")
print(data)
print(data.describe())

#y = b0+b1x1
# define the dependent and independent variables
y = data['GPA']
x1 = data['SAT']

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print(results.summary())
print("aimona")
print("ossa")

plt.scatter(x1,y)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()

