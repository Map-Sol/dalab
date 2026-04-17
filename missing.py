import pandas as pd
import numpy as np
df=pd.DataFrame({
    'Name':['Alice','bob',None],
    'Age':[30,np.nan,31],
    'Salary':[5000,6000,np.nan]})
print('Drop:\n',df.dropna())
print('Fill Constant:\n',df.fillna({'Name':'Unknown','Age':0,'Salary':0}))
df_mean=df.copy()
df_mean['Age']=df_mean['Age'].fillna(df_mean['Age'].mean())

df_mean['Salary']=df_mean['Salary'].fillna(df_mean['Salary'].mean())

print('Fill Mean :\n', df_mean)
print('ffill Mean :\n', df.ffill())
print('bFill Mean:\n', df.bfill())
