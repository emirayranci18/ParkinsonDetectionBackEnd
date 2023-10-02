import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd_data = pd.read_csv('parkinsons.data')
pd_data.head(10)
pd_data.shape
pd_data.groupby('status').count()
pd_data.info()
pd_data.isnull().sum()
pd_data.iloc[:,1:][~pd_data.iloc[:,1:].applymap(np.isreal).all(1)]
pd_data.describe().transpose()
#d_data.corr()
pd_data.kurtosis(numeric_only  = True)
pd_data.skew(numeric_only  = True)
print("The average vocal fundamental frequency of a person is {:.2f} and 99% of the people have a Fo of {:.2f}".format(
    pd_data['MDVP:Fo(Hz)'].mean(), pd_data['MDVP:Fo(Hz)'].quantile(0.90)))
q3 = pd_data['MDVP:Fhi(Hz)'].quantile(0.75)
q1 = pd_data['MDVP:Fhi(Hz)'].quantile(0.25)
t = q3-q1
outliers_above = q3+t
outliers_below = q1-t
mean_val = pd_data['MDVP:Fhi(Hz)'].loc[pd_data['MDVP:Fhi(Hz)']<=outliers_above].mean()
pd_data['MDVP:Fhi(Hz)'] = pd_data['MDVP:Fhi(Hz)'].mask(pd_data['MDVP:Fhi(Hz)']>outliers_above,mean_val)
print("The minimum vocal fundamental frequency of a person is {:.2f} and 99% of the people have a Flo of {:.2f}".format(pd_data['MDVP:Flo(Hz)'].mean(),pd_data['MDVP:Flo(Hz)'].quantile(0.90)))
q3 = pd_data['MDVP:Flo(Hz)'].quantile(0.75)
q1 = pd_data['MDVP:Flo(Hz)'].quantile(0.25)
t = q3-q1
outliers_above = q3+t
outliers_below = q1-t
max_val = pd_data['MDVP:Flo(Hz)'].loc[pd_data['MDVP:Flo(Hz)']<=outliers_above].max()
pd_data['MDVP:Flo(Hz)'] = pd_data['MDVP:Flo(Hz)'].mask(pd_data['MDVP:Flo(Hz)']>outliers_above,max_val)
print("The minimum vocal fundamental frequency of a person is {:.2f} and 99% of the people have a Jitter of {:.2f}".format(pd_data['MDVP:Jitter(%)'].mean(),pd_data['MDVP:Jitter(%)'].quantile(0.90)))
q3 = pd_data['MDVP:Jitter(%)'].quantile(0.75)
q1 = pd_data['MDVP:Jitter(%)'].quantile(0.25)
t = q3-q1
outliers_above = q3+t
outliers_below = q1-t
max_val = pd_data['MDVP:Jitter(%)'].loc[pd_data['MDVP:Jitter(%)']<=outliers_above].max()
pd_data['MDVP:Jitter(%)'] = pd_data['MDVP:Jitter(%)'].mask(pd_data['MDVP:Jitter(%)']>outliers_above,max_val)
q3 = pd_data['MDVP:Jitter(Abs)'].quantile(0.75)
q1 = pd_data['MDVP:Jitter(Abs)'].quantile(0.25)
t = q3-q1
outliers_above = q3+t
outliers_below = q1-t
mean_val = pd_data['MDVP:Jitter(Abs)'].loc[pd_data['MDVP:Jitter(Abs)']<=outliers_above].mean()
pd_data['MDVP:Jitter(Abs)'] = pd_data['MDVP:Jitter(Abs)'].mask(pd_data['MDVP:Jitter(Abs)']>outliers_above,mean_val)
q3 = pd_data['MDVP:PPQ'].quantile(0.75)
q1 = pd_data['MDVP:PPQ'].quantile(0.25)
t = q3-q1
outliers_above = q3+t
outliers_below = q1-t
max_val = pd_data['MDVP:PPQ'].loc[pd_data['MDVP:PPQ']<=outliers_above].max()
pd_data['MDVP:PPQ'] = pd_data['MDVP:PPQ'].mask(pd_data['MDVP:PPQ']>outliers_above,max_val)
q3 = pd_data['Jitter:DDP'].quantile(0.75)
q1 = pd_data['Jitter:DDP'].quantile(0.25)
t = q3-q1
outliers_above = q3+t
outliers_below = q1-t
max_val = pd_data['Jitter:DDP'].loc[pd_data['Jitter:DDP']<=outliers_above].max()
pd_data['Jitter:DDP'] = pd_data['Jitter:DDP'].mask(pd_data['Jitter:DDP']>outliers_above,max_val)
pd_data.kurtosis(numeric_only  = True)
pd_data.skew(numeric_only  = True)
pd_data.to_csv('cleanedParkinson.data')