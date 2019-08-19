# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:42:14 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\HP\\Desktop\\Recruitment task\\AirQualityUCI.csv")

dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\Recruitment task\\AirQualityUCI.csv",na_filter=True,na_values=-200)
d1=dataset.copy()

dataset.info()

dataset.shape
dataset.columns
'''dataset.isin([-200])
dataset.isin([-200]).sum()'''

'''dataset.loc[dataset['CO(GT)'].isin([-200])]
dataset.loc[dataset['CO(GT)'].isin([-200])].index[0]
dataset.loc[dataset['CO(GT)'].isin([-200])].index.tolist()'''

dataset.isna().sum()
dataset.head(10)
dataset.tail()
dataset.info()
dataset.dropna(how='all',inplace=True)
dataset.dropna(thresh=10,axis=0,inplace=True)
dataset.RH.isna().sum()

dataset.describe()

dataset['Hour']=dataset['Time'].apply(lambda x:(x.split('.')[0]))
#dataset['Hour']=dataset.Hour.str.slice(-8,-6).astype('float')
dataset['Hour']=dataset.Hour.astype("int")
dataset.Hour.head()

dataset['year']=pd.DatetimeIndex(dataset['Date']).year
dataset['Month']=pd.DatetimeIndex(dataset['Date']).month
#dataset.drop(['month'],axis=1,inplace=True)
#dataset['day']=pd.DatetimeIndex(dataset['Date']).day
#dataset['Month']=dataset.index.Month
#dataset.set_index('Month',inplace=True)
dataset.isna().sum()

dataset.drop(['NMHC(GT)'],axis=1,inplace=True)
#dataset['CO(GT)']=dataset.groupby(['Month','Hour'])["CO(GT)"].transform(lambda x: x.fillna(x.mean()))
dataset['CO(GT)']=dataset['CO(GT)'].fillna(dataset.groupby(['Month','Hour'])['CO(GT)'].transform('mean'))
dataset['NO2(GT)']=dataset['NO2(GT)'].fillna(dataset.groupby(['Month','Hour'])['NO2(GT)'].transform('mean'))
dataset['NOx(GT)']=dataset['NOx(GT)'].fillna(dataset.groupby(['Month','Hour'])['NOx(GT)'].transform('mean'))


dataset[dataset['PT08.S1(CO)'].isnull()]
#dataset['PT08.S1(CO)']=dataset.groupby(['Hour'])["PT08.S1(CO)"].transform(lambda x: x.fillna(x.mean()))
dataset['PT08.S1(CO)']=dataset['PT08.S1(CO)'].fillna(dataset.groupby(['Hour'])['PT08.S1(CO)'].transform('mean'))
dataset['PT08.S2(NMHC)']=dataset['PT08.S2(NMHC)'].fillna(dataset.groupby(['Hour'])['PT08.S2(NMHC)'].transform('mean'))
dataset['PT08.S3(NOx)']=dataset['PT08.S3(NOx)'].fillna(dataset.groupby(['Hour'])['PT08.S3(NOx)'].transform('mean'))
dataset['NO2(GT)']=dataset['NO2(GT)'].fillna(dataset.groupby(['Hour'])['NO2(GT)'].transform('mean'))
dataset['NOx(GT)']=dataset['NOx(GT)'].fillna(dataset.groupby(['Hour'])['NOx(GT)'].transform('mean'))
dataset['AH']=dataset['AH'].fillna(dataset.groupby(['Hour'])['AH'].transform('mean'))

dataset.isna().sum()
dataset.info()
dataset['Time']=dataset['Time'].apply(lambda x: x.replace('.',':'))
dataset['Time'].value_counts()
dataset[dataset["Time"].isin(['09:00:00:-200'])]
dataset.at[4887,'Time']='09:00:00'
dataset['Time'].value_counts()
#dataset['Time'].isin([-200]).sum()
dataset['Time']=pd.to_datetime(dataset['Time'],format='%H:%M:%S').dt.time
dataset['Timestamp']=dataset.loc[:,'Date']=pd.to_datetime(dataset.Date.astype(str)+' '+dataset.Time.astype(str))
dataset['Timestamp']=dataset['Timestamp'].apply(lambda x: x.timestamp()).astype('int')
##eda
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(dataset.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white',annot=True)

sns.lmplot(x='CO(GT)',y='RH',data=dataset)
sns.lmplot(x='PT08.S1(CO)',y='RH',data=dataset)
sns.lmplot(x='C6H6(GT)',y='RH',data=dataset)
sns.lmplot(x='PT08.S2(NMHC)',y='RH',data=dataset)
dataset.groupby(['Hour','RH'])['Hour'].count()[0:20].plot.bar()
dataset.groupby(['Date','RH'])['Date'].count()[0:20].plot.bar()
sns.distplot(dataset['RH'])
sns.jointplot(x='AH',y='RH',data=dataset)
sns.regplot(x='T',y='RH',data=dataset)
sns.regplot(x='Month',y='RH',data=dataset)
#sns.barplot(x='Date',y='RH',data=dataset)
sns.barplot(x='year',y='RH',data=dataset)
#sns.factorplot(x=['PT08.S5(O3)'],y='RH',data=dataset,kind='bar')
dataset['RH'].hist()

dataset.info()
dataset.drop(['Date'],axis=1,inplace=True)
dataset.drop(['Time'],axis=1,inplace=True)
dataset.drop(['year'],axis=1,inplace=True)

dataset.to_csv('air_preprocessing.csv',index=False)

y=dataset.RH.values
dataset.drop(['RH'],axis=1,inplace=True)
x=dataset.values

##scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.3,random_state=0)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
model.score(x_train,y_train)
model.score(x,y)
model.score(x_test,y_test)

from sklearn.model_selection import cross_val_score,KFold
kfold=KFold(n_splits=10)
score=cross_val_score(model,x,y,cv=kfold)
score
score.mean()

y_pred = model.predict(x_test)
y_pred

actual=np.exp(y_test)
predict=np.exp(y_pred)

plt.scatter(model.predict(x),y)
plt.scatter(y_test,y_pred)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
rms
error=np.exp(rms)
error

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog=y, exog=x).fit()
regressor_OLS.summary()


##decisiontreeregression
from sklearn.tree import DecisionTreeRegressor
dc_reg=DecisionTreeRegressor()
dc_reg
dc_reg.fit(x,y)
dc_reg.score(x,y)
dc_reg.score(x_train,y_train)
dc_reg.score(x_test,y_test)

dc_pred=dc_reg.predict(x_test)

from sklearn.model_selection import cross_val_score,KFold
dc_kfold=KFold(n_splits=10)
score1=cross_val_score(dc_reg,x,y,cv=dc_kfold)
score1
score1.mean()

#rmse
from sklearn.metrics import mean_squared_error
from math import sqrt
dc_rms=sqrt(mean_squared_error(y_test,dc_pred))
dc_rms
error=np.exp(dc_rms)
error


############# GRID SEARCH FOR DecisionTreeREGRESSOR #############

#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
param_test1 ={'criterion':['mse'],
              
              'max_depth': [1,5,10,15],
              'min_samples_split':[2,3,4,5,6]
              }


model = GridSearchCV(dc_reg, param_grid=param_test1, n_jobs=1)
model.fit(x_train,y_train)
print("Best Hyper Parameters:",model.best_params_)

model.score(x,y)
model.score(x_train,y_train)
model.score(x_test,y_test)

gd_pred=model.predict(x_test)

from sklearn.model_selection import cross_val_score,KFold
gd_kfold = KFold(n_splits=10,random_state=0)
score = cross_val_score(model,x,y,cv=gd_kfold, n_jobs=1)
score.mean()


from sklearn.metrics import mean_squared_error
from math import sqrt
gd_rms = sqrt(mean_squared_error(y_test, gd_pred))
gd_rms
error=np.exp(gd_rms)
error

##randomForestRegression

from sklearn.ensemble import RandomForestRegressor
rfr_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rfr_reg.fit(x,y)
rfr_reg.score(x_train,y_train)
rfr_reg.score(x_test,y_test)
rfr_reg.score(x,y)


rfr_pred=rfr_reg.predict(x_test)

from sklearn model_selection import cross_val_score,KFold
rfr_kfold=KFold(n_splits=10)
score2=cross_val_score(rfr_reg,x,y,cv=rfr_kfold)
score2.mean()

from sklearn.metrics import mean_squared_error
from math import sqrt
rms2=sqrt(mean_squared_error(y_test,rfr_pred))
rms2
error=np.exp(rms2)
error



###suportVectormachine
from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x,y)
svr_reg.score(x,y)
svr_reg.score(x_train,y_test)
svr_reg.score(x_test,y_test)

svr_pred=svr_reg.predict(x_test)

from sklearn.model_selection import cross_val_score,KFold
svr_kfold=KFold(n_splits=10)
svr_score=cross_val_score(svr_reg,x,y,cv=svr_kfold)
svr_score.mean()

from sklearn.metrics import mean_squared_error
from math import sqrt
svr_rms=sqrt(mean_squared_error(y_test,svr_pred))
svr_rms
error=np.exp(svr_rms)
error
