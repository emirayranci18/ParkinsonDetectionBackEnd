import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
plt.style.use('ggplot')

df = pd.read_csv("cleanedParkinson.data")
df.head()
df.info()
df.describe()
# status column value counts
print(df.status.value_counts())
sns.countplot(x='status',data=df)
#plt.title('The Count of values of the Status Column')
#plt.show()
fig,ax=plt.subplots(figsize=(25,11))
#sns.heatmap(df.corr(),annot=True)
#plt.title("The Heatmap of correlation of the columns of the dataset")
#plt.show()
labels= df["status"]
features= df.drop(["name", "status"], axis=1)
scaler=MinMaxScaler((-1,1))
X=scaler.fit_transform(features)
y=labels
X_train,X_test,y_train,y_test= train_test_split(X,labels,test_size=0.2,random_state=111)
model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
with open("modelParkinson.pkl", "wb") as file:
    pickle.dump(model, file)
