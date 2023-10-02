from sklearn.ensemble import RandomForestClassifier
import joblib
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

# Veri setini yükleme
parkinson_data = pd.read_csv("cleanedParkinson.data")

# Özellik adlarını ayıklama
features = parkinson_data.drop(['name', 'status'], axis=1).columns

# Özellik adlarını kaydetme
with open('features.txt', 'w') as f:
    f.write('\n'.join(features))

# Özellikleri ayırma
X = parkinson_data[features].values
y = parkinson_data['status'].values

# Rastgele Orman sınıflandırıcısını eğitme
rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True)
rf.fit(X, y)

labels= parkinson_data["status"]
features= parkinson_data.drop(["name", "status", "MDVP:APQ", "NHR" , "RPDE", "DFA" , "spread1", "spread2", "D2", "PPE"], axis=1)
scaler=MinMaxScaler((-1,1))
X=scaler.fit_transform(features)
y=labels
X_train,X_test,y_train,y_test= train_test_split(X,labels,test_size=0.2,random_state=111)
model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Modeli pkl olarak kaydetme
joblib.dump(rf, 'parkinson_model.pkl')
