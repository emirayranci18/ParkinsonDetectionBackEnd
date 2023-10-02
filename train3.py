import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

parkinson_data = pd.read_csv("cleanedParkinson.data")

all_features = parkinson_data.drop(['name', 'status'], axis=1).columns

X = parkinson_data[all_features].values
y = parkinson_data['status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_all = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True)
rf_all.fit(X_train, y_train)

selected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                     'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                     'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                     'Shimmer:DDA', 'HNR']

selected_data = parkinson_data[selected_features]  # Seçili sütunlara sahip yeni bir veri seti oluşturun
selected_data.to_csv('selected_data.txt', index=False)  # Veri setini CSV dosyası olarak kaydedin

X_train_selected = X_train[:, [all_features.get_loc(feature) for feature in selected_features]]
X_test_selected = X_test[:, [all_features.get_loc(feature) for feature in selected_features]]

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42, verbose=True)
rf_selected.fit(X_train_selected, y_train)

accuracy_train_all = rf_all.score(X_train, y_train)
accuracy_train_selected = rf_selected.score(X_train_selected, y_train)

accuracy_test_all = rf_all.score(X_test, y_test)
accuracy_test_selected = rf_selected.score(X_test_selected, y_test)

joblib.dump(rf_selected, 'parkinson_model_selected.pkl')
print("Test verileri üzerinde doğruluk (Seçili özellikler):", accuracy_test_selected)
