import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Memuat dataset
data_path = 'dataset/healthcare_dataset.csv'
data = pd.read_csv(data_path)

print(data.head())

# Encode variabel kategorikal
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

# Definisikan fitur dan target
X = data.drop('Medical Condition', axis=1)  
y = data['Medical Condition']  

# Membagi data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi dan melatih model RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Membuat prediksi pada data pengujian
y_pred = clf.predict(X_test)

# Mengevaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Menyimpan model dan label encoder
joblib.dump(clf, 'random_forest_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Fungsi untuk mengetes model yang telah disimpan
def test_model():
    # Memuat dataset yang sama
    data = pd.read_csv(data_path)
    
    # Memuat label encoder yang disimpan
    label_encoders = joblib.load('label_encoders.pkl')
    
    # Encode variabel kategorikal menggunakan label encoder yang disimpan
    for column in data.columns:
        if data[column].dtype == 'object' and column in label_encoders:
            le = label_encoders[column]
            data[column] = le.transform(data[column].astype(str))
    
    # Memuat model yang disimpan
    model = joblib.load('random_forest_model.pkl')
    
    # Memisahkan fitur dari target
    X_new = data.drop('Medical Condition', axis=1)
    y_new = data['Medical Condition']
    
    # Membagi data menjadi training dan testing sets
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
    
    # Membuat prediksi pada dataset pengujian yang sama
    y_pred_new = model.predict(X_test_new)
    
    # Mengevaluasi model pada dataset pengujian yang sama
    print("Accuracy on new data:", accuracy_score(y_test_new, y_pred_new))
    print("Classification Report on new data:")
    print(classification_report(y_test_new, y_pred_new))

# Mengetes model dengan dataset yang sama
test_model()