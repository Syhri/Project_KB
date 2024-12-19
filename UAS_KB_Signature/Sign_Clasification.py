import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Fungsi untuk memuat gambar dan label dari direktori dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for person in os.listdir(folder):
        person_folder = os.path.join(folder, person)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Memuat gambar dalam skala abu-abu
                if img is not None:
                    img = cv2.resize(img, (150, 150))  # Mengubah ukuran gambar menjadi 150x150
                    images.append(img)
                    labels.append(person)
    return images, labels

# Memuat dataset
dataset_dir = 'dataset/'
images, labels = load_images_from_folder(dataset_dir)

# Mengubah daftar gambar dan label menjadi array numpy
X = np.array(images)
y = np.array(labels)

# Meratakan gambar untuk digunakan dalam model SVM
X = X.reshape(X.shape[0], -1)

# Encode label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Membagi data menjadi training dan testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inisialisasi model SVM
model = SVC(kernel='linear', probability=True, random_state=42)

# Melatih model
model.fit(X_train, y_train)

# Membuat prediksi pada data pengujian
y_pred = model.predict(X_test)

# Mengevaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Menyimpan model dan label encoder
# joblib.dump(model, 'svm_signature_model.pkl')
# joblib.dump(label_encoder, 'label_encoder.pkl')