import cv2
import numpy as np
import joblib

# Fungsi untuk memuat dan memproses gambar tanda tangan baru
def load_and_process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (150, 150)) 
    img = img.reshape(1, -1)  
    img = img.astype('float32') / 255 
    return img

model = joblib.load('svm_signature_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Path ke gambar tanda tangan baru yang ingin diklasifikasikan
new_image_path = 'test_signature.jpeg'

new_image = load_and_process_image(new_image_path)

predicted_label_num = model.predict(new_image)[0]

predicted_label = label_encoder.inverse_transform([predicted_label_num])[0]

print(f'Tanda tangan tersebut diklasifikasikan sebagai: {predicted_label}')