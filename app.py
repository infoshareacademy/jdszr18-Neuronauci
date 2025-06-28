import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Załaduj wytrenowany model
model = tf.keras.models.load_model("emotion_cnn_model.h5")

# Słownik klas
class_labels = {
    0: 'angry',
    1: 'contempt',
    2: 'disgust',
    3: 'fear',
    4: 'happiness',
    5: 'neutral',
    6: 'sadness',
    7: 'surprise'
}

# Tytuł aplikacji
st.title("Emotion Detector 🤖")
st.write("Prześlij zdjęcie twarzy, a model spróbuje wykryć emocję.")

# Upload zdjęcia
uploaded_file = st.file_uploader("Wgraj obraz (jpg/png)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Wgrane zdjęcie", use_column_width=True)

    # Przetwarzanie obrazu
    img = image.resize((75, 75))  # Rozmiar zgodny z modelem
    img_array = np.array(img) / 255.0  # Normalizacja
    img_array = np.expand_dims(img_array, axis=0)  # Dodanie wymiaru batch

    # Predykcja
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Wynik klasyfikacji:")
    st.write(f"**Emocja**: {class_labels[predicted_class]}")
    st.write(f"**Pewność**: {confidence * 100:.2f}%")

    # Opcjonalnie: Pokaż rozkład procentowy dla każdej klasy
    st.subheader("Rozkład prawdopodobieństwa:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob * 100:.2f}%")


# aby uruchomić aplikację w terminalu przejdź do folderu, gdzie masz app.py i wpisz: streamlit run app.py