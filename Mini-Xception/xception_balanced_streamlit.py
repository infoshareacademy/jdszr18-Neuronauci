import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Załaduj wytrenowany model (upewnij się, że plik modelu .h5 jest w tym samym folderze)
model = tf.keras.models.load_model("xception_nowa_baza.h5")

# Słownik klas (dopasuj do swojego zbioru danych)
class_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise',
    7: 'contempt'
}

st.title("Emotion Detector 🤖")
st.write("Prześlij zdjęcie twarzy, a model spróbuje wykryć emocję.")

uploaded_file = st.file_uploader("Wgraj obraz (jpg/png)", type=["jpg", "png"])

if uploaded_file is not None:
    # Otwórz i przekonwertuj obraz na skalę szarości (1 kanał)
    image = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(image, caption="Wgrane zdjęcie (grayscale)", use_container_width=True)

    # Zmień rozmiar do 48x48, tak jak wymaga model
    img = image.resize((48, 48))

    # Zamiana obrazu na tablicę numpy i normalizacja
    img_array = np.array(img) / 255.0

    # Dodaj wymiary: (1, 48, 48, 1) — batch size i kanał
    img_array = np.expand_dims(img_array, axis=0)   # (1, 48, 48)
    img_array = np.expand_dims(img_array, axis=-1)  # (1, 48, 48, 1)

    # Predykcja
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Wynik klasyfikacji:")
    st.write(f"**Emocja:** {class_labels[predicted_class]}")
    st.write(f"**Pewność:** {confidence * 100:.2f}%")

    st.subheader("Rozkład prawdopodobieństwa:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob * 100:.2f}%")
