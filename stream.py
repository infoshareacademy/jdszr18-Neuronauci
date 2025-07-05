# emotion_detector_lime_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ------------------- KONFIGURACJA ------------------- #
CLASS_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}
INPUT_SIZE = (75, 75)  # szerokość, wysokość

# ------------------- WCZYTYWANIE MODELU ------------------- #
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("paulina_nowy_set_dodany_lime_confus_matrix_dla_test.h5")

model = load_model()

# ------------------- FUNKCJE ------------------- #
def detect_face(image: Image.Image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return faces

def preprocess_for_model(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize(INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr

def predict_fn(images: np.ndarray):
    imgs = []
    for img in images:
        resized = cv2.resize(img, INPUT_SIZE)
        resized = resized.astype("float32") / 255.0
        imgs.append(resized)
    imgs = np.stack(imgs)
    return model.predict(imgs, verbose=0)

def explain_with_lime(image_arr: np.ndarray):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_arr,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
    )
    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=False,
        hide_rest=False,
        num_features=10,
        min_weight=0.0,
    )
    return temp, mask

# ------------------- STREAMLIT UI ------------------- #
st.set_page_config(page_title="Emotion Detector with LIME", layout="centered")
st.title("Emotion Detector 🤖")
st.write("Prześlij zdjęcie twarzy, a model wykryje emocję i pokaże co wpłynęło na decyzję (LIME).")

uploaded_file = st.file_uploader("Wgraj obraz (jpg/png)", type=["jpg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Wgrane zdjęcie", use_container_width=True)

    faces = detect_face(original_image)

    if len(faces) == 0:
        st.warning("Nie znaleziono twarzy na zdjęciu!")
        st.stop()

    x, y, w, h = faces[0]
    cropped_image = np.array(original_image)[y:y+h, x:x+w]
    cropped_pil = Image.fromarray(cropped_image)

    st.subheader("Wykryta twarz")
    st.image(cropped_pil, caption="Przycięta twarz", use_container_width=True)

    img_array = preprocess_for_model(cropped_pil)
    img_array_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array_batch, verbose=0)
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.subheader("Wynik klasyfikacji:")
    st.markdown(f"**Emocja:** {CLASS_LABELS[predicted_class]}")
    st.markdown(f"**Pewność:** {confidence * 100:.2f}%")

    st.subheader("Rozkład prawdopodobieństwa:")
    prob_cols = st.columns(len(CLASS_LABELS))
    for i, prob in enumerate(prediction[0]):
        with prob_cols[i]:
            st.metric(label=CLASS_LABELS[i], value=f"{prob * 100:.1f}%")

    st.subheader("Wizualizacja LIME:")
    lime_original, lime_mask = explain_with_lime(img_array * 255.0)
    lime_img = mark_boundaries(lime_original / 255.0, lime_mask)
    st.image(lime_img, caption="LIME – wpływ cech na decyzję", use_container_width=True)

    st.info("Żółte/ciemne obszary pokazują regiony istotne dla klasyfikacji.")
