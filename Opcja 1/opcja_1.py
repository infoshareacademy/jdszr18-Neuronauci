import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random
import cv2
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------- konfiguracja ---------- #
MODEL_PATH = "paulina_nowy_set_dodany_lime_confus_matrix_dla_test.h5"
INPUT_SIZE = (75, 75)

class_labels = {
    0: "Złość",
    1: "Obrzydzenie",
    2: "Strach",
    3: "Szczęście",
    4: "Neutralność",
    5: "Smutek",
    6: "Zaskoczenie"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@st.cache_resource(show_spinner=False)
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def make_gradcam_heatmap(img_array, m, last_conv, target_class):
    grad_model = tf.keras.models.Model([m.inputs], [m.get_layer(last_conv).output, m.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.expand_dims(img_array, 0))
        loss = preds[:, target_class]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.5, cmap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cmap)
    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

def preprocess_image(pil_img):
    img = pil_img.resize(INPUT_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

openai_api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=openai_api_key)

def get_chatgpt_recommendation(emotion):
    prompt = (
        f"Wykryto emocję: {emotion}. "
        "Napisz krótką, empatyczną rekomendację (2–3 zdania) pomagającą w tym stanie. "
        "Po polsku, bez imion ani oceniania."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Jesteś wspierającym i empatycznym psychologiem."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Błąd zapytania do ChatGPT:\n{e}"

sad_jokes = [
    "Jak nazywa się ryba, która gra na gitarze? Rock’n’rolka.",
    "Co robi informatyk, gdy jest głodny? Szuka pliku cookie.",
    "Dlaczego komputerowi nigdy nie jest zimno? Bo ma Windows.",
    "Dlaczego programista nie lubi natury? Za dużo bugów.",
]

def show_typing(text, delay=0.02):
    container = st.empty()
    full_text = ""
    for char in text:
        full_text += char
        container.markdown(f"<p style='font-size: 22px;'>{full_text}</p>", unsafe_allow_html=True)
        time.sleep(delay)

# ---------- UI główne ---------- #
st.set_page_config(page_title="EmoTrack 🤖", layout="centered")

st.markdown("<h1 style='font-size: 40px;'>EmoTrack 🤖✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 22px;'>Wgraj zdjęcie twarzy – model wykryje emocję i pokaże Grad‑CAM.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("**Wgraj obraz (JPG/PNG)**", type=["jpg", "png"], label_visibility="visible")

if uploaded_file:
    try:
        original_pil = Image.open(uploaded_file).convert("RGB")
        st.image(original_pil, caption="Wgrane zdjęcie", use_container_width=True)

        original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        faces = face_cascade.detectMultiScale(original_cv, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("Nie wykryto twarzy na zdjęciu.")
        else:
            (x, y, w, h) = faces[0]
            face_img = original_cv[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            face_pil = Image.fromarray(face_rgb)
            smaller_face = face_pil.resize((int(face_pil.width * 0.67), int(face_pil.height * 0.67)))

            st.markdown("<h3 style='font-size: 22px;'>Wycinek twarzy do analizy</h3>", unsafe_allow_html=True)
            st.image(smaller_face, use_container_width=False)

            face_arr = preprocess_image(face_pil)
            last_conv = find_last_conv_layer(model)

            with st.spinner("Analiza twarzy..."):
                preds = model.predict(np.expand_dims(face_arr, 0), verbose=0)[0]

            pred_class = int(np.argmax(preds))
            confidence = float(np.max(preds))
            emotion_name = class_labels[pred_class]

            st.markdown("<h3 style='font-size: 26px; margin-bottom: 5px;'>Wynik klasyfikacji</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 20px; margin: 0;'><strong>Emocja:</strong> {emotion_name}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 20px; margin: 0;'><strong>Pewność:</strong> {confidence*100:.2f}%</p>", unsafe_allow_html=True)

            st.markdown("<h3 style='font-size: 24px; margin-top: 30px;'>🧠 Rekomendacja</h3>", unsafe_allow_html=True)
            show_typing(get_chatgpt_recommendation(emotion_name))

            if emotion_name == "Smutek":
                st.markdown("<h3 style='font-size: 24px;'>🧩 Żart na poprawę humoru</h3>", unsafe_allow_html=True)
                show_typing(random.choice(sad_jokes))

            st.markdown("<h3 style='font-size: 26px; margin-top: 30px;'>Rozkład wyników</h3>", unsafe_allow_html=True)
            emotions = list(class_labels.values())
            probabilities = preds

            fig, ax = plt.subplots(figsize=(10, 5))
            base_color = np.array([160, 215, 160]) / 255.0
            highlight_color = np.array([46, 125, 50]) / 255.0
            colors = []

            for i, prob in enumerate(probabilities):
                if i == pred_class:
                    colors.append(highlight_color)
                else:
                    intensity = 0.5 + 0.5 * prob
                    color = base_color * intensity
                    color = np.clip(color, 0, 1)
                    colors.append(color)

            sns.barplot(x=emotions, y=probabilities, palette=colors, ax=ax)
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', length=0)
            ax.set_yticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # 🎨 Kolorowanie etykiet osi X
            for tick_label, i in zip(ax.get_xticklabels(), range(len(emotions))):
                if i == pred_class:
                    tick_label.set_color('#111111')
                    tick_label.set_fontweight('bold')
                else:
                    tick_label.set_color('gray')

            for i, prob in enumerate(probabilities):
                ax.text(i, prob + 0.02, f"{prob*100:.1f}%", ha='center', fontsize=14)

            sns.despine(left=True, bottom=True)
            st.pyplot(fig)

            st.markdown("<h3 style='font-size: 24px;'>📊 Grad‑CAM (wyjaśnienie)</h3>", unsafe_allow_html=True)
            if last_conv:
                heatmap = make_gradcam_heatmap(face_arr, model, last_conv, pred_class)
                overlay = overlay_heatmap(heatmap, (face_arr * 255).astype("uint8"))
                st.image(overlay, caption="Grad‑CAM overlay", width=350)
                st.markdown("**Legenda:** Czerwone i żółte obszary pokazują, które części twarzy miały największy wpływ na rozpoznanie tej emocji.")
            else:
                st.warning("Model nie posiada warstwy Conv2D – Grad‑CAM niedostępny.")

    except Exception as e:
        st.error(f"Błąd przetwarzania obrazu:\n{e}")

st.markdown("---")
st.caption("© 2025 Emotion Detector App by Neuronauci")
