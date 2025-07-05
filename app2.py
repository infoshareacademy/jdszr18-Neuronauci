import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random


model = tf.keras.models.load_model("emotion_cnn_model.h5")


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


recommendations = {
    'angry': [
        "Od dłuższego czasu wydajesz się być zdenerwowany – spróbuj ćwiczeń oddechowych.",
        "Wyglądasz na podenerwowanego – zjedz coś słodkiego, może baton poprawi Ci humor?",
        "Złość to naturalna emocja – idź na szybki spacer i rozładuj napięcie."
    ],
    'contempt': [
        "Wygląda na to, że coś Cię irytuje – zrób sobie chwilę przerwy.",
        "Twoja twarz wyraża zniecierpliwienie – posłuchaj ulubionej muzyki.",
        "Masz dość? Może czas na detoks od ekranów i odrobina natury?"
    ],
    'disgust': [
        "Wydajesz się zniechęcony – może obejrzysz filmik z małymi kotkami?",
        "Twoja twarz zdradza obrzydzenie – zmień otoczenie lub zaparz aromatyczną herbatę.",
        "Czasem wystarczy 5 minut ciszy, żeby poczuć się lepiej – spróbuj!"
    ],
    'fear': [
        "Od dłuższego czasu wydajesz się być zestresowany – spróbuj ćwiczeń oddechowych.",
        "Czujesz niepokój? Zadzwoń do bliskiej osoby – rozmowa może pomóc.",
        "Lęk nie musi Cię definiować – wypróbuj aplikację z medytacją na 3 minuty."
    ],
    'happiness': [
        "Emanujesz szczęściem – dziel się tym nastrojem z innymi!",
        "Cieszysz się – zrób zdjęcie i zapisz tę chwilę.",
        "Wspaniale Cię widzieć w takim nastroju – może mała celebracja?"
    ],
    'neutral': [
        "Twój nastrój jest stabilny – to dobry moment na coś przyjemnego.",
        "Wyglądasz spokojnie – może czas na drobną przyjemność?",
        "Neutralność to też emocja – zrób coś spontanicznego!"
    ],
    'sadness': [
        "Wyglądasz na smutnego – nie jesteś sam/a, porozmawiaj z kimś bliskim.",
        "Masz gorszy dzień? Zrób sobie ciepłą herbatę i włącz ulubiony film.",
        "Smutek to część życia – przytul się do poduszki i oddychaj głęboko."
    ],
    'surprise': [
        "Zaskoczenie? Zanotuj to, może to ważne!",
        "Wyglądasz na zaskoczonego – coś Cię zaskoczyło pozytywnie?",
        "Zrób screena i podziel się tą chwilą – zaskoczenia są warte zapamiętania."
    ]
}


sad_jokes = [
    "Jak nazywają się ulubione chipsy hydraulika? Kranczips!",
    "Co ile miesięcy chemik jeździ na wakacje? CO2.",
    "Dlaczego długopisy nie chodzą do szkoły? Bo się wypisały.",
    "Co mówi zero do ósemki? Fajny pasek!",
    "Co robi detektyw podczas czytania książki? Śledzi tekst.",
    "Dlaczego komputerowi nigdy nie jest zimno? Bo ma Windows.",
    "Dlaczego programista nie lubi natury? Za dużo bugów.",
    "Co robi informatyk, gdy jest głodny? Szuka pliku cookie.",
]


st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px 40px;
        border-radius: 15px;
        font-family: Arial, sans-serif;
    }
    h1 {
        color: #4B0082;
        font-weight: 900;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

        # POTENCJALNIE DO ZMIANY "Emotion Detector" NA "EmoTrack"
st.title("Emotion Detector 🤖✨")
st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", width=120)
st.write("Prześlij zdjęcie twarzy, a model spróbuje wykryć emocję.")


st.sidebar.title("Opcje")
st.sidebar.write("""
- Aplikacja do wykrywania emocji.
- Model klasyfikuje 8 podstawowych emocji na podstawie zdjęcia twarzy.
- Wgraj zdjęcie w formacie JPG lub PNG.
- Otrzymasz wynik klasyfikacji, rekomendacje i żarty na poprawę humoru.
"""
"""
> Aby uruchomić aplikację:
> 1. Przejdź do folderu z plikiem app.py  
> 2. Wpisz w terminalu: `streamlit run app.py`
""")


uploaded_file = st.file_uploader("Wgraj obraz (jpg/png)", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Wgrane zdjęcie", use_column_width=True)

        # Przetwarzanie obrazu
        img = image.resize((75, 75))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Model analizuje emocje...'):
            prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        emotion_name = class_labels[predicted_class]
        emotion_key = emotion_name.lower()


        st.subheader("Wynik klasyfikacji:")
        st.write(f"**Emocja**: {emotion_name.capitalize()}")
        st.write(f"**Pewność**: {confidence * 100:.2f}%")

        st.subheader("🧠 Rekomendacja:")
        chosen_reco = random.choice(recommendations.get(emotion_key, ["Brak rekomendacji."]))
        st.info(chosen_reco)


        if emotion_key == 'sadness':
            st.subheader("🧩 Żart na poprawę humoru:")
            st.success(random.choice(sad_jokes))

        st.subheader("🔎 Rozkład prawdopodobieństwa:")
        for i, prob in enumerate(prediction[0]):
            label = class_labels[i]
            percent = float(prob * 100)
            bar = "█" * int(percent // 4)
            space = " " * (25 - len(bar))
            st.write(f"**{label.capitalize():<10}** | {bar}{space} {percent:5.2f}%")

    except Exception as e:
        st.error(f"Błąd: Nie udało się przetworzyć obrazu.\n{e}")



st.markdown("---")
st.markdown("© 2025 Emotion Detector App by Neuronauci. All rights reserved.")


