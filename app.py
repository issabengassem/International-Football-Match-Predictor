import streamlit as st
from src.predict import predict_match

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ International Football Match Predictor")

st.write(
    "Predict the outcome of an international football match using Machine Learning."
)

# Inputs

home_team = st.text_input(
    "Home Team",
    value="Morocco"
)

away_team = st.text_input(
    "Away Team",
    value="Brazil"
)

tournament = st.text_input(
    "Tournament",
    value="FIFA World Cup"
)

country = st.text_input(
    "Country",
    value="United States"
)

neutral = st.checkbox(
    "Neutral Venue",
    value=True
)

year = st.number_input(
    "Year",
    min_value=1900,
    max_value=2100,
    value=2026
)

# Prediction button

if st.button("Predict Match"):

    prediction, probabilities = predict_match(
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        country=country,
        neutral=neutral,
        year=year
    )

    st.success(f"Prediction: {prediction}")

    st.subheader("Probabilities")

    classes = [
        "Away Win",
        "Draw",
        "Home Win"
    ]

    for cls, prob in zip(classes, probabilities):
        st.write(f"{cls}: {prob:.2%}")