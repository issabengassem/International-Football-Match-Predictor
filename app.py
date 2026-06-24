import streamlit as st
import pandas as pd
from src.predict import predict_match

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #16a34a, #0ea5e9);
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.card {
    padding: 1.5rem;
    border-radius: 18px;
    background-color: #ffffff;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.prediction {
    font-size: 2rem;
    font-weight: 800;
    color: #16a34a;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>⚽ International Football Match Predictor</h1>
    <p>Predict Home Win, Draw, or Away Win using historical international football data.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Match Information")

    home_team = st.text_input("Home Team", value="Morocco")
    away_team = st.text_input("Away Team", value="Brazil")
    tournament = st.text_input("Tournament", value="FIFA World Cup")
    country = st.text_input("Match Country", value="United States")
    neutral = st.checkbox("Neutral Venue", value=True)
    year = st.number_input("Year", min_value=1900, max_value=2100, value=2026)

    predict_button = st.button("Predict Match Outcome", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if predict_button:
        prediction, probabilities = predict_match(
            home_team=home_team,
            away_team=away_team,
            tournament=tournament,
            country=country,
            neutral=neutral,
            year=year
        )

        st.markdown(
            f'<div class="prediction">{prediction}</div>',
            unsafe_allow_html=True
        )

        classes = ["Away Win", "Draw", "Home Win"]

        prob_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        })

        st.bar_chart(
            prob_df.set_index("Outcome")
        )

        for cls, prob in zip(classes, probabilities):
            st.write(f"**{cls}:** {prob:.2%}")
    else:
        st.info("Enter match information and click predict.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Model: Logistic Regression | Features: team strength, goals scored, goals conceded, tournament, venue, year")