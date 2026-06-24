import streamlit as st
import pandas as pd
from src.predict import predict_match

# =========================
# Page config
# =========================

st.set_page_config(
    page_title="World Cup Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# =========================
# Styling
# =========================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-card {
    background-color: white;
    color: #111827;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.hero {
    text-align: center;
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #16a34a, #0284c7);
    color: white;
    margin-bottom: 30px;
}

.prediction-box {
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    background: #dcfce7;
    color: #166534;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load teams from dataset
# =========================

@st.cache_data
def load_options():
    df = pd.read_csv("data/raw/results.csv")

    teams = sorted(
        set(df["home_team"].dropna().unique())
        | set(df["away_team"].dropna().unique())
    )

    tournaments = sorted(df["tournament"].dropna().unique())
    countries = sorted(df["country"].dropna().unique())

    return teams, tournaments, countries


teams, tournaments, countries = load_options()

# =========================
# Header
# =========================

st.markdown("""
<div class="hero">
    <h1>⚽ International Football Match Predictor</h1>
    <p>Predict match outcomes using historical football results and machine learning.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# Layout
# =========================

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("🏟️ Match Details")

    home_team = st.selectbox(
        "Home Team",
        teams,
        index=teams.index("Morocco") if "Morocco" in teams else 0
    )

    away_team = st.selectbox(
        "Away Team",
        teams,
        index=teams.index("Brazil") if "Brazil" in teams else 1
    )

    tournament = st.selectbox(
        "Tournament",
        tournaments,
        index=tournaments.index("FIFA World Cup") if "FIFA World Cup" in tournaments else 0
    )

    country = st.selectbox(
        "Match Country",
        countries,
        index=countries.index("United States") if "United States" in countries else 0
    )

    neutral = st.checkbox("Neutral Venue", value=True)

    year = st.number_input(
        "Year",
        min_value=1900,
        max_value=2100,
        value=2026
    )

    predict_button = st.button(
        "Predict Match Outcome",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("📊 Prediction")

    if predict_button:

        if home_team == away_team:
            st.error("Home team and away team must be different.")
        else:
            prediction, probabilities = predict_match(
                home_team=home_team,
                away_team=away_team,
                tournament=tournament,
                country=country,
                neutral=neutral,
                year=year
            )

            st.markdown(
                f'<div class="prediction-box">{prediction}</div>',
                unsafe_allow_html=True
            )

            classes = ["Away Win", "Draw", "Home Win"]

            prob_df = pd.DataFrame({
                "Outcome": classes,
                "Probability": probabilities
            })

            st.bar_chart(prob_df.set_index("Outcome"))

            st.write("### Probability Breakdown")

            for cls, prob in zip(classes, probabilities):
                st.write(f"**{cls}:** {prob:.2%}")

    else:
        st.info("Choose match details and click predict.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer
# =========================

st.markdown("---")
st.caption(
    "Model: Logistic Regression | Features: team win rate, goals scored, goals conceded, tournament, venue, and year."
)