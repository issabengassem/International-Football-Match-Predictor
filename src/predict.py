import pandas as pd
import joblib


# Load saved artifacts
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")

team_win_rate = joblib.load("models/team_win_rate.pkl")
team_avg_goals_scored = joblib.load("models/team_avg_goals_scored.pkl")
team_avg_goals_conceded = joblib.load("models/team_avg_goals_conceded.pkl")


def predict_match(home_team, away_team, tournament, country, neutral, year):
    data = pd.DataFrame(
        {
            "home_team": [home_team],
            "away_team": [away_team],
            "tournament": [tournament],
            "country": [country],
            "neutral": [neutral],
            "year": [year],
            "home_team_win_rate": [team_win_rate.get(home_team, 0.5)],
            "away_team_win_rate": [team_win_rate.get(away_team, 0.5)],
            "home_team_avg_goals_scored": [team_avg_goals_scored.get(home_team, 1.3)],
            "away_team_avg_goals_scored": [team_avg_goals_scored.get(away_team, 1.3)],
            "home_team_avg_goals_conceded": [team_avg_goals_conceded.get(home_team, 1.3)],
            "away_team_avg_goals_conceded": [team_avg_goals_conceded.get(away_team, 1.3)],
        }
    )

    data_encoded = pd.get_dummies(
        data,
        columns=["home_team", "away_team", "tournament", "country"]
    )

    data_encoded = data_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    num_cols = [
        "year",
        "home_team_win_rate",
        "away_team_win_rate",
        "home_team_avg_goals_scored",
        "away_team_avg_goals_scored",
        "home_team_avg_goals_conceded",
        "away_team_avg_goals_conceded",
    ]

    data_encoded[num_cols] = scaler.transform(data_encoded[num_cols])

    prediction = model.predict(data_encoded)[0]
    probabilities = model.predict_proba(data_encoded)[0]

    return prediction, probabilities


if __name__ == "__main__":
    prediction, probabilities = predict_match(
        home_team="Morocco",
        away_team="Brazil",
        tournament="FIFA World Cup",
        country="United States",
        neutral=True,
        year=2026
    )

    print("Prediction:", prediction)
    print("Probabilities:", probabilities)
    print("Classes:", model.classes_)