import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. Load data
# =========================

df = pd.read_csv("data/raw/results.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Remove matches without scores
df = df.dropna(subset=["home_score", "away_score"])

# Create year feature
df["year"] = df["date"].dt.year


# =========================
# 2. Create target
# =========================

df["result"] = "Draw"

df.loc[df["home_score"] > df["away_score"], "result"] = "Home Win"
df.loc[df["home_score"] < df["away_score"], "result"] = "Away Win"


# =========================
# 3. Training data for historical features
# =========================

train_df = df[df["year"] <= 2020].copy()


# =========================
# 4. Team win rate
# =========================

team_stats = {}

for _, row in train_df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    team_stats.setdefault(home, {"wins": 0, "matches": 0, "goals_scored": 0, "goals_conceded": 0})
    team_stats.setdefault(away, {"wins": 0, "matches": 0, "goals_scored": 0, "goals_conceded": 0})

    team_stats[home]["matches"] += 1
    team_stats[away]["matches"] += 1

    team_stats[home]["goals_scored"] += row["home_score"]
    team_stats[away]["goals_scored"] += row["away_score"]

    team_stats[home]["goals_conceded"] += row["away_score"]
    team_stats[away]["goals_conceded"] += row["home_score"]

    if row["result"] == "Home Win":
        team_stats[home]["wins"] += 1
    elif row["result"] == "Away Win":
        team_stats[away]["wins"] += 1


team_win_rate = {
    team: stats["wins"] / stats["matches"]
    for team, stats in team_stats.items()
}

team_avg_goals_scored = {
    team: stats["goals_scored"] / stats["matches"]
    for team, stats in team_stats.items()
}

team_avg_goals_conceded = {
    team: stats["goals_conceded"] / stats["matches"]
    for team, stats in team_stats.items()
}


# =========================
# 5. Add engineered features
# =========================

df["home_team_win_rate"] = df["home_team"].map(team_win_rate).fillna(0.5)
df["away_team_win_rate"] = df["away_team"].map(team_win_rate).fillna(0.5)

df["home_team_avg_goals_scored"] = df["home_team"].map(team_avg_goals_scored).fillna(1.3)
df["away_team_avg_goals_scored"] = df["away_team"].map(team_avg_goals_scored).fillna(1.3)

df["home_team_avg_goals_conceded"] = df["home_team"].map(team_avg_goals_conceded).fillna(1.3)
df["away_team_avg_goals_conceded"] = df["away_team"].map(team_avg_goals_conceded).fillna(1.3)


# =========================
# 6. Build features and target
# =========================

features = [
    "home_team",
    "away_team",
    "tournament",
    "country",
    "neutral",
    "year",
    "home_team_win_rate",
    "away_team_win_rate",
    "home_team_avg_goals_scored",
    "away_team_avg_goals_scored",
    "home_team_avg_goals_conceded",
    "away_team_avg_goals_conceded",
]

X = df[features]
y = df["result"]

X_encoded = pd.get_dummies(
    X,
    columns=[
        "home_team",
        "away_team",
        "tournament",
        "country"
    ]
)


# =========================
# 7. Chronological split
# =========================

train_mask = df["year"] <= 2020
test_mask = df["year"] > 2020

X_train = X_encoded[train_mask].copy()
X_test = X_encoded[test_mask].copy()

y_train = y[train_mask]
y_test = y[test_mask]


# =========================
# 8. Scale numerical columns
# =========================

num_cols = [
    "year",
    "home_team_win_rate",
    "away_team_win_rate",
    "home_team_avg_goals_scored",
    "away_team_avg_goals_scored",
    "home_team_avg_goals_conceded",
    "away_team_avg_goals_conceded",
]

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# =========================
# 9. Train Logistic Regression
# =========================

model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 10. Evaluate
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# =========================
# 11. Save model artifacts
# =========================

joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X_encoded.columns.tolist(), "models/model_columns.pkl")

joblib.dump(team_win_rate, "models/team_win_rate.pkl")
joblib.dump(team_avg_goals_scored, "models/team_avg_goals_scored.pkl")
joblib.dump(team_avg_goals_conceded, "models/team_avg_goals_conceded.pkl")

print("Model and artifacts saved successfully.")