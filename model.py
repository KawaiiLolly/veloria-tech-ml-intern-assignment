"""
model.py
"""

import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "match_data.csv"


def load_data():
    cols = ["team1", "team2", "winner", "venue", "home_team"]
    return pd.read_csv(DATA_FILE).dropna(subset=cols)


def create_features(df):
    team_enc = LabelEncoder()
    venue_enc = LabelEncoder()
    winner_enc = LabelEncoder()

    teams = pd.concat([df["team1"], df["team2"]]).unique()
    team_enc.fit(teams)

    df["team1_enc"] = team_enc.transform(df["team1"])
    df["team2_enc"] = team_enc.transform(df["team2"])
    df["venue_enc"] = venue_enc.fit_transform(df["venue"])
    df["winner_enc"] = winner_enc.fit_transform(df["winner"])
    df["is_home"] = (df["team1"] == df["home_team"]).astype(int)

    x = df[["team1_enc", "team2_enc", "venue_enc", "is_home"]]
    y = df["winner_enc"]

    return x, y, team_enc, venue_enc, winner_enc


def train_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    return model, preds, y_test, acc


def evaluate_model(preds, y_test, winner_enc):
    print("\nModel Evaluation")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

    labels = sorted(set(y_test) | set(preds))

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test,
            preds,
            labels=labels,
            target_names=winner_enc.inverse_transform(labels),
            zero_division=0,
        )
    )


def show_feature_importance(model, features):
    print("\nFeature Importance")
    print("-" * 50)

    for feature, score in zip(features, model.feature_importances_):
        print(f"{feature:<12}: {score:.4f}")


def save_model(model, team_enc, venue_enc, winner_enc):
    files = {
        "winner_predictor.pkl": model,
        "team_encoder.pkl": team_enc,
        "venue_encoder.pkl": venue_enc,
        "winner_encoder.pkl": winner_enc,
    }

    for name, obj in files.items():
        with open(f"./models/{name}", "wb") as f:
            pickle.dump(obj, f)

    print("\nSaved Files")
    print("-" * 50)

    for name in files:
        print(name)


def main():
    print("\nCricket Match Winner Prediction")
    print("-" * 50)

    df = load_data()
    print(f"Total Matches Loaded: {len(df)}")

    x, y, team_enc, venue_enc, winner_enc = create_features(df)

    model, preds, y_test, acc = train_model(x, y)

    print(f"\nAccuracy: {acc:.2f}")

    evaluate_model(
        preds,
        y_test,
        winner_enc,
    )

    show_feature_importance(
        model,
        x.columns,
    )

    save_model(
        model,
        team_enc,
        venue_enc,
        winner_enc,
    )


if __name__ == "__main__":
    main()