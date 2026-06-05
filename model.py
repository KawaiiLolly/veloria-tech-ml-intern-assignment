"""
    model.py
"""

import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "match_data.csv"


def load_data():
    """Load and clean dataset"""

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "team1",
        "team2",
        "winner",
        "venue",
        "home_team",
    ]

    df = df.dropna(subset=required_columns)

    return df


def create_features(df):
    """Create model features"""

    team_encoder = LabelEncoder()
    venue_encoder = LabelEncoder()
    winner_encoder = LabelEncoder()

    all_teams = pd.concat(
        [
            df["team1"],
            df["team2"]
        ]
    ).unique()

    team_encoder.fit(all_teams)

    df["team1_enc"] = team_encoder.transform(
        df["team1"]
    )

    df["team2_enc"] = team_encoder.transform(
        df["team2"]
    )

    df["venue_enc"] = venue_encoder.fit_transform(
        df["venue"]
    )

    df["winner_enc"] = winner_encoder.fit_transform(
        df["winner"]
    )

    df["is_home"] = (
        df["team1"] == df["home_team"]
    ).astype(int)

    X = df[
        [
            "team1_enc",
            "team2_enc",
            "venue_enc",
            "is_home",
        ]
    ]

    y = df["winner_enc"]

    return (
        X,
        y,
        team_encoder,
        venue_encoder,
        winner_encoder,
    )


def train_model(X, y):
    """Train Random Forest"""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return (
        model,
        predictions,
        y_test,
        accuracy,
    )


def evaluate_model(
    predictions,
    y_test,
    winner_encoder,
):
    """Print model metrics"""

    print("\nModel Evaluation")
    print("-" * 50)

    print(
        f"Accuracy: {accuracy_score(y_test, predictions):.2f}"
    )

    labels = sorted(
        set(y_test) |
        set(predictions)
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=winner_encoder.inverse_transform(
                labels
            ),
            zero_division=0,
        )
    )


def show_feature_importance(
    model,
    feature_names,
):
    """Display feature importance"""

    print("\nFeature Importance")
    print("-" * 50)

    for feature, score in zip(
        feature_names,
        model.feature_importances_,
    ):
        print(
            f"{feature:<15}: {score:.4f}"
        )


def save_model(
    model,
    team_encoder,
    venue_encoder,
    winner_encoder,
):
    """Save model and encoders"""

    with open(
        "./models/winner_predictor.pkl",
        "wb",
    ) as f:
        pickle.dump(model, f)

    with open(
        "./models/team_encoder.pkl",
        "wb",
    ) as f:
        pickle.dump(team_encoder, f)

    with open(
        "./models/venue_encoder.pkl",
        "wb",
    ) as f:
        pickle.dump(venue_encoder, f)

    with open(
        "./models/winner_encoder.pkl",
        "wb",
    ) as f:
        pickle.dump(winner_encoder, f)

    print("\nSaved Files")
    print("-" * 50)
    print("winner_predictor.pkl")
    print("team_encoder.pkl")
    print("venue_encoder.pkl")
    print("winner_encoder.pkl")


def main():

    print(
        "\nCricket Match Winner Prediction"
    )

    print("-" * 50)

    df = load_data()

    print(
        f"Total Matches Loaded: {len(df)}"
    )

    (
        X,
        y,
        team_encoder,
        venue_encoder,
        winner_encoder,
    ) = create_features(df)

    (
        model,
        predictions,
        y_test,
        accuracy,
    ) = train_model(X, y)

    print(
        f"\nAccuracy: {accuracy:.2f}"
    )

    evaluate_model(
        predictions,
        y_test,
        winner_encoder,
    )

    show_feature_importance(
        model,
        X.columns,
    )

    save_model(
        model,
        team_encoder,
        venue_encoder,
        winner_encoder,
    )


if __name__ == "__main__":
    main()