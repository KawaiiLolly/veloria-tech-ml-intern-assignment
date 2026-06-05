"""
    prediction.py
"""

import pickle
import pandas as pd

def load_files():
    """Load model and encoders"""

    with open("winner_predictor.pkl", "rb") as f:
        model = pickle.load(f)

    with open("team_encoder.pkl", "rb") as f:
        team_encoder = pickle.load(f)

    with open("venue_encoder.pkl", "rb") as f:
        venue_encoder = pickle.load(f)

    with open("winner_encoder.pkl", "rb") as f:
        winner_encoder = pickle.load(f)

    return (
        model,
        team_encoder,
        venue_encoder,
        winner_encoder,
    )


def get_user_input():
    print("\nEnter Match Details")
    print("-" * 40)

    team1 = input("Team 1: ").strip()
    team2 = input("Team 2: ").strip()
    venue = input("Venue: ").strip()
    home_team = input("Home Team: ").strip()

    return (team1,team2,venue,home_team)


def predict_winner(
    model,
    team_encoder,
    venue_encoder,
    winner_encoder,
    team1,
    team2,
    venue,
    home_team,
):
    """Predict match winner"""

    try:
        team1_enc = team_encoder.transform([team1])[0]
        team2_enc = team_encoder.transform([team2])[0]
        venue_enc = venue_encoder.transform([venue])[0]

    except ValueError as e:
        print("\nError:")
        print(
            "Unknown team or venue found."
        )
        print(e)
        return

    is_home = int(team1 == home_team)

    x_new = pd.DataFrame(
        [
            [
                team1_enc,
                team2_enc,
                venue_enc,
                is_home,
            ]
        ],
        columns=[
            "team1_enc",
            "team2_enc",
            "venue_enc",
            "is_home",
        ],
    )

    prediction = model.predict(x_new)[0]

    winner = winner_encoder.inverse_transform([prediction])[0]

    print("\nPrediction")
    print("-" * 40)
    print(f"Predicted Winner: {winner}")


def main():

    (model,team_encoder,venue_encoder,winner_encoder) = load_files()

    (team1, team2,venue,home_team,) = get_user_input()

    predict_winner(
        model,
        team_encoder,
        venue_encoder,
        winner_encoder,
        team1,
        team2,
        venue,
        home_team,
    )


if __name__ == "__main__":
    main()