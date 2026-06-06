"""
prediction.py
"""

import pickle
import pandas as pd


def load_files():
    files = [
        "winner_predictor.pkl",
        "team_encoder.pkl",
        "venue_encoder.pkl",
        "winner_encoder.pkl",
    ]
    objs = []
    for file in files:
        with open(file, "rb") as f:
            objs.append(pickle.load(f))

    return objs


def get_user_input():
    print("\nEnter Match Details")
    print("-" * 40)
    team1 = input("Team 1: ").strip()
    team2 = input("Team 2: ").strip()
    venue = input("Venue: ").strip()
    home_team = input("Home Team: ").strip()
    return team1, team2, venue, home_team


def predict_winner(model,team_enc,venue_enc,winner_enc, team1, team2,venue, home_team,):
    try:
        x = pd.DataFrame(
            [[
                team_enc.transform([team1])[0],
                team_enc.transform([team2])[0],
                venue_enc.transform([venue])[0],
                int(team1 == home_team),
            ]],
            columns=[
                "team1_enc",
                "team2_enc",
                "venue_enc",
                "is_home",
            ],
        )

    except ValueError as e:
        print("\nError")
        print("-" * 40)
        print("Unknown team or venue found.")
        print(e)
        return

    y = model.predict(x)[0]
    winner = winner_enc.inverse_transform([y])[0]

    print("\nPrediction")
    print("-" * 40)
    print(f"Predicted Winner: {winner}")


def main():
    model, team_enc, venue_enc, winner_enc = load_files()
    team1, team2, venue, home_team = get_user_input()
    predict_winner(model, team_enc, venue_enc, winner_enc,team1,team2,venue, home_team,)


if __name__ == "__main__":
    main()