import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_data_for_pi(df):
    relevant_cols = ["HomeTeam", "AwayTeam", "Date", "GF_home", "GF_away"]

    df_prepared = df[relevant_cols].copy()

    df_prepared["Date"] = pd.to_datetime(df_prepared["Date"])

    df_prepared = df_prepared.sort_values("Date").reset_index(drop=True)

    return df_prepared


def get_pi_ratings(df, params):
    teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
    keys = list(set(list(df["HomeTeam"]) + list(df["AwayTeam"])))

    pi_dictionary = {f"Home {key}": 0 for key in keys}
    pi_dictionary.update({f"Away {key}": 0 for key in keys})

    c = params[0]
    mu1 = params[1]
    mu2 = params[2]

    home_key = "Home {}"
    away_key = "Away {}"

    df_list = df.values.tolist()

    def exp_goal_diff(c, hr, ar):
        if ar >= 0:
            egda = 10 ** np.abs((ar) / c) - 1
        else:
            egda = -(10 ** np.abs((ar) / c) - 1)
        if hr >= 0:
            egdh = 10 ** np.abs((hr) / c) - 1
        else:
            egdh = -(10 ** np.abs((hr) / c) - 1)
        return egdh - egda

    def get_error(obs_goals, exp_goals):
        return np.abs(obs_goals - exp_goals)

    def get_weighted_error(c, error, obs_goals, exp_goals):
        if exp_goals < obs_goals:
            we1 = c * np.log10((1 + error))
            we2 = -(we1)
        else:
            we1 = -(c * np.log10((1 + error)))
            we2 = -(we1)
        return we1, we2

    def update_ratings(wehome, weaway, hrhome, hraway, arhome, araway, mu1, mu2):
        hrhome_new = hrhome + (wehome * mu1)
        hraway_new = hrhome + (hrhome_new - hrhome) * mu2
        araway_new = araway + (weaway * mu1)
        arhome_new = arhome + (araway_new - araway) * mu2
        return hrhome_new, hraway_new, arhome_new, araway_new

    results = []

    # Process each match
    for _, row in df.iterrows():
        home = row["HomeTeam"]  # HomeTeam
        away = row["AwayTeam"]  # AwayTeam
        date = row["Date"]  # Date
        home_score = row["GF_home"]  # GF_home
        away_score = row["GF_away"]  # GF_away

        h_hr = pi_dictionary[home_key.format(home)]
        h_ar = pi_dictionary[away_key.format(home)]
        a_hr = pi_dictionary[home_key.format(away)]
        a_ar = pi_dictionary[away_key.format(away)]

        egd = exp_goal_diff(c, h_hr, a_ar)
        obs_goals = home_score - away_score
        error = get_error(obs_goals, egd)
        wehome, weaway = get_weighted_error(c, error, obs_goals, egd)

        h_hr_new, h_ar_new, a_hr_new, a_ar_new = update_ratings(
            wehome, weaway, h_hr, h_ar, a_hr, a_ar, mu1, mu2
        )

        pi_dictionary[home_key.format(home)] = h_hr_new
        pi_dictionary[away_key.format(home)] = h_ar_new
        pi_dictionary[home_key.format(away)] = a_hr_new
        pi_dictionary[away_key.format(away)] = a_ar_new

        results.append(
            {
                "Date": date,
                "HomeTeam": home,
                "AwayTeam": away,
                "HomeGoals": home_score,
                "AwayGoals": away_score,
                "f_pi_Home Rating_home": h_hr,
                "f_pi_Away Rating_home": h_ar,
                "f_pi_Home Rating_away": a_hr,
                "f_pi_Away Rating_away": a_ar,
                "f_pi_Exp GD Pi": egd,
                "f_pi_Pi Diff": h_hr - a_ar,
            }
        )

    return pd.DataFrame(results)


# Usage example:
def run_pi_analysis(csv_file, params):
    """
    Run complete PI rating analysis on a dataset
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Prepare data for PI ratings
    df_prepared = prepare_data_for_pi(df)

    # Calculate PI ratings
    df_with_ratings = get_pi_ratings(df_prepared, params)

    return df_with_ratings


def get_team_rankings(df_pi_ratings):
    """
    Get team rankings based on their latest PI ratings
    """
    latest_date = df_pi_ratings["Date"].max()
    latest_ratings = df_pi_ratings[df_pi_ratings["Date"] == latest_date]

    home_ratings = {}
    away_ratings = {}

    # Get all unique teams
    all_teams = set(df_pi_ratings["HomeTeam"].unique()) | set(
        df_pi_ratings["AwayTeam"].unique()
    )

    for team in all_teams:
        home_ratings[team] = 0
        away_ratings[team] = 0

    for _, row in df_pi_ratings.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_ratings[home_team] = row["Home Home Rating"]
        away_ratings[home_team] = row["Home Away Rating"]
        home_ratings[away_team] = row["Away Home Rating"]
        away_ratings[away_team] = row["Away Away Rating"]

    team_ratings = []
    for team in all_teams:
        avg_rating = (home_ratings[team] + away_ratings[team]) / 2
        team_ratings.append(
            {
                "Team": team,
                "Home Rating": home_ratings[team],
                "Away Rating": away_ratings[team],
                "Average Rating": avg_rating,
            }
        )

    rankings_df = pd.DataFrame(team_ratings)
    rankings_df = rankings_df.sort_values(
        "Average Rating", ascending=False
    ).reset_index(drop=True)

    # Add rank column
    rankings_df["Rank"] = rankings_df.index + 1

    # Reorder columns
    rankings_df = rankings_df[
        ["Rank", "Team", "Average Rating", "Home Rating", "Away Rating"]
    ]

    return rankings_df


# Function to plot team rankings
def plot_team_rankings(rankings_df, top_n=20):
    """
    Plot team rankings
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=rankings_df.head(top_n), x="Team", y="Average Rating", palette="viridis"
    )

    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {top_n} Teams by PI Rating")
    plt.tight_layout()

    return plt


# Usage example:
def analyze_team_rankings(pi_ratings_df, plot=True, top_n=20):
    """
    Analyze and display team rankings
    """
    # Get rankings
    rankings = get_team_rankings(pi_ratings_df)

    # Display rankings
    print("\nTeam Rankings:")
    print(rankings.to_string(index=False))

    if plot:
        # Plot rankings
        plt = plot_team_rankings(rankings, top_n)
        plt.show()
        plt = plot_team_rating_history(pi_ratings_df, "Nott'ham Forest")

    return rankings


def get_team_rating_history(df_pi_ratings, team_name):
    """
    Get rating history for a specific team
    """
    team_history = []

    # Get matches where team played
    team_matches = df_pi_ratings[
        (df_pi_ratings["HomeTeam"] == team_name)
        | (df_pi_ratings["AwayTeam"] == team_name)
    ].sort_values("Date")

    for _, match in team_matches.iterrows():
        if match["HomeTeam"] == team_name:
            rating = match["Home Home Rating"]
        else:
            rating = match["Away Home Rating"]

        team_history.append({"Date": pd.to_datetime(match["Date"]), "Rating": rating})

    return pd.DataFrame(team_history)


def plot_team_rating_history(df_pi_ratings, team_name):
    """
    Plot rating history for a specific team with improved date formatting
    """
    import matplotlib.dates as mdates

    history = get_team_rating_history(df_pi_ratings, team_name)

    plt.figure(figsize=(12, 6))

    # Plot the ratings
    plt.plot(history["Date"], history["Rating"], marker="o", markersize=4)

    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format as YYYY-MM

    # Customize the plot
    plt.title(f"{team_name} PI Rating History")
    plt.xlabel("Date")
    plt.ylabel("PI Rating")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    # Add points to show actual matches
    plt.scatter(history["Date"], history["Rating"], color="red", s=30, alpha=0.5)

    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()

    return history


if __name__ == "__main__":
    # Example usage:
    params = [1.0, 0.1, 0.1]  # Example values for c, mu1, mu2
    results = run_pi_analysis("data/features_17_24_ema_span=30.csv", params)
    results.to_csv("check.csv")

    params = [1, 0.1, 0.3]
    pi_ratings_df = run_pi_analysis("data/features_17_24_ema_span=30.csv", params)
    rankings = analyze_team_rankings(pi_ratings_df, plot=True, top_n=30)
    rankings.to_csv("rankings_pi.csv")
