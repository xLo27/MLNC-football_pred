import pandas as pd
import os


def calculate_player_elo(data):

# def process_multiple_files(files):
#     all_data = []

#     for filename in files:
#         if filename.endswith('.csv'):
#             data = pd.read_csv(filename)
#             all_data.append(data)

#     combined_data = pd.concat(all_data, ignore_index=True)
#     return combined_data

# data = process_multiple_files(["2122_players.csv", "2223_players.csv", "2324_players.csv", "2425_players.csv"])
    print(data.head(), data.tail())
    def get_positions(position_str):
        return position_str.split(',')

    data['Positions'] = data['Pos.'].apply(get_positions)
    data['Positions'] = data['Pos.'].apply(get_positions)
    data['KP'] = pd.to_numeric(data['KP'], errors='coerce')
    data['G+A'] = pd.to_numeric(data['G+A'], errors='coerce')
    data['Gls'] = pd.to_numeric(data['Gls'], errors='coerce')
    data['Ast'] = pd.to_numeric(data['Ast'], errors='coerce')
    data['Tkl'] = pd.to_numeric(data['Tkl'], errors='coerce')
    data['Clr'] = pd.to_numeric(data['Clr'], errors='coerce')
    data['Recov'] = pd.to_numeric(data['Recov'], errors='coerce')
    data['Sh'] = pd.to_numeric(data['Sh'], errors='coerce')
    data['Pass%'] = pd.to_numeric(data['Sh'], errors='coerce')
    data['Int'] = pd.to_numeric(data['Int'], errors='coerce')
    data['Blocks'] = pd.to_numeric(data['Blocks'], errors='coerce')


    initial_elo = 1500
    elo_ratings = {player: initial_elo for player in data['Player'].unique()}
    data['Primary Position'] = data['Positions'].apply(lambda pos_list: pos_list[0] if pos_list else None)

    position_weights = {
        'FW': 1.2,
        'MF': 1.0,
        'DF': 1.0,
        'GK': 0.7
    }

    max_values = {
        'Gls': data['Gls'].max(),
        'Ast': data['Ast'].max(),
        'KP': data['KP'].max(),
        'G+A': (data['Gls'] + data['Ast']).max(),
        'Pass%': data['Pass%'].max(),
        'Tkl': data['Tkl'].max(),
        'Int': data['Int'].max(),
        'Clr': data['Clr'].max(),
        'Blocks': data['Blocks'].max(),
        'Recov': data['Recov'].max(),
    }

    performance_max_weights = {
        'FW': (
            max_values['Gls'] * 5
            + max_values['Ast'] * 4
            + max_values['KP'] * 1.0
        ),
        'MF': (
            max_values['G+A'] * 3
            + max_values['KP'] * 2
            + max_values['Pass%'] * 1.0
            + max_values['Tkl'] * 1.5
            + max_values['Int'] * 1.0
        ),
        'DF': (
            max_values['Tkl'] * 3
            + max_values['Clr'] * 2
            + max_values['Int'] * 2
            + max_values['Pass%'] * 0.5
            + max_values['Blocks'] * 2
        ),
        'GK': (
            max_values['Blocks'] * 3
            + max_values['Pass%'] * 0.2
            + max_values["Recov"] * 2
        ),
    }

    performance_weights = {
        'FW': lambda row: (
            (row['Gls'] * 5 + row['Ast'] * 4 + row['KP'] * 1.0)
            / performance_max_weights['FW']
        ),
        'MF': lambda row: (
            (row['G+A'] * 3 + row['KP'] * 2 + row['Pass%'] * 1.0 + row['Tkl'] * 1.5 + row['Int'] * 1.0)
            / performance_max_weights['MF']
        ),
        'DF': lambda row: (
            (row['Tkl'] * 3 + row['Clr'] * 2 + row['Int'] * 2 + row['Pass%'] * 0.5 + row['Blocks'] * 2)
            / performance_max_weights['DF']
        ),
        'GK': lambda row: (
            (row['Blocks'] * 3 + row['Pass%'] * 0.2 + row['Recov'] * 2)
            / performance_max_weights['GK']
        ),
    }
    def calculate_elo_change(player_elo, opponent_elo, result, k_factor=32, performance=1.0):
        expected_score = 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
        return k_factor * (result - expected_score) * performance

    def update_elo_ratings(data, elo_ratings):
        for player in elo_ratings.keys():
            player_matches = data[data['Player'] == player]

            for _, match in player_matches.iterrows():
                if match['Result'].startswith('W'):
                    result = 1
                elif match['Result'].startswith('L'):
                    result = 0
                else:
                    result = 0.5

                team_players = data[(data['Team'] == match['Team']) & (data['Date'] == match['Date'])]
                team_elo = team_players['Player'].map(elo_ratings).mean()

                opponent_players = data[(data['Team'] == match['Opp']) & (data['Date'] == match['Date'])]
                opponent_elo = opponent_players['Player'].map(elo_ratings).mean()


                primary_position = match['Primary Position']
                performance_score = performance_weights.get(primary_position, lambda x : 1.0)(match)

                elo_change = calculate_elo_change(
                    elo_ratings[player], opponent_elo, result, performance=performance_score
                )
                elo_ratings[player] += elo_change * 10

    update_elo_ratings(data, elo_ratings)

    ranked_players = pd.DataFrame({
        'Player': list(elo_ratings.keys()),
        'Team': [data[data['Player'] == player].iloc[0]['Team'] for player in elo_ratings.keys()],
        'Position': [data[data['Player'] == player].iloc[0]['Primary Position'] for player in elo_ratings.keys()],
        'Elo': list(elo_ratings.values())
    }).sort_values(by='Elo', ascending=False).reset_index(drop=True)

    ranked_players['Rank'] = ranked_players.index + 1

    ranked_players.to_csv('ranked_players.csv', index=False)
    print("saved")