from statsbombpy import sb
import os
from tqdm import tqdm
import warnings
import pandas as pd
import pickle
import argparse

warnings.simplefilter("ignore")

selected_columns = [
    'ball_receipt_outcome',
    'carry_end_location',
    'dribble_outcome',
    'foul_won_penalty',
    'id',
    'index',
    'location',
    'match_id',
    'out',
    'pass_angle',
    'pass_assisted_shot_id',
    'pass_cross',
    'pass_end_location',
    'pass_goal_assist',
    'pass_height',
    'pass_length',
    'pass_outcome',
    'pass_recipient',
    'pass_recipient_id',
    'pass_shot_assist',
    'pass_type',
    'play_pattern',
    'player',
    'player_id',
    'position',
    'possession',
    'possession_team',
    'possession_team_id',
    'shot_end_location',
    'shot_freeze_frame',
    'shot_key_pass_id',
    'shot_outcome',
    'shot_statsbomb_xg',
    'shot_type',
    'tactics',
    'team',
    'team_id',
    'timestamp',
    'type'
]

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=os.path.join('..', 'data'))
parser.add_argument('--events-dir', type=str, default='simplified_events')
parser.add_argument('--competition-id', type=int, default=2)
parser.add_argument('--season-id', type=int, default=27)
args = parser.parse_args()

data_dir = args.data_dir
events_dir_name = args.events_dir

matches_df = sb.matches(competition_id=args.competition_id, season_id=args.season_id)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    
matches_df.to_csv(os.path.join(data_dir, 'matches.csv'))

if not os.path.exists(os.path.join(data_dir, events_dir_name)):
    os.mkdir(os.path.join(data_dir, events_dir_name))
    
# Downloading Events
for i in tqdm(range(len(matches_df))):
    row = matches_df.iloc[i]
    match_id = row['match_id']
    output_dir = os.path.join(data_dir, events_dir_name, f'{match_id}.csv')
    if os.path.exists(output_dir):
        continue
    df = sb.events(match_id=match_id)
    # Remove unnecessary columns
    available_columns = [c for c in selected_columns if c in df.columns]
    empty_columns = [c for c in selected_columns if c not in df.columns]
    df = df[available_columns]
    # Sorting the df based on time
    df = df.sort_values(by=['index'], ascending=True)
    # Adding non existed columns
    for c in empty_columns:
        df[c] = ""
    df.to_csv(output_dir)
   
# Storing Players
players = dict()
for i in tqdm(range(len(matches_df))):
    row = matches_df.iloc[i]
    match_id = row['match_id']
    home_team, away_team = row['home_team'], row['away_team']
    df_dir = os.path.join(data_dir, events_dir_name, f'{match_id}.csv')
    df = pd.read_csv(df_dir)
    home_players = set(df[df['team'] == home_team]['player'].dropna())
    players[home_team] = players.get(home_team, set()).union(home_players)
    away_players = set(df[df['team'] == away_team]['player'].dropna())
    players[away_team] = players.get(away_team, set()).union(away_players)
    
with open(os.path.join(data_dir, 'players.pkl'), 'wb') as f:
    pickle.dump(players, f)