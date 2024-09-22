import pandas as pd
from ftpr.preprocessing import PhaseExtractor
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--matches', type=str, default='./data/matches.csv')
parser.add_argument('--events', type=str, default='./data/events_simplified')
args = parser.parse_args()

if not os.path.exists(args.matches):
    raise FileNotFoundError('Matches csv not found!')

if not os.path.exists(args.events):
    raise FileNotFoundError('Events directory not found!')

matches_df = pd.read_csv(args.matches)

teams = set(matches_df['home_team'])

ext = PhaseExtractor(matches_df, args.events)

for team in teams:
    if os.path.exists(f'./data/team_phases/{team}.csv'):
        print(team, 'Passed!')
    else:
        print(f'Extracting {team} data...')
        ext.extract_phases(team, output_dir='./data/team_phases')