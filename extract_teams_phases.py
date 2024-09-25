import pandas as pd
from ftpr.preprocessing import PhaseExtractor
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--matches', type=str, default=os.path.join('data', 'matches.csv'))
parser.add_argument('--events', type=str, default=os.path.join('data', 'events_simplified'))
parser.add_argument('--output-dir', type=str, default=os.path.join('data', 'team_phases'))
args = parser.parse_args()

if not os.path.exists(args.matches):
    raise FileNotFoundError('Matches csv not found!')

if not os.path.exists(args.events):
    raise FileNotFoundError('Events directory not found!')

matches_df = pd.read_csv(args.matches)

teams = set(matches_df['home_team'])

ext = PhaseExtractor(matches_df, args.events)

for team in teams:
    if os.path.exists(os.path.join(args.output_dir, f'{team}.csv')):
        print(team, 'Passed!')
    else:
        print(f'Extracting {team} data...')
        ext.extract_phases(team, output_dir=args.output_dir)