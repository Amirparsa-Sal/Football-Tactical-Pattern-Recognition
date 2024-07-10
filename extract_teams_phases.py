import pandas as pd
from ftpr.preprocessing import PhaseExtractor
import os

matches_df = pd.read_csv('./data/matches.csv')

teams = set(matches_df['home_team'])

ext = PhaseExtractor(matches_df, './data/events_simplified')

for team in teams:
    if os.path.exists(f'./data/team_phases/{team}.csv'):
        print(team, 'Passed!')
    else:
        print(f'Extracting {team} data...')
        ext.extract_phases(team, output_dir='./data/team_phases')