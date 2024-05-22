import pandas as pd
from preprocessing import PhaseExtractor

team_name = 'Manchester City'

matches_df = pd.read_csv('./data/matches.csv')

ext = PhaseExtractor(matches_df, './data/events_simplified')

ext.extract_phases(team_name, output_dir='./data/team_phases')