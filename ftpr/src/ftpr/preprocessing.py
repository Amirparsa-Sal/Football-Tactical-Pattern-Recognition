import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy

class PhaseExtractor:
    DELAY_EVENTS = ['foul won', 'foul committed', 'injury stoppage', 'referee ball-drop', 'half end', 'half start', 'offside', 'substitution']

    HAVING_BALL_EVENTS = [
        'carry', 'ball recovery', 'goal keeper', 'clearance',
        'interception', 'dribble', 'shot', 'pass', 'block'
    ]
    # pass_outcome: Incomplete, Out, Pass Offside
    # ball_receipt_outcome: Incomplete
    # dribble outcome: Incomplete
    
    LOSING_BALL_EVENTS = ['dispossessed', 'error', 'miscontrol']


    IGNORE_EVENTS = [
        'ball receipt*', 'camera on*', 'pressure', 'dribbled past', 'duel', 'bad behaviour',
        'player on', 'player off', 'shield', '50/50', 'starting xi', 'tactical shift', 'own goal for', 'own goal against'
    ]

    def __init__(self, matches_df, events_dir_path) -> None:
        self.matches_df = matches_df
        self.events_dir_path = events_dir_path

    def _find_next_event(self, df, starting_index, team_name):
        i = starting_index
        while i < len(df):
            if df.iloc[i]['team'] == team_name:
                return i
            i += 1
        return i

    def _get_outcome(self, df, index):
        row = df.iloc[index]
        event = row['type'].lower()
        if event == 'ball receipt*':
            return row['ball_receipt_outcome']
        if event == 'dribble':
            return row['dribble_outcome']
        elif event == 'pass':
            return row['pass_outcome']
        return None
    
    def _get_event_final_location(self, df, index):
        row = df.iloc[index]
        event = row['type'].lower()
        if event == 'pass':
            return row['pass_end_location']
        if event == 'shot':
            return row['shot_end_location']
        elif event == 'carry':
            return row['carry_end_location']
        raise ValueError('Event {event} does not have final location!')
    
    def _handle_locations(self, df):
        new_df = df.copy()
        for i in range(len(df) - 1):
            next_row = df.iloc[i + 1]
            row = df.iloc[i]
            event = row['type'].lower()
            if event in ['pass', 'shot', 'carry'] and row[f'{event}_end_location'] != next_row['location']:
                new_df.iat[i, new_df.columns.get_loc(f'{event}_end_location')] = next_row['location']
        return new_df

    def extract_phases(self, team_name, output_dir=None):
        # find the matches related to the team
        self.matches_df['target'] = (self.matches_df['home_team'] == team_name) + (self.matches_df['away_team'] == team_name)
        match_ids = self.matches_df[self.matches_df['target']]['match_id']
        
        result_df = None
        # loop over each match
        phase_id = 0
        for id in tqdm(match_ids):
            # open the match df
            df_dir = os.path.join(self.events_dir_path, f'{id}.csv')
            match_df = pd.read_csv(df_dir)
            if result_df is None:
                result_df = pd.DataFrame(None, columns=match_df.columns)
                result_df = result_df.assign(phase_id=None)

            # delete rows if their event is in ignore_events
            match_df = match_df[~match_df['type'].str.lower().isin(PhaseExtractor.IGNORE_EVENTS)]

            # Find the first event related to the team
            start_index = self._find_next_event(match_df, 2, team_name)
            current_index = start_index

            while current_index < len(match_df):
                event = match_df.iloc[current_index]['type'].lower()
                team =  match_df.iloc[current_index]['team']

                if event in PhaseExtractor.HAVING_BALL_EVENTS and team == team_name and (event != 'dribble' or  match_df.iloc[current_index]['dribble_outcome'] != 'Incomplete'):
                    current_index += 1

                elif event in PhaseExtractor.HAVING_BALL_EVENTS or event in PhaseExtractor.LOSING_BALL_EVENTS or event in PhaseExtractor.DELAY_EVENTS:
                    if start_index != current_index:
                        phase_df = self._handle_locations(match_df.iloc[start_index:current_index])
                        phase_df = phase_df.assign(phase_id=phase_id)
                        phase_id += 1
                        result_df = pd.concat((result_df, phase_df))

                    start_index = self._find_next_event(match_df, current_index + 1, team_name)
                    current_index = start_index

                else:
                    raise NotImplementedError(f'This type of event ({event}) is not implemented! (Index: {match_df.iloc[current_index]["index"]}, file: {df_dir})')
                

        result_df = result_df.reset_index(drop=True)

        if output_dir:
            # create output directory if not exists
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            
            result_df.to_csv(os.path.join(output_dir, f'{team_name}.csv'), index=False)
        
        return result_df

def normalize(dataset):
    assert len(dataset) > 0
    n_features = dataset[0].shape[1]
    for series in dataset:
        assert series.shape[1] == n_features
    ds = deepcopy(dataset)
    # Computing max and min
    max_features = np.array([float('-inf') for _ in range(n_features)])
    min_features = np.array([float('inf') for _ in range(n_features)])
    for serie in ds:
        max_features = np.maximum(max_features, np.max(serie, axis=0))
        min_features = np.minimum(min_features, np.min(serie, axis=0))
        
    # Normalizing the series
    for i, serie in enumerate(ds):
        ds[i] = (serie - min_features) / (max_features - min_features)
    return ds

def z_normalize(dataset):
    assert len(dataset) > 0
    n_features = dataset[0].shape[1]
    for series in dataset:
        assert series.shape[1] == n_features
    ds = deepcopy(dataset)
    # Computing max and min
    values = [[] for _ in range(n_features)]
    for serie in ds:
        for i in range(n_features):
            values[i].extend(serie[:, i])
    
    means = np.array([np.mean(value) for value in values])
    stds = np.array([np.std(value) for value in values])
    
    # Normalizing the series
    for i, serie in enumerate(ds):
        ds[i] = (serie - means) / stds
    return ds