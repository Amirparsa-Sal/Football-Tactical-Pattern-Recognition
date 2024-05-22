import pandas as pd
import os
from tqdm import tqdm

class PhaseExtractor:
    DELAY_EVENTS = ['foul won', 'foul committed', 'injury stoppage', 'referee ball-drop', 'half end', 'half start']

    HAVING_BALL_EVENTS = [
        'ball receipt*', 'carry', 'ball recovery', 'goal keeper', 'clearance',
        'interception', 'dribble', 'shot', 'pass', 'block'
    ]

    LOSING_BALL_EVENTS = ['dispossessed', 'offside', 'error', 'miscontrol']


    IGNORE_EVENTS = [
        'camera on*', 'pressure', 'dribbled past', 'substitution', 'duel', 'bad behaviour', 'player on',
        'player off', 'shield', '50/50', 'starting xi', 'tactical shift', 'own goal for', 'own goal against'
    ]

    def __init__(self, matches_df, events_dir_path) -> None:
        self.matches_df = matches_df
        self.events_dir_path = events_dir_path

    def __find_next_event(self, df, starting_index, team_name):
        i = starting_index
        while i < len(df):
            if df.iloc[i]['team'] == team_name:
                return i
            i += 1
        return i

    def extract_phases(self, team_name, output_dir, verbose=True):
        # find the matches related to the team
        self.matches_df['target'] = (self.matches_df['home_team'] == team_name) + (self.matches_df['away_team'] == team_name)
        match_ids = self.matches_df[self.matches_df['target']]['match_id']
        
        # create output directory if not exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        result_df = None
        phases_ranges = []
        # loop over each match
        phase_id = 0
        for id in tqdm(match_ids):
            # open the match df
            df_dir = os.path.join(self.events_dir_path, f'{id}.csv')
            match_df = pd.read_csv(df_dir)
            if result_df is None:
                result_df = pd.DataFrame(None, columns=match_df.columns)
                result_df = result_df.assign(phase_id=None)
                print(result_df)

            # delete rows if their event is in ignore_events
            match_df = match_df[~match_df['type'].str.lower().isin(PhaseExtractor.IGNORE_EVENTS)]

            # Find the first event related to the team
            start_index = self.__find_next_event(match_df, 2, team_name)
            current_index = start_index

            while start_index < len(match_df):
                event = match_df.iloc[current_index]['type'].lower()
                team =  match_df.iloc[current_index]['team']
                # if the event is in having_ball_events

                if team != team_name or event in PhaseExtractor.LOSING_BALL_EVENTS or event in PhaseExtractor.DELAY_EVENTS:
                    # Extract the series
                    phase_df = match_df.iloc[start_index:current_index]
                    phase_df = phase_df.assign(phase_id=phase_id)
                    phase_id += 1
                    result_df = pd.concat((result_df, phase_df))
                    # clear the current series
                    start_index = self.__find_next_event(match_df, current_index + 1, team_name)
                    current_index = start_index

               
                elif event in PhaseExtractor.HAVING_BALL_EVENTS:
                    # add the event to the current series
                    current_index += 1
                    # if the event is shot extract the series
                    # if event == 'shot':
                    #     result_df = pd.concat((result_df, match_df.iloc[start_index:current_index]))
                    #     phases_ranges.append(current_index - start_index)
                    #     start_index = self.__find_next_event(match_df, current_index, team_name)
                    #     current_index = start_index

                else:                    
                    raise NotImplementedError(f'This type of event ({event}) is not implemented! (Index: {match_df.iloc[current_index]["index"]}, file: {df_dir})')


        # print(result_df)
        result_df=result_df.reset_index(drop=True)
        result_df.to_csv(os.path.join(output_dir, f'{team_name}.csv'), index=False)
    
            
            
            

            
                





