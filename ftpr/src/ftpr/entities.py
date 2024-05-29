import numpy as np
import pandas as pd

class Phase:

    def __init__(self, df: pd.DataFrame, id_column='phase_id') -> None:
        self.id = df[id_column]
        self.id_column = id_column
        self._df = df
        self.iloc = df.iloc
        self.__drop_nan()

    @property
    def df(self):
        return self._df
    
    def __drop_nan(self):
        if self.is_splited():
            self._df = self._df[self._df['location_x'].notna() & self._df['location_y'].notna()]
        else:
            self._df = self._df[self._df['location'].notna()]

    @df.setter
    def df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('You must pass a pandas dataframe object!')
        self._df = df
        
    def split_locations(self, location_columns=None):

        '''A function to split x and y axis of location columns (location, pass_end_location, carry_end_location)'''
        if not location_columns:
            location_columns = ['location']
        new_df = self._df.copy()
        # loop over specified columns
        for column in location_columns:
            # Add two new columns for x and y axis
            index = new_df.columns.get_loc(column)
            new_df.insert(index + 1, f'{column}_x', 0)
            new_df.insert(index + 2, f'{column}_y', 0)
            # set dtype of columns to np.float64
            new_df = new_df.astype({f'{column}_x': np.float64, f'{column}_y': np.float64})
            # loop over the entire df
            for i in range(len(self._df)):
                # set the value of x and y columns
                if not pd.isna(new_df.iloc[i][column]):
                    x, y = new_df.iloc[i][column][1:-1].replace(' ', '').split(',')[:2]
                    new_df.iat[i, index + 1] = float(x)
                    new_df.iat[i, index + 2] = float(y)
                else:
                    new_df.iat[i, index + 1] = None
                    new_df.iat[i, index + 2] = None
            # drop the specified column
            new_df.drop(column, axis = 1, inplace=True)
        return Phase(new_df, self.id_column)
                
    def filter_static_events(self):
        '''A function to remove static events (the events starting and ending at the same location)'''
        if len(self._df) == 0:
            return self
        new_df = self._df.copy()
        new_df.insert(len(self._df.columns), 'keep', True)
        location = None
        # loop over the entire df except the last event
        for i in range(len(self._df) - 1):
            row = self._df.iloc[i]
            new_location = self.get_location(i)
            # if the event is ball receipt
            if row['type'] == 'Ball Receipt*':
                # drop if the ending location is as same as the starting location or the location is as same as the next event's location
                if location == new_location or (i < len(self._df) - 1 and self.get_location(i + 1) == new_location):
                    new_df.iat[i, len(self._df.columns)] = False
            # drop if the event is carry and the ending location is as same as the starting location
            elif row['type'] == 'Carry' and new_location == self.get_location(i, 'Carry'):
                new_df.iat[i, len(self._df.columns)] = False
        # if the last event is ball receipt and its location is different from the last end location: remove it
        if len(self._df) == 1:
            new_df.iat[0, len(self._df.columns)] = True
        else:
            last_event = self._df.iloc[-2]['type']
            if self._df.iloc[-1]['type'] == 'Ball Receipt*' and self.get_location(-2, last_event) != self.get_location(-1):
                new_df.iat[-1, len(self._df.columns)] = False
            else:
                new_df.iat[-1, len(self._df.columns)] = True
        return Phase(self._df[new_df['keep']], self.id_column)

    def __remove_duplicate_locations(self, arr):
        result = [arr[0].tolist()]
        for i in range(1, len(arr)):
            if arr[i].tolist() != result[-1]:
                result.append(arr[i].tolist())
        return result
            
    def get_location_series(self, location_columns, remove_duplicates=False):
        result = np.zeros((len(self._df), len(location_columns) * 2))
        # if the location is splitted convert the columns to numpy array and concatenate them
        if self.is_splited():
            for i, col in enumerate(location_columns):
                result[:, i * 2] = np.array(self._df[f'{col}_x'])
                result[:, i * 2 + 1] = np.array(self._df[f'{col}_y'])
        else:
            # else if the location is not splitted iterate over the column and create the final array
            for i in range(len(self._df)):
                for j, col in enumerate(location_columns):
                    x, y = self._df.iloc[i][col][1:-1].replace(' ', '').split(',')
                    result[i, j * 2] = x
                    result[i, j * 2 + 1] = y
        return self.__remove_duplicate_locations(result) if remove_duplicates else result

    def get_summary(self):
        columns = ['location', 'pass_end_location', 'carry_end_location']
        if 'Shot' in self._df['type']:
            columns.append('shot_end_location')
        new_columns = []
        if self.is_splited():
            for col in columns:
                new_columns.append(f'{col}_x')
                new_columns.append(f'{col}_y')
        else:
            new_columns = columns    
        new_columns.extend(['type', 'timestamp'])
        return self._df[new_columns]
    
    def is_splited(self):
        return 'location_x' in self._df.columns

    def get_location(self, i, event=None):
        i = i % len(self._df)
        col = f'{event.lower()}_end_location' if event else 'location'
        if self.is_splited():
            return self._df.iloc[i][f'{col}_x'], self._df.iloc[i][f'{col}_y']
        return self._df.iloc[i][col][1:-1].replace(' ', '').split(',')
    
    def __getitem__(self, key):
        return self._df[key]
    
    def __len__(self):
        return len(self._df)