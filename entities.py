import numpy as np

class Phase:

    def __init__(self, df_slice, id_column='phase_id') -> None:
        self.id = df_slice[id_column]
        self.id_column = id_column
        self.df_slice = df_slice

    def split_locations(self, location_columns=None):

        '''A function to split x and y axis of location columns (location, pass_end_location, carry_end_location)'''
        if not location_columns:
            location_columns = ['location']
        new_df = self.df_slice.copy()
        # loop over specified columns
        for column in location_columns:
            # Add two new columns for x and y axis
            index = new_df.columns.get_loc(column)
            new_df.insert(index + 1, f'{column}_x', 0)
            new_df.insert(index + 2, f'{column}_y', 0)
            # set dtype of columns to np.float64
            new_df = new_df.astype({f'{column}_x': np.float64, f'{column}_y': np.float64})
            # loop over the entire df
            for i in range(len(self.df_slice)):
                # set the value of x and y columns
                if not pd.isna(new_df.iloc[i][column]):
                    x, y = new_df.iloc[i][column][1:-1].replace(' ', '').split(',')
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
        new_df = self.df_slice.copy()
        new_df.insert(len(self.df_slice.columns), 'keep', True)
        location = None
        # loop over the entire df
        for i in range(len(self.df_slice)):
            row = self.df_slice.iloc[i]
            new_location = row['location']
            # if the event is ball receipt
            if row['type'] == 'Ball Receipt*':
                # drop if the ending location is as same as the starting location or the location is as same as the next event's location
                if location == new_location or (i < len(self.df_slice) - 1 and self.df_slice.iloc[i + 1]['location'] == new_location):
                    new_df.iat[i, len(self.df_slice.columns)] = False
            # drop if the event is carry and the ending location is as same as the starting location
            elif row['type'] == 'Carry' and new_location == row['carry_end_location']:
                new_df.iat[i, len(self.df_slice.columns)] = False
        return Phase(self.df_slice[new_df['keep']], self.id_column)

    def __remove_duplicate_locations(self, arr):
        result = [arr[0].tolist()]
        for i in range(1, len(arr)):
            if arr[i].tolist() != result[-1]:
                result.append(arr[i].tolist())
        return result
            
    def get_location_series(self, location_columns, remove_duplicates=False, splitted_locations=False):
        result = np.zeros((len(self.df_slice), len(location_columns) * 2))
        # if the location is splitted convert the columns to numpy array and concatenate them
        if splitted_locations:
            for i, col in enumerate(location_columns):
                result[:, i * 2] = np.array(self.df_slice[f'{col}_x'])
                result[:, i * 2 + 1] = np.array(self.df_slice[f'{col}_y'])
        else:
            # else if the location is not splitted iterate over the column and create the final array
            for i in range(len(self.df_slice)):
                for j, col in enumerate(location_columns):
                    x, y = self.df_slice.iloc[i][col][1:-1].replace(' ', '').split(',')
                    result[i, j * 2] = x
                    result[i, j * 2 + 1] = y
        return self.__remove_duplicate_locations(result) if remove_duplicates else result

    def get_summary(self, splitted_location=False):
        columns = ['location', 'pass_end_location', 'carry_end_location']
        new_columns = []
        if splitted_location:
            for col in columns:
                new_columns.append(f'{col}_x')
                new_columns.append(f'{col}_y')
        else:
            new_columns = columns    
        new_columns.extend(['type', 'timestamp'])
        return self.df_slice[new_columns]