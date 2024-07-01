from spmf import Spmf
from typing import List
import pandas as pd

def run_miner(algorithm: str, input_filename: str, output_filename: str, arguments: List[str], pickle=False, csv_filename=None, ascending=False):
    spmf = Spmf(algorithm_name=algorithm, input_filename=input_filename, 
                output_filename=output_filename, arguments=arguments)
    spmf.run()
    df = spmf.to_pandas_dataframe(pickle=pickle)
    df = df.sort_values(by='sup', ascending=ascending)
    if csv_filename:
        df.to_csv(csv_filename)
    return df

def find_patterns(df, query: str, pattern_col='pattern'):
    indeces = []
    pattern_col_index = df.columns.get_loc(pattern_col)
    for i in range(len(df)):
        sequence = df.iloc[i, pattern_col_index]
        for item in sequence:
            splited = item.split(' ')
            if query in splited:
                indeces.append(i)
    return df.iloc[indeces]

def rank_patterns(df, scores, mapping, min_length=1, ascending=False, pattern_col='pattern', sup_col='sup'):
    pattern_col_index = df.columns.get_loc(pattern_col)
    sup_col_index = df.columns.get_loc(sup_col)
    result = {"pattern": [], "translate": [], "score": []}
    for i in range(len(df)):
        pattern = df.iloc[i, pattern_col_index]
        translate = []
        score = 0
        count = 0
        for itemset in pattern:
            splited = itemset.split(' ')
            translate.append(tuple(mapping[int(index)] for index in splited))
            for index in splited:
                if int(index) in scores:
                    score += scores[int(index)]
                count += 1
        score = 0 if count < min_length else score / count * df.iloc[i, sup_col_index]
        # score = count * df.iloc[i, -1]
        result['pattern'].append(pattern)
        result['translate'].append(translate)
        result['score'].append(score)
        
    result = pd.DataFrame(result)
    result = result.sort_values(by='score', ascending=ascending)
    return result