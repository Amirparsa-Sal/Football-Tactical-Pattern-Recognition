from ftpr.visualization import PhaseVisualizer
import pandas as pd
from ftpr.dataloader import load_phases
from ftpr.representation import PlayerDescretizer, MultiParallelDescritizer, EventDescretizer, LocationDescretizer, CMSPADEWriter
from ftpr.miner import find_patterns, rank_patterns, run_miner
from ftpr.preprocessing import PhaseExtractor
from ftpr.clustering import PhaseClustering
import numpy as np
import pickle
import time

def calc_average_length(df):
    s = 0
    pattern_col_index = df.columns.get_loc('pattern')
    for i in range(len(df)):
        pattern = df.iloc[i, pattern_col_index]
        # print(type(pattern))
        s += len(pattern)
    return s / len(df)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

top = 10
# cluster_nums = [10, 100, 400]
cluster_nums = [100]
event_types = {
	'pass': ['corner', 'free kick', 'goal kick', 'interception', 'kick off', 'recovery', 'throw-in']
}
event_des = EventDescretizer('event', events=PhaseExtractor.HAVING_BALL_EVENTS, event_types=event_types)
location_des = LocationDescretizer('location')
multi_des = MultiParallelDescritizer('multi', descretizers=[event_des, location_des])
mapping = multi_des.get_decode_mapping()


if __name__ == '__main__':
    data_multiindex = dict()
    for cluster_num in cluster_nums:
        # data_multiindex[(f'{cluster_num} Clusters', 'CM-SPADE')] = []
        # data_multiindex[(f'{cluster_num} Clusters', 'VMSP')] = []
        data_multiindex[('CM-SPADE', 'Length')] = []
        data_multiindex[('CM-SPADE', 'Count')] = []
        data_multiindex[('VMSP', 'Length')] = []
        data_multiindex[('VMSP', 'Count')] = []
        # data_multiindex[(f'{cluster_num} Clusters', 'FS')] = []


    # team = 'Manchester City'
    # df = pd.read_csv(f'./data/team_phases/{team}.csv')
    # phases = load_phases(df, filter_static_events=True, min_phase_length=3, n_jobs=5)
    # with open(f'./data/players/{team}.pkl', 'rb') as f:
    #     players = pickle.load(f)
    
    # clustering = PhaseClustering(phases)
    clustering_data = load_pickle('./exp/results_kmeans_dtw_z.pkl')
    clustering: PhaseClustering = clustering_data['phase_clusterings'][0]

    for cluster_num in cluster_nums:
        index = clustering_data['n_clusters'].index(cluster_num)
        print(index)
        cls_pred = clustering_data['cls_preds'][index]
        clustering.labels_ = cls_pred
        clustering.n_clusters = cluster_num
        
        # cls_pred, clus = clustering.kmeans_fit(n_clusters=cluster_num, metric='dtw', show_progress=True)
        cluster_scores = clustering.get_cluster_scores(metric='shot')
        best_cluster_indeces = np.argsort(cluster_scores)[::-1]
        
        for ith in range(top):
            phases_in_cluster = clustering.get_cluster_phases(best_cluster_indeces[ith])
            # data_multiindex[(f'{cluster_num} Clusters', 'P')].append(len(phases_in_cluster))
            # data_multiindex[(f'{cluster_num} Clusters', 'S')].append(cluster_scores[best_cluster_indeces[ith]])
            
            writer = CMSPADEWriter()
            multi_des.apply(phases_in_cluster)
            writer.write(multi_des.apply(phases_in_cluster, mode='parallel'), 'output.tmp')
            tic = time.time()
            df = run_miner(algorithm="VMSP", input_filename="output.tmp", output_filename="output.txt", arguments=["20%", "100", "1"])
            # data_multiindex[(f'{cluster_num} Clusters', 'VMSP')].append(calc_average_length(df))
            data_multiindex[('VMSP', 'Length')].append(calc_average_length(df))
            data_multiindex[('VMSP', 'Count')].append(len(df))
            print(time.time() - tic)
            tic = time.time()
            df = run_miner(algorithm="CM-SPADE", input_filename="output.tmp", output_filename="output.txt", arguments=["20%"])
            # data_multiindex[(f'{cluster_num} Clusters', 'CM-SPADE')].append(calc_average_length(df))
            data_multiindex[('CM-SPADE', 'Length')].append(calc_average_length(df))
            data_multiindex[('CM-SPADE', 'Count')].append(len(df))
            print(time.time() - tic)
            # df = run_miner(algorithm="CM-SPADE", input_filename="output.tmp", output_filename="output.txt", arguments=["10%"])
            # data_multiindex[(f'{cluster_num} Clusters', 'FS')].append(len(df))
            print(data_multiindex)

    columns = pd.MultiIndex.from_tuples(data_multiindex.keys(), names=['Num Clusters', 'Alg'])
    df = pd.DataFrame(data_multiindex, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], columns=columns)
    # df.to_csv('sequence_mining.csv')
    print(df)