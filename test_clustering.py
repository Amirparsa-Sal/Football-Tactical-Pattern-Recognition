from ftpr.visualization import PhaseVisualizer
import pandas as pd
from dtaidistance.dtw_ndim import distance_matrix_fast
from sklearn.metrics import silhouette_score
import numpy as np
from ftpr.clustering import PhaseClustering
from ftpr.dataloader import load_phases
from collections import Counter
import pickle
import os
import argparse
from tqdm import tqdm
import random

def monitor_distance(clusters_distances, is_last):
    global inertias, n_clusters
    if is_last:
        cluster_info_dict = dict()
        clusters, distances = zip(*clusters_distances)
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_info_dict:
                cluster_info_dict[cluster_id] = {'sum': 0, 'count': 0}
            cluster_info_dict[cluster_id]['sum'] += distances[i]
            cluster_info_dict[cluster_id]['count'] += 1
        
        total_avg = 0
        for _, value in cluster_info_dict.items():
            value['avg'] = value['sum'] / value['count']
            total_avg += value['avg']
        
        inertias.append(total_avg / n_clusters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str)
    parser.add_argument('--metric', type=str, default='dtw')
    parser.add_argument('--min-length', type=int, default=3)
    parser.add_argument('--filter-static', type=str, default='True')
    parser.add_argument('--normalized', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--step', type=int)
    parser.add_argument('--jobs', type=int, default=5)
    args = parser.parse_args()
    args.normalized = args.normalized == 'True'
    args.filter_static = args.filter_static == 'True'

    available_types = ['kmeans', 'agglomerative']
    if args.type.lower() not in available_types:
        raise ValueError(f'Type argument must be in the following list: {available_types}')
    
    available_metrics = ['dtw', 'euclidean']
    if args.metric.lower() not in available_metrics:
        raise ValueError(f'Metric argument must be in the following list: {available_metrics}')

    if not os.path.exists(args.input):
        raise ValueError('The given input path does not exists!')
    
    # Loading the phases
    df = pd.read_csv(args.input)
    phases = load_phases(df, filter_static_events=args.filter_static, min_phase_length=args.min_length, n_jobs=args.jobs)
    
    np.random.seed(42)
    random.seed(42)
    
    # Perform clustering
    clustering = PhaseClustering(phases, normalize_series=args.normalized)
    distances = distance_matrix_fast(clustering.series)
    
    inertias = []
    silhouettes = []
    clusterings = []

    if args.type.lower() == 'kmeans':
        for n_clusters in tqdm(range(args.start, args.end + 1, args.step)):
            cls_pred, clustering = clustering.kmeans_fit(n_clusters=n_clusters, metric=args.metric, monitor_distances=monitor_distance, show_progress=False)    
            silhouettes.append(silhouette_score(distances, cls_pred))
            clusterings.append(clustering)
            
    else:
        for n_clusters in tqdm(range(args.start, args.end + 1, args.step)):
            cls_pred, clustering = clustering.agglomerative_fit(n_clusters=n_clusters, metric=args.metric)  
            silhouettes.append(silhouette_score(distances, cls_pred))
            clusterings.append(clustering)
    
    
    result_dict = {
        'n_clusters': list(range(args.start, args.end + 1, args.step)),
        'clusterings': clusterings,
        'silhouettes': silhouettes,
        'inertias': inertias
    }

    with open(args.output, 'wb') as f:
        pickle.dump(result_dict, f)
    

