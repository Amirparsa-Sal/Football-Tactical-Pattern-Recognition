from .entities import Phase
from .preprocessing import normalize, z_normalize
from typing import List
import numpy as np
from dtaidistance import dtw_ndim
from dtaidistance.clustering.kmeans import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

class PhaseClustering:

    SUPPORTED_METRICS = {'dtw', 'softdtw', 'euclidean', 'custom'}
    SUPPORTED_LINKAGES = {'ward', 'complete', 'average', 'single'}

    def __init__(self, phases: List[Phase], remove_duplicates=True, normalize_series='min-max') -> None:
        self.phases = phases
        self.series = self._phase_to_series(phases, remove_duplicates)
        self.normalized = normalize_series
        self.n_clusters = 0
        if normalize_series == 'min-max':
            self.normalized_series = normalize(self.series)
        if normalize_series == 'z':
            self.normalized_series = z_normalize(self.series)   
        else:
            self.normalized = False
        self.done = False
        self.hash = None
    
    def _phase_to_series(self, phases, remove_duplicates):
        series = []
        for phase in phases:
            s = phase.get_location_series(remove_duplicates=remove_duplicates)
            series.append(np.array(s))
        return series
    
    def set_hash(self, hash):
        self.hash = hash

    def __hash__(self) -> int:
        if not self.hash:
            raise RuntimeError('PhaseClustering object is not hashable. Use set_hash method to set a hash.')
        return hash(self.hash)
    
    def _check_metric(self, metric):
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f'The chosen metric ({metric}) is not supported. Try one of these: {self.SUPPORTED_METRICS}')
    
    def _check_linkage(self, linkage):
        if linkage not in self.SUPPORTED_LINKAGES:
            raise ValueError(f'The chosen linkage ({linkage}) is not supported. Try one of these: {self.SUPPORTED_LINKAGES}')
    
    def _check_done(self):
        if not self.done:
            raise RuntimeError('You must perform clustering at least once to use this method!')
    
    def _check_cluster_id(self, cluster_id):
        if cluster_id >= self.n_clusters:
            raise ValueError(f'The chosen cluster does not exist!')
        
    def agglomerative_fit(self, n_clusters, metric, linkage='complete', metric_fn=None):
        # Validate the arguments
        self._check_metric(metric)
        self._check_linkage(linkage)

        if metric == 'Custom' and metric_fn is None:
            raise ValueError('You must provide metric_fn when you chose custom metric!')
    
        clustering = None
        series = self.normalized_series if self.normalized else self.series

        if metric in ['dtw', 'softdtw']:    
            distances = dtw_ndim.distance_matrix_fast(series)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage, compute_full_tree=True)
            self.labels_ = clustering.fit_predict(distances)

        elif metric == 'euclidean':
            series = self._to_time_series_dataset(series)
            series = series.reshape(len(series), -1)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True)
            self.labels_ = clustering.fit_predict(series)

        elif metric == 'custom':
            distances = np.zeros((len(series), len(series)), dtype=np.float64)
            for i in range(len(series) - 1):
                for j in range(i + 1, len(series)):
                    dist = metric_fn(series[i], series[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage, compute_full_tree=True)
            self.labels_ = clustering.fit_predict(distances)
        
        self.done = True
        self.n_clusters = n_clusters
        return self.labels_, clustering
    
    def kmeans_fit(self, n_clusters, metric, monitor_distances=None, **kwargs):
        # Validate the arguments
        self._check_metric(metric)

        clustering = KMeans(n_clusters, **kwargs)
        series = self.normalized_series if self.normalized else self.series

        cls_pred, _ = clustering.fit(series, use_c=True, monitor_distances=monitor_distances)
        
        self.labels_ = [0 for _ in range(len(series))]
        for cluster_id in cls_pred:
            for phase_id in cls_pred[cluster_id]:
                self.labels_[phase_id] = cluster_id

        self.done = True
        self.n_clusters = n_clusters
        return self.labels_, clustering

    def get_cluster_scores(self, metric='shot', avg=False):
        self._check_done()
        if metric not in ['shot', 'xg', 'goal']:
            raise ValueError("Metric value must be in ['shot', 'xg', 'goal'].")
        n_clusters = len(self.labels_)
        cluster_scores = [0 for _ in range(self.n_clusters)]
        cluster_counts = [0 for _ in range(self.n_clusters)]
        for i, phase in enumerate(self.phases):
            cluster_id = self.labels_[i]
            cluster_counts[cluster_id] += 1
            shot_df = phase[phase['type'] == 'Shot']
            if metric == 'shot':
                cluster_scores[cluster_id] += len(shot_df)
            elif metric == 'xg':
                cluster_scores[cluster_id] += shot_df['shot_statsbomb_xg'].sum()
            elif metric == 'goal':
                cluster_scores[cluster_id] += len(shot_df[shot_df['shot_outcome'] == 'Goal'])
        if avg:
            result = [0 for _ in range(n_clusters)]
            for i, count in enumerate(cluster_counts):
                if count != 0:
                    result[i] = cluster_scores[i] / count
            return np.array(result)
        return np.array(cluster_scores)


    def get_cluster_phases(self, cluster_id):
        self._check_done()
        self._check_cluster_id(cluster_id)
        return [phase for i, phase in enumerate(self.phases) if self.labels_[i] == cluster_id]

    def get_cluster_series(self, cluster_id):
        self._check_done()
        self._check_cluster_id(cluster_id)
        return [serie for i, serie in enumerate(self.series) if self.labels_[i] == cluster_id]


    def _to_time_series_dataset(self, dataset):
        if len(dataset) == 0:
            raise ValueError('Dataset must not be empty!')
        
        n_features = len(dataset[0][0])
        max_length = -1
        for series in dataset:
            if len(series) > max_length:
                max_length = len(series)

        result = np.zeros((len(dataset), max_length, n_features))
        npad = [(0, 0)] * np.array(dataset[0]).ndim
        for i, series in enumerate(dataset):
            npad[0] = (0, max_length - len(series))
            padded_series = np.pad(series, pad_width=npad, mode='edge')
            result[i] = padded_series      
        return result