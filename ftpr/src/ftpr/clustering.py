from .entities import Phase
from typing import List
import numpy as np
from dtaidistance import dtw_ndim
from tslearn.utils import to_time_series_dataset
from sklearn.cluster import AgglomerativeClustering

class PhaseClustering:

    SUPPORTED_METRICS = {'dtw', 'softdtw', 'euclidean', 'custom'}
    SUPPORTED_LINKAGES = {'ward', 'complete', 'average', 'single'}

    def __init__(self, phases: List[Phase], location_column='location', remove_duplicates=True) -> None:
        self.phases = phases
        self.series = []
        for phase in phases:
            s = phase.get_location_series(location_columns=[location_column], remove_duplicates=remove_duplicates)
            self.series.append(np.array(s))
        self.done = False
        self.hash = None
    
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
        
    def agglomerative_fit(self, n_clusters, metric, linkage='complete', metric_fn=None):
        # Validate the arguments
        self._check_metric(metric)
        self._check_linkage(linkage)

        if metric == 'Custom' and metric_fn is None:
            raise ValueError('You must provide metric_fn when you chose custom metric!')
    
        clustering = None

        if metric == 'dtw':
            distances = dtw_ndim.distance_matrix_fast(self.series)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage, compute_full_tree=True)
            clustering.fit_predict(distances)

        elif metric == 'euclidean':
            series = self._to_time_series_dataset(self.series)
            series = series.reshape(len(series), -1)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage, compute_full_tree=True)
            clustering.fit_predict(series)

        elif metric == 'custom':
            distances = np.zeros((len(self.series), len(self.series)), dtype=np.float64)
            for i in range(len(self.series) - 1):
                for j in range(i + 1, len(self.series)):
                    dist = metric_fn(self.series[i], self.series[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage, compute_full_tree=True)
            clustering.fit_predict(distances)
        
        self.labels_ = clustering.labels_
        self.done = True
        return self.labels_
    
    def get_cluster_phases(self, cluster_id):
        self._check_done()
        return [phase for i, phase in enumerate(self.phases) if self.labels_[i] == cluster_id]

    def get_cluster_series(self, cluster_id):
        self._check_done()
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