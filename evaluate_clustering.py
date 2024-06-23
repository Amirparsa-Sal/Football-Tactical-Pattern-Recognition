import argparse
import os
import pickle
from dtaidistance.dtw_ndim import distance_matrix_fast
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
from ftpr.clustering import PhaseClustering

def create_and_save_plot(x, y, path, title=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y)
    fig.savefig(path)
    if title:
        ax.set_title(title)
    plt.close(fig)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def clustering_dict_to_list(cls_pred):
	total_num = 0
	for _, value in cls_pred.items():
		total_num += len(value)
	result = [0] * total_num
	for key, value in cls_pred.items():
		for id in value:
			result[id] = key
	return result

def get_silhouette_info(shilhouette_scores, cls_pred):
	cls_pred_dict = None
	if not isinstance(cls_pred, dict):
		cls_pred_dict = dict()
		for i, cls in enumerate(cls_pred):
			cls_pred_dict[cls] = cls_pred_dict.get(cls, []) + [i]
	else:
		cls_pred_dict = cls_pred

	silhouette_info = []
	for cls, indeces in cls_pred_dict.items():
		cls_scores = []
		for index in indeces:
			cls_scores.append(shilhouette_scores[index])
		cls_scores.sort(reverse=True)
		silhouette_info.append((cls, cls_scores, np.mean(cls_scores)))
	
	silhouette_info = sorted(silhouette_info, key=lambda x: x[2], reverse=True)
	return silhouette_info
	

if __name__ == '__main__':
    # Defining parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--top', type=int)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    # check if folder exists
    if not os.path.exists(args.input_dir):
        raise ValueError(f'The input folder {args.input_dir} does not exist!')
    
    # get all pickle files in the folder
    file_names = [f for f in os.listdir(args.input_dir) if f.endswith('.pkl')]
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for name in file_names:
        exp_name = name.split('.')[0]
        exp_out_dir = os.path.join(args.output_dir, exp_name)
        if not os.path.exists(exp_out_dir):
            os.mkdir(exp_out_dir)
        # Read pickle file and series
        clustering_data = load_pickle(os.path.join(args.input_dir, name))

        if 'phase_clusterings' not in clustering_data or len(clustering_data['phase_clusterings']) == 0:
            raise ValueError('Clustering data does not contain any phase_clusterings object!')
        
        if 'clusterings' not in clustering_data or \
            'n_clusters' not in clustering_data or \
            'cls_preds' not in clustering_data or \
            len(clustering_data['clusterings']) != len(clustering_data['n_clusters']):
            raise ValueError('Clustering data format is curropted!')
        
        phase_clustering: PhaseClustering = clustering_data['phase_clusterings'][0]
        series = phase_clustering.series
        
        # Average Silhouette Diagram with respect to n_clusters
        silhouettes_avg = []
        silhouettes_top = []
        silhouettes_avg_topk_clusters = []
        distance = distance_matrix_fast(series)
        for i, clustering in enumerate(clustering_data['clusterings']):
            cls_pred = clustering_data['cls_preds'][i]
            scores = silhouette_samples(distance, cls_pred, metric='precomputed')
            silhouettes_avg.append(np.average(scores))
            # top-k silhoeuutes and silhouette in top-k
            n_clusters = min(args.top, clustering_data['n_clusters'][i])
            phase_clustering.labels_ = cls_pred
            best_clusters = phase_clustering.get_cluster_scores()[:n_clusters]
            best_clusters_indeces = np.argsort(best_clusters)[::-1]
            silhouette_info = get_silhouette_info(scores, cls_pred)
            # top-k silhoeuutes
            s_avg = 0
            count_avg = 0
            for info in silhouette_info[:n_clusters]:
                count_avg += len(info[1])
                s_avg += info[2] * len(info[1])
            silhouettes_top.append(s_avg / count_avg)
            # silhouette in top-k
            s_topk = 0
            count_topk = 0
            for info in silhouette_info:
                if info[0] in best_clusters_indeces:
                    count_topk += len(info[1])
                    s_topk += info[2] * len(info[1])
            silhouettes_avg_topk_clusters.append(s_topk / count_topk)

        xs = clustering_data['n_clusters']
        create_and_save_plot(xs, silhouettes_avg, os.path.join(exp_out_dir, 'silhouette_avg.png'), title='Average Silhouette')
        
        # Average top-k Silhouette Diagram
        create_and_save_plot(xs, silhouettes_top, os.path.join(exp_out_dir, 'topk-silhoeutte.png'), title=f'Average Top-{args.top} Silhouette')

        # Average Silhouette Diagram of top-k clusters
        create_and_save_plot(xs, silhouettes_avg_topk_clusters, os.path.join(exp_out_dir, 'silhouette_topk.png'), title=f'Average Silhouette in Top-{args.top}')

        # Average Inertia Diagram with respect to n_clusters
        if 'kmeans' in name.lower():
            create_and_save_plot(xs, clustering_data['inertias'], os.path.join(exp_out_dir, 'inertia.png'), title='Inertia')

        # TODO: Subplots for top-k clusters (for each n_clusters)

        # TODO: Rank-index Matrix (Later)