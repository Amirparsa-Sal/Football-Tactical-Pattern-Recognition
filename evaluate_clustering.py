import argparse
import os
import pickle
from dtaidistance.dtw_ndim import distance_matrix_fast
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
from ftpr.clustering import PhaseClustering
from mplsoccer import Pitch

def inertia(distance_matrix, cluster_indices):
    """
    Compute the fitness of a clustering solution.

    Args:
    distance_matrix (numpy.ndarray): A 2D array containing pairwise distances between time series.
    cluster_indices (list): A list containing the cluster index for each time series.
    k (int): The number of clusters.

    Returns:
    float: The fitness value of the clustering solution.
    """
    k = len(cluster_indices)
    clusters = {i: [] for i in range(k)}
    num_series = len(cluster_indices)

    # Group the indices of the time series by their cluster assignments
    for idx, cluster_id in enumerate(cluster_indices):
        clusters[cluster_id].append(idx)

    intra_cluster_distances = []
    inter_cluster_distances = []

    # Calculate intra-cluster distances
    for cluster_id, indices in clusters.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                intra_cluster_distances.append(distance_matrix[indices[i], indices[j]])

    # Calculate inter-cluster distances
    for i in range(k):
        for j in range(i + 1, k):
            if not clusters[i] or not clusters[j]:
                continue
            for idx_i in clusters[i]:
                for idx_j in clusters[j]:
                    inter_cluster_distances.append(distance_matrix[idx_i, idx_j])

    intra_cluster_mean = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
    inter_cluster_mean = np.mean(inter_cluster_distances) if inter_cluster_distances else np.inf

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    fitness = intra_cluster_mean / (inter_cluster_mean + epsilon)

    return fitness

def create_and_save_plot(xs, ys, path, labels=None, title=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for i, x in enumerate(xs):
        if labels:
            plt.legend(loc='upper left')
            ax.plot(x, ys[i], label=labels[i])
        else:
            ax.plot(x, ys[i])
    if title:
        ax.set_title(title)
    fig.savefig(path)
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

    all_stats = {
        "silhouettes_avg": dict(),
        f'top{args.top}-silhouettes': dict(),
        f'silhouettes_in_top{args.top}_clusters': dict(),
        "inertias": dict(),
        "empty_clusters": dict(),
        "average_num_data_in_clusters": dict(),
        "std_num_data_in_clusters": dict()
    }
    
    for name in file_names:
        print(f'{name}:')
        exp_name = name.split('.')[0]
        exp_out_dir = os.path.join(args.output_dir, exp_name)
        
        if not os.path.exists(exp_out_dir):
            os.mkdir(exp_out_dir)
        
        if not os.path.exists(os.path.join(exp_out_dir, 'clusters_arrows')):
            os.mkdir(os.path.join(exp_out_dir, 'clusters_arrows'))
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
        distances = distance_matrix_fast(series)
        
        # Average Silhouette Diagram with respect to n_clusters
        all_stats['silhouettes_avg'][name] = []
        all_stats[f'top{args.top}-silhouettes'][name] = []
        all_stats[f'silhouettes_in_top{args.top}_clusters'][name] = []
        all_stats['inertias'][name] = []
        all_stats['empty_clusters'][name] = []
        all_stats['average_num_data_in_clusters'][name] = []
        all_stats['std_num_data_in_clusters'][name] = []
        print('Plotting Diagrams...')
        for i, clustering in enumerate(clustering_data['clusterings']):
            cls_pred = clustering_data['cls_preds'][i]
            phase_clustering.labels_ = cls_pred
            phase_clustering.n_clusters = clustering_data['n_clusters'][i]
            # average data in clusters
            data_in_clusters = [0 for _ in range(clustering_data['n_clusters'][i])]
            for c in cls_pred:
                data_in_clusters[c] += 1
            all_stats['average_num_data_in_clusters'][name].append(np.mean(data_in_clusters))
            all_stats['std_num_data_in_clusters'][name].append(np.std(data_in_clusters))
            # empty clusters
            n_empty = 0
            for j in range(clustering_data['n_clusters'][i]):
                if j not in cls_pred:
                    n_empty += 1
            all_stats['empty_clusters'][name].append(n_empty)
            #
            # inertia
            all_stats['inertias'][name].append(inertia(distances, cls_pred))
            # silhouette
            scores = silhouette_samples(distances, cls_pred, metric='precomputed')
            all_stats['silhouettes_avg'][name].append(np.average(scores))
            # top-k silhoeuutes and silhouette in top-k
            n_clusters = min(args.top, clustering_data['n_clusters'][i])
            best_clusters = phase_clustering.get_cluster_scores()
            best_clusters_indeces = np.argsort(best_clusters)[::-1][:n_clusters]
            silhouette_info = get_silhouette_info(scores, cls_pred)
            # top-k silhoeuutes
            s_avg = 0
            count_avg = 0
            for info in silhouette_info[:n_clusters]:
                count_avg += len(info[1])
                s_avg += info[2] * len(info[1])
            all_stats[f'top{args.top}-silhouettes'][name].append(s_avg / count_avg)
            # silhouette in top-k
            s_topk = 0
            count_topk = 0
            for info in silhouette_info:
                if info[0] in best_clusters_indeces:
                    count_topk += len(info[1])
                    s_topk += info[2] * len(info[1])
            all_stats[f'silhouettes_in_top{args.top}_clusters'][name].append(s_topk / count_topk)
            
            # Subplots for top-k clusters (for each n_clusters)
            pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
            fig, axs = pitch.grid(nrows=4, ncols=3, figheight=30,
                      endnote_height=0.03, endnote_space=0,
                      axis=False,
                      title_height=0.08, grid_height=0.84)
            fig.set_facecolor('#22312b')
            for idx, ax in enumerate(axs['pitch'].flat):
                if idx < clustering_data['n_clusters'][i]:
                    phase_clustering.labels_ = cls_pred
                    phase_clustering.n_clusters = clustering_data['n_clusters'][i]
                    series_in_cluster = phase_clustering.get_cluster_series(best_clusters_indeces[idx])
                    for ser in series_in_cluster:
                        pitch.scatter(ser[0, 0], ser[0, 1], s=200, ax=ax)
                        for z in range(len(ser) - 1):
                            pitch.arrows(ser[z, 0], ser[z, 1], ser[z + 1, 0], ser[z + 1, 1],
                                            color='#777777', ax=ax, width=1)
                    ax.set_title(f'#Phases: {len(series_in_cluster)} #Score: {best_clusters[best_clusters_indeces[idx]]}', fontsize=20, color='white')
            fig.savefig(os.path.join(exp_out_dir, 'clusters_arrows', f"{clustering_data['n_clusters'][i]}.png"))
            
        xs = clustering_data['n_clusters']
        
        for key in all_stats:
            create_and_save_plot([xs], [all_stats[key][name]], os.path.join(exp_out_dir, f'{key}.png'), title=key)
        
    # Create comparison plots
    xs = clustering_data['n_clusters']
    xss = [xs for _ in range(len(file_names))]
    
    for key in all_stats:
        ys = []
        labels = []
        for name, stat in all_stats[key].items():
            labels.append(name)
            ys.append(stat)
            
        create_and_save_plot(xss, ys, os.path.join(args.output_dir, f'{key}.png'), labels=labels, title=key)
    