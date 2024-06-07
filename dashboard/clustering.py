from ftpr.visualization import PhaseVisualizer
import pandas as pd
from tslearn.barycenters import dtw_barycenter_averaging
from mplsoccer import Pitch
import numpy as np
from ftpr.clustering import PhaseClustering
from ftpr.dataloader import load_phases
from collections import Counter
import streamlit as st
import threading

num_cols = 3
num_rows = 4

st.set_page_config(layout="wide") 

def hash_clustering(obj: PhaseClustering):
    return hash(obj)

@st.cache_data(max_entries=12, hash_funcs={PhaseClustering: hash_clustering})
def plot_cluster(clustering: PhaseClustering, cluster_index: int):
    series_in_cluster = clustering.get_cluster_series(best_cluster_indeces[cluster_index])
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
    fig.set_facecolor('#22312b')

    for ser in series_in_cluster:
        pitch.scatter(ser[0, 0], ser[0, 1], s=200, ax=ax)
        for i in range(len(ser) - 1):
            pitch.arrows(ser[i, 0], ser[i, 1], ser[i + 1, 0], ser[i + 1, 1],
                            color='#777777', ax=ax, width=1)
    
    return fig, len(series_in_cluster)

def fill_col(col, fig):
    with col:
        st.pyplot(fig)

def on_prev_button_clicked():
    st.session_state['batch'] = st.session_state['batch'] - 1

def on_next_button_clicked():
    st.session_state['batch'] = st.session_state['batch'] + 1

@st.cache_data(max_entries=2)
def perform_clustering(team_name, n_clusters, min_phase_length, metric):
    print(metric)
    df = pd.read_csv(f'../data/team_phases/{team_name}.csv')

    phases = load_phases(df, filter_static_events=True, min_phase_length=min_phase_length, n_jobs=5)

    clustering = PhaseClustering(phases)
    clustering.set_hash(f'{team_name}-Agglo-{min_phase_length}-{metric}')

    cls_pred = clustering.agglomerative_fit(n_clusters=n_clusters, metric=metric, linkage='complete')

    cluster_score = [0 for _ in range(n_clusters)]
    cluster_dist = []
    for cluster_id in range(n_clusters):
        phases_in_cluster = clustering.get_cluster_phases(cluster_id)
        cluster_dist.append(len(phases_in_cluster))
        for phase in phases_in_cluster:
            cluster_score[cluster_id] += Counter(phase['type']).get('Shot', 0)
    cluster_score = np.array(cluster_score)

    best_cluster_indeces = np.argsort(cluster_score)[::-1]

    return clustering, cls_pred, cluster_score, best_cluster_indeces

matches_df_dir = '../data/matches.csv'

# Extract Team names
matches_df = pd.read_csv(matches_df_dir)
matches_df = matches_df.sort_values('match_date')
all_teams = list(matches_df['home_team'].unique())
all_teams.sort()

# Variables
st.sidebar.markdown('## Select a team')
team_name = st.sidebar.selectbox('Choose', [''] + all_teams)

if team_name:
    st.sidebar.markdown('## Num Clusters')
    n_clusters = st.sidebar.slider('Num Clusters', 2, 200, value=100)

    st.sidebar.markdown('## Min Length')
    min_phase_length = st.sidebar.slider('Min Length', 1, 10, value=3)

    st.sidebar.markdown('## Metric')
    metric = st.sidebar.selectbox('Metric', ['dtw', 'euclidean'])

    num_batches = int((n_clusters / num_rows / num_cols) - 0.5) + 1

    if 'batch' in st.session_state:
        batch = st.session_state['batch'] % num_batches
    else:
        batch = 0
        st.session_state['batch'] = 0   

    # Clustering

    clustering, cls_pred, cluster_score, best_cluster_indeces = perform_clustering(team_name, n_clusters, min_phase_length, metric)
    
    batch_progress = st.progress((batch + 1) / num_batches, text = f'Cluster {batch + 1} / {num_batches}')
    loading_progress = st.progress(0, text = f'Loading Batch (0 / {num_cols * num_rows})...')
    cols = st.columns([1] +  (num_cols * [10]) + [1])

    with cols[0]:
        prev_button = st.button('<', use_container_width=True, disabled=False, on_click=on_prev_button_clicked)
        

    with cols[-1]:
        next_button = st.button('>', use_container_width=True, disabled=False, on_click=on_next_button_clicked)

    figs = dict()
    for j in range(num_rows):
        figs[j] = dict()
        for i in range(1, 1 + num_cols):
            with cols[i]:        
                index = batch * num_rows * num_cols + j * num_cols + (i - 1)
                if index < n_clusters:
                    fig, num_phases = plot_cluster(clustering, index)
                    st.markdown(f'Num Phases: {num_phases}, Num Shots: {cluster_score[best_cluster_indeces[index]]}')
                    st.pyplot(fig)
                    # figs[j][i] = {'fig': fig, 'num_phases': num_phases, 'num_shots': cluster_score[best_cluster_indeces[index]]}
                loading_progress.progress((j * num_cols + i) / (num_rows * num_cols))
    
    # for j in range(num_rows):
    #     for i in range(1, 1 + num_cols):
    #         with cols[i]:                
    #             st.markdown(f'Num Phases: {figs[j][i]["num_phases"]}, Num Shots: {figs[j][i]["num_shots"]}')
    #             st.pyplot(figs[j][i]['fig'])
    
    