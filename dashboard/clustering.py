import pandas as pd
from mplsoccer import Pitch
import numpy as np
from ftpr.clustering import PhaseClustering
from ftpr.dataloader import load_phases
from collections import Counter
import streamlit as st
from ftpr.visualization import PhaseVisualizer

num_cols = 3
num_rows = 4
bins = (120 // 4, 80 // 4)

events_config = {
    'Pass': {
        'color': '#00ffff'
    },
    'Carry': {
        'color': '#00aa00',
        'dashed': True
    },
    'Shot': {
        'color': '#aa0000'
    } 
}

location_columns = ['location', 'pass_end_location', 'carry_end_location', 'shot_end_location']

st.set_page_config(layout="wide") 

def hash_clustering(obj: PhaseClustering):
    return hash(obj)

def on_cluster_button_click(index):
    st.session_state['inspect_mode'] = True
    st.session_state['cluster_index'] = index
    st.session_state['phase_index'] = 0

@st.cache_data(max_entries=12, hash_funcs={PhaseClustering: hash_clustering})
def plot_cluster(clustering: PhaseClustering, cluster_index: int, visualization: str):
    series_in_cluster = clustering.get_cluster_series(best_cluster_indeces[cluster_index])
    if visualization == 'Arrows':
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        fig.set_facecolor('#22312b')

        for ser in series_in_cluster:
            pitch.scatter(ser[0, 0], ser[0, 1], s=200, ax=ax)
            for i in range(len(ser) - 1):
                pitch.arrows(ser[i, 0], ser[i, 1], ser[i + 1, 0], ser[i + 1, 1],
                                color='#777777', ax=ax, width=1)
    else:
        pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
              pitch_color='#22312b', line_color='#efefef', linewidth=1)
        pv = PhaseVisualizer(figsize=(16, 11))
        fig, ax = pv.plot_heatmap(pitch, series_in_cluster, bins=bins)

    return fig, len(series_in_cluster)

def on_prev_button_clicked():
    if st.session_state['inspect_mode']:
        st.session_state['phase_index'] = st.session_state['phase_index'] - 1
    else:
        st.session_state['batch'] = st.session_state['batch'] - 1

def on_next_button_clicked():
    if st.session_state['inspect_mode']:
        st.session_state['phase_index'] = st.session_state['phase_index'] + 1
    else:
        st.session_state['batch'] = st.session_state['batch'] + 1

def on_back_to_menu_button_clicked():
    st.session_state['inspect_mode'] = False

@st.cache_data(max_entries=1)
def load_csv(dir):
    return pd.read_csv(dir)

@st.cache_data(max_entries=1)
def load_team_phases(team, _df, filter_static_events=True, min_phase_length=3, n_jobs=5):
    return load_phases(_df, filter_static_events=filter_static_events, min_phase_length=min_phase_length, n_jobs=n_jobs)

@st.cache_data(max_entries=2)
def perform_clustering(type, team_name, n_clusters, min_phase_length, metric, normalized):
    df = load_csv(f'../data/team_phases/{team_name}.csv')

    phases = load_team_phases(team_name, df, filter_static_events=True, min_phase_length=min_phase_length, n_jobs=5)

    clustering = PhaseClustering(phases, normalize_series=normalized)
    st.session_state['clustering'] = clustering
    clustering.set_hash(f'{team_name}-{type}-{n_clusters}-{min_phase_length}-{metric}-{normalized}')

    cls_pred = None
    if type == 'Agglomerative':
        cls_pred = clustering.agglomerative_fit(n_clusters, metric, linkage='complete')
    else:
        cls_pred = clustering.kmeans_fit(n_clusters, metric, show_progress=False)

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

if 'inspect_mode' not in st.session_state:
    st.session_state['inspect_mode'] = False

if team_name:

    if not st.session_state['inspect_mode']:
        types = ['Agglomerative', 'K-means']
        st.sidebar.markdown('## Clustering Type')
        type = st.sidebar.selectbox('Clustering Type', types)

        st.sidebar.markdown('## Num Clusters')
        n_clusters = st.sidebar.slider('Num Clusters', 2, 500, value=100)

        st.sidebar.markdown('## Min Length')
        min_phase_length = st.sidebar.slider('Min Length', 1, 10, value=3)

        st.sidebar.markdown('## Metric')
        metrics = ['dtw', 'softdtw', 'euclidean']
        metric = st.sidebar.selectbox('Metric', metrics)

        st.sidebar.markdown('## Normalized')
        normalized_list = ['min-max', 'z', 'No']
        normalized = st.sidebar.selectbox('Normalized', normalized_list)


        num_batches = int((n_clusters / num_rows / num_cols) - 0.5) + 1

        if 'batch' in st.session_state:
            batch = st.session_state['batch'] % num_batches
        else:
            batch = 0
            st.session_state['batch'] = 0   

        # Clustering

        clustering, cls_pred, cluster_score, best_cluster_indeces = perform_clustering(type, team_name, n_clusters, min_phase_length, metric, normalized)
        st.session_state['best_cluster_indeces'] = best_cluster_indeces

        batch_progress = st.progress((batch + 1) / num_batches, text = f'Cluster {batch + 1} / {num_batches}')
        loading_progress = st.progress(0, text = f'Loading Batch (0 / {num_cols * num_rows})...')
        cols = st.columns([1] +  (num_cols * [10]) + [1])
        
        st.sidebar.markdown('## Visualization')
        visualization_list = ['Arrows', 'Heatmap']
        visualization = st.sidebar.selectbox('Visualization', visualization_list)
        
        with cols[0]:
            prev_button = st.button('<', use_container_width=True, disabled=False, on_click=on_prev_button_clicked)        

        with cols[-1]:
            next_button = st.button('>', use_container_width=True, disabled=False, on_click=on_next_button_clicked)

        figs = dict()
        for j in range(num_rows):
            for i in range(1, 1 + num_cols):
                with cols[i]:        
                    index = batch * num_rows * num_cols + j * num_cols + (i - 1)
                    if index < n_clusters:
                        fig, num_phases = plot_cluster(clustering, index, visualization=visualization)
                        st.button(f'Num Phases: {num_phases}, Num Shots: {cluster_score[best_cluster_indeces[index]]}',
                                key=f'button_{index}', on_click=on_cluster_button_click, args=(index,))
                        st.pyplot(fig)
                    loading_progress.progress((j * num_cols + i) / (num_rows * num_cols))
        loading_progress.empty()

    else:
        st.sidebar.empty()
        cluster_index = st.session_state['cluster_index']
        best_cluster_indeces = st.session_state['best_cluster_indeces']
        clustering: PhaseClustering = st.session_state['clustering']
        phases_in_cluster = clustering.get_cluster_phases(best_cluster_indeces[cluster_index])
        
        phase_index = st.session_state['phase_index'] % len(phases_in_cluster)
        phase = phases_in_cluster[phase_index]
        phase = phase.split_locations(location_columns)

        st.button('Back to Main Page', on_click=on_back_to_menu_button_clicked)
        first_row = phase.iloc[0]
        st.markdown(f'Phase ID: {first_row["phase_id"]}')
        st.markdown(f'Phase UUID: {first_row["id"]}')
        st.markdown(f'Match ID: {first_row["match_id"]}')
        
        col1, col2, col3 = st.columns([1, 10, 1])

        series_in_cluster = clustering.get_cluster_series(best_cluster_indeces[cluster_index])

        with col1:
            prev_button = st.button('<', use_container_width=True, disabled=False, on_click=on_prev_button_clicked)
        
        with col3:
            next_button = st.button('>', use_container_width=True, disabled=False, on_click=on_next_button_clicked)

        with col2:
            visualizer = PhaseVisualizer(figsize=(16, 11))
            fig, ax = visualizer.plot_phase(phase, events_config)
            st.pyplot(fig)
            st.dataframe(phase.get_summary())
            st.write(series_in_cluster[phase_index])