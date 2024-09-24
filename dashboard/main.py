import pandas as pd
from mplsoccer import Pitch
import numpy as np
from ftpr.clustering import PhaseClustering
from ftpr.dataloader import load_phases
import streamlit as st
from ftpr.visualization import PhaseVisualizer
from ftpr.preprocessing import PhaseExtractor
from ftpr.representation import EventDescretizer, LocationDescretizer, MultiSequentialDescritizer, MultiParallelDescritizer, CMSPADEWriter, PlayerDescretizer
from ftpr.miner import rank_patterns, run_miner
import os
import pickle 
import multiprocessing
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=4)
parser.add_argument('--cols', type=int, default=3)
parser.add_argument('--players-path', type=str, default=os.path.join('..', 'data', 'players.pkl'))
parser.add_argument('--matches-path', type=str, default=os.path.join('..', 'data', 'matches.csv'))
args = parser.parse_args()

num_cols = args.cols
num_rows = args.rows

players_data_path = args.players_path
matches_df_path = args.matches_path

location_columns = ['location', 'pass_end_location', 'carry_end_location', 'shot_end_location']


bins = (120 // 4, 80 // 4)

event_types = {
	'pass': ['corner', 'free kick', 'goal kick', 'interception', 'kick off', 'recovery', 'throw-in']
}

event_desc = EventDescretizer('event', events=PhaseExtractor.HAVING_BALL_EVENTS, event_types=event_types)
location_desc = LocationDescretizer('loc')

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

st.set_page_config(layout="wide") 

def on_ranking_sliders_change():
    st.session_state['no_mining'] = True
    
def on_pattern_mining_params_change():
    if 'miner_process' in st.session_state   and st.session_state['miner_process']:
        st.session_state['miner_process'].terminate()
        print('terminated')
        st.session_state['miner_process'].join()

def run_miner_process(algorithm, input, output, args, queue):
    try:
        output = run_miner(algorithm, input, output, args)
        queue.put(output)
    except subprocess.CalledProcessError as e:
        queue.put(f"Subprocess failed with error: {e.output.decode()}")
    
@st.cache_data(max_entries=3)
def load_players(team_name):
    with open(os.path.join(players_data_path, f'{team_name}.pkl'), 'rb') as f:
        players = pickle.load(f)
    return list(players[team_name])

def create_select_box(name, options, default_value=None, session_key=None, container=st.sidebar):
    if session_key in st.session_state:
        index = options.index(st.session_state[session_key])
    else:
        index = options.index(default_value) if default_value else 0
    return container.selectbox(name, options, index=index, key=session_key)

def create_slider(name, min_range, max_range, default_value, session_key=None, container=st.sidebar, onchange=None):
    if session_key in st.session_state:
        value = st.session_state[session_key]
    else:
        value = default_value
    if onchange:
        return container.slider(name, min_range, max_range, value=value, key=session_key, on_change=onchange)    
    return container.slider(name, min_range, max_range, value=value, key=session_key)

def hash_clustering(obj: PhaseClustering):
    return hash(obj)

def save_config():
    st.session_state['n_clusters'] = n_clusters
    st.session_state['min_phase_length'] = min_phase_length
    st.session_state['normalized'] = normalized
    st.session_state['type'] = type
    st.session_state['metric'] = metric
    st.session_state['ranking_metric'] = ranking_metric
    st.session_state['team_name'] = team_name
    st.session_state['visualization'] = visualization
    
def on_cluster_button_click(index):
    st.session_state['mode'] = 'inspect'
    st.session_state['cluster_index'] = index
    st.session_state['phase_index'] = 0
    st.session_state['best_cluster_indeces'] = best_cluster_indeces
    save_config()
    
def on_patterns_button_click(index):
    st.session_state['mode'] = 'patterns'
    st.session_state['cluster_index'] = index
    st.session_state['best_cluster_indeces'] = best_cluster_indeces
    save_config()
    
@st.cache_data(max_entries=12, hash_funcs={PhaseClustering: hash_clustering})
def plot_cluster(clustering: PhaseClustering, cluster_index: int, _best_cluster_indeces, visualization: str, ranking_metric: str, avg):
    series_in_cluster = clustering.get_cluster_series(_best_cluster_indeces[cluster_index])
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
    if st.session_state['mode'] == 'inspect':
        st.session_state['phase_index'] = st.session_state['phase_index'] - 1
    elif st.session_state['mode'] == 'main':
        st.session_state['batch'] = st.session_state['batch'] - 1

def on_next_button_clicked():
    if st.session_state['mode'] == 'inspect':
        st.session_state['phase_index'] = st.session_state['phase_index'] + 1
    elif st.session_state['mode'] == 'main':
        st.session_state['batch'] = st.session_state['batch'] + 1

def on_back_to_menu_button_clicked():
    st.session_state['mode'] = 'main'

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
    st.session_state['clustering'] = clustering
    
    cls_pred = None
    if type == 'Agglomerative':
        cls_pred = clustering.agglomerative_fit(n_clusters, metric, linkage='complete')
    else:
        cls_pred = clustering.kmeans_fit(n_clusters, metric, show_progress=True)

    return clustering, cls_pred

@st.cache_data(max_entries=1, hash_funcs={PhaseClustering: hash_clustering})
def rank_clusters(clustering: PhaseClustering, metric, avg):
    cluster_score = clustering.get_cluster_scores(metric=metric, avg=avg)
    best_cluster_indeces = np.argsort(cluster_score)[::-1]
    return cluster_score, best_cluster_indeces

# Extract Team names
matches_df = pd.read_csv(matches_df_path)
matches_df = matches_df.sort_values('match_date')
all_teams = list(matches_df['home_team'].unique())
all_teams.sort()


# Variables
st.sidebar.markdown('## Settings')
teams = [''] + all_teams
team_name = create_select_box('Team Name', options=teams, default_value='', session_key='team_name')

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'main'

if team_name:

    if st.session_state['mode'] == 'main':
        types = ['Agglomerative', 'K-means']
        type = create_select_box('Clustering Type', options=types, session_key='clustering_type')
        
        n_clusters = create_slider('Num Clusters', min_range=2, max_range=500, default_value=100, session_key='n_clusters')
        
        min_phase_length = create_slider('Min Length', min_range=1, max_range=10, default_value=3, session_key='min_phase_length')

        metrics = ['dtw', 'euclidean']
        metric = create_select_box('Metric', options=metrics, session_key='metric')

        normalized_list = ['min-max', 'z', 'No']
        normalized = create_select_box('Normalization', options=normalized_list, session_key='normalized')

        ranking_metrics = ['Shot', 'xG', 'Goal', 'Length']
        ranking_metric = create_select_box('Ranking Metric', options=ranking_metrics, session_key='ranking_metric')
        avg_metric = st.sidebar.checkbox('Average', value=False)

        num_batches = int((n_clusters / num_rows / num_cols) - 0.5) + 1

        if 'batch' in st.session_state:
            batch = st.session_state['batch'] % num_batches
        else:
            batch = 0
            st.session_state['batch'] = 0   

        # Clustering
        clustering, cls_pred = perform_clustering(type, team_name, n_clusters, min_phase_length, metric, normalized)
        cluster_score, best_cluster_indeces = rank_clusters(clustering, ranking_metric.lower(), avg_metric)
        st.session_state['best_cluster_indeces'] = best_cluster_indeces
        batch_progress = st.progress((batch + 1) / num_batches, text = f'Cluster {batch + 1} / {num_batches}')
        
        cols = st.columns([1, 3 * num_cols - 2, 1])
        
        # st.sidebar.markdown('## Visualization')
        visualization = 1
        
        visualization_list = ['Heatmap', 'Arrows']
        visualization = create_select_box('Visualization', options=visualization_list, session_key='visualization')
        
        with cols[0]:
            prev_button = st.button('Prev', use_container_width=True, disabled=False, on_click=on_prev_button_clicked)        

        with cols[1]:
            loading_progress = st.progress(0, text = f'Loading Batch (0 / {num_cols * num_rows})...')
            
        with cols[2]:
            next_button = st.button('Next', use_container_width=True, disabled=False, on_click=on_next_button_clicked)

        figs = dict()
        for j in range(num_rows):
            row_cols = st.columns([3 for _ in range(num_cols)])
            for i in range(1, 1 + num_cols):
                with row_cols[i - 1]:               
                    col1, col2 = st.columns([2, 1])             
                    index = batch * num_rows * num_cols + j * num_cols + (i - 1)
                    if index < n_clusters:
                        fig, num_phases = plot_cluster(clustering, index, best_cluster_indeces, visualization=visualization, ranking_metric=ranking_metric, avg=avg_metric)
                        with col1:
                            st.button(f'Num Phases: {num_phases}\n{ranking_metric}: {cluster_score[best_cluster_indeces[index]]}',
                                key=f'button_{index}', on_click=on_cluster_button_click, args=(best_cluster_indeces[index],), use_container_width=True)
                        with col2:
                            st.button('Patterns', key=f'temp_{best_cluster_indeces[index]}', use_container_width=True, on_click=on_patterns_button_click, args=(best_cluster_indeces[index],))
                        st.pyplot(fig)
                    loading_progress.progress((j * num_cols + i) / (num_rows * num_cols))
        loading_progress.empty()

    elif st.session_state['mode'] == 'inspect':
        st.sidebar.empty()
        cluster_index = st.session_state['cluster_index']
        clustering: PhaseClustering = st.session_state['clustering']
        phases_in_cluster = clustering.get_cluster_phases(cluster_index)
        
        phase_index = st.session_state['phase_index'] % len(phases_in_cluster)
        phase = phases_in_cluster[phase_index]
        phase = phase.split_locations(location_columns)
        
        st.button('Back to Main Page', on_click=on_back_to_menu_button_clicked)
        first_row = phase.iloc[0]
        # st.markdown(f'Phase ID: {first_row["phase_id"]}')
        # st.markdown(f'Phase UUID: {first_row["id"]}')
        # st.markdown(f'Match ID: {first_row["match_id"]}')
        
        batch_progress = st.progress((phase_index + 1) / len(phases_in_cluster), text = f'Phase {phase_index + 1} / {len(phases_in_cluster)}')
        
        col1, col2, col3 = st.columns([1, 7, 1])

        series_in_cluster = clustering.get_cluster_series(cluster_index)

        with col1:
            prev_button = st.button('Prev', use_container_width=True, disabled=False, on_click=on_prev_button_clicked)
        
        with col3:
            next_button = st.button('Next', use_container_width=True, disabled=False, on_click=on_next_button_clicked)

        with col2:
            visualizer = PhaseVisualizer(figsize=(16, 11))
            fig, ax = visualizer.plot_phase(phase, events_config)
            st.pyplot(fig)
            st.dataframe(phase.get_summary())
            st.write(series_in_cluster[phase_index])
            
    elif st.session_state['mode'] == 'patterns':
        st.sidebar.empty()
        players = load_players(team_name)
        algorithm = st.sidebar.selectbox('Algorithm', ['VMSP', 'CM-SPADE'], on_change=on_pattern_mining_params_change)
        include_events = st.sidebar.checkbox('Include Events', value=True, on_change=on_pattern_mining_params_change)
        include_locations = st.sidebar.checkbox('Include Locations', value=True, on_change=on_pattern_mining_params_change)
        include_players = st.sidebar.checkbox('Include Players', value=True, on_change=on_pattern_mining_params_change)
        # strategy = st.sidebar.selectbox('Representation Strategy', ['Sequential', 'Parallel'])
        support = st.sidebar.slider('Min Support(%)', 0, 100, 10, on_change=on_pattern_mining_params_change)
        max_gap = st.sidebar.slider('Max Gap', 1, 50, 1, disabled=algorithm=='CM-SPADE', on_change=on_pattern_mining_params_change)
        ranking_metric = st.sidebar.selectbox('Ranking Metric', ['Custom', 'Support'], on_change=on_pattern_mining_params_change)
        player_desc = PlayerDescretizer('players', players)
        st.sidebar.button('Back to Main Page', on_click=on_back_to_menu_button_clicked)
        
        all_descs = []
        if include_events:
            all_descs.append(event_desc)
        if include_locations:
            all_descs.append(location_desc)
        if include_players:
            all_descs.append(player_desc)
        
        # multi_desc = MultiParallelDescritizer('multi', all_descs) if strategy == 'Parallel' else MultiSequentialDescritizer('multi', all_descs)
        multi_desc = MultiParallelDescritizer('multi', all_descs)
        
        mapping = multi_desc.get_decode_mapping()
        scores = {key:0.5 for key in mapping}
        start_index = 0
        
        if ranking_metric == 'Custom':
            container = st.container(border=True)
            for desc in all_descs:
                states_num = desc.get_states_num()
                expander = container.expander(desc.name.capitalize())
                cols = expander.columns(4)
                for i in range(0, states_num, len(cols)):
                    for j in range(len(cols)):
                        if i + j < states_num:
                            with cols[j]:
                                scores[start_index + i + j] = create_slider(mapping[start_index + i + j], 0.0, 1.0,
                                                                    scores[start_index + i + j], session_key=mapping[start_index + i + j], container=st,
                                                                    onchange=on_ranking_sliders_change)
                start_index += states_num

        cluster_index = st.session_state['cluster_index']
        clustering: PhaseClustering = st.session_state['clustering']
        phases_in_cluster = clustering.get_cluster_phases(cluster_index)
        
        writer = CMSPADEWriter()
        # writer.write(multi_desc.apply(phases_in_cluster, mode=strategy.casefold()), 'output.tmp')
        writer.write(multi_desc.apply(phases_in_cluster, mode='parallel'), 'output.tmp')
        
        df = None
        args = None
        result_queue = multiprocessing.Queue()

        if 'no_mining' not in st.session_state:
            if algorithm == 'VMSP':
                # df = run_miner(algorithm="VMSP", input_filename="output.tmp", output_filename="output.txt", arguments=[f"{support}%", "100", f"{max_gap}"])
                args = ("VMSP", "output.tmp", "output.txt", [f"{support}%", "100", f"{max_gap}"], result_queue)

            else:
                args = ("CM-SPADE", "output.tmp", "output.txt", [f"{support}%"], result_queue)
                # df = run_miner(algorithm="CM-SPADE", input_filename="output.tmp", output_filename="output.txt", arguments=[f"{support}%"])

            miner_process = multiprocessing.Process(target=run_miner_process, args=args)
            st.session_state['miner_process'] = miner_process
            miner_process.start()
            print(miner_process.pid)
            df = result_queue.get()
            print('Process Started')
            miner_process.join()
            df['sup'] = df['sup'] / len(phases_in_cluster)
            print('returned from process')
            
            del st.session_state['miner_process']
            print('Got Output')
        else:
            df = st.session_state['miner_df']
            del st.session_state['no_mining']
        
        st.session_state['miner_df'] = df.copy()
        
        print('Ranking Patterns...')
        df = rank_patterns(df, scores, mapping, ranking_metric)
        
        st.write(df)