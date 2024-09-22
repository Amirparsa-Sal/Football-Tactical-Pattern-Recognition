import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import argparse
import os
from ftpr.entities import Phase
from ftpr.visualization import PhaseVisualizer
from collections import Counter

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

data_dir = '../data/team_phases/'
matches_df_dir = '../data/matches.csv'

matches_df = pd.read_csv(matches_df_dir)
matches_df = matches_df.sort_values('match_date')
all_teams = list(matches_df['home_team'].unique())
all_teams.sort()


def get_match_info(df, i):
    home_team = df.iloc[i, df.columns.get_loc('home_team')]
    away_team = df.iloc[i, df.columns.get_loc('away_team')]
    match_date = df.iloc[i, df.columns.get_loc('match_date')]
    match_id = df.iloc[i, df.columns.get_loc('match_id')]
    return home_team, away_team, match_date, match_id

@st.cache_data(max_entries=1)
def extract_matches(team_name):
    matches_df['target'] = (matches_df['home_team'] == team_name) + (matches_df['away_team'] == team_name)
    selected_matches = matches_df[matches_df['target']]
    all_matches = []
    all_match_ids = []
    for i in range(len(selected_matches)):
        home, away, date, match_id = get_match_info(selected_matches, i)
        all_matches.append(f'{i + 1}- {home} vs. {away} ({date})')
        all_match_ids.append(match_id)
    return all_matches, all_match_ids

@st.cache_data(max_entries=1)
def read_match_csv_cache(path, match_id):
    team_phases_df =  pd.read_csv(path)
    match_df = team_phases_df[team_phases_df['match_id'] == match_id]
    phase_to_len = Counter(match_df['phase_id'])
    phase_lengths = extract_phase_lengths(phase_to_len)
    return team_phases_df, match_df, phase_lengths, phase_to_len

def extract_phase_lengths(phase_to_len):
    lengths = []
    for _, value in phase_to_len.items():
        if len(lengths) == 0:
            lengths.append(value)
            continue
        i = 0
        while i < len(lengths) and lengths[i] < value: i += 1

        if i < len(lengths) and lengths[i] == value: 
              continue
        lengths.insert(i, value)
    return lengths

def on_prev_button_clicked():
    st.session_state['phase_index'] = st.session_state['phase_index'] - 1

def on_next_button_clicked():
    st.session_state['phase_index'] = st.session_state['phase_index'] + 1

# Team name
st.sidebar.markdown('## Select a team')
team_name = st.sidebar.selectbox('Choose', [''] + all_teams)

if team_name:
    # Match
    st.sidebar.markdown('## Select a match')

    all_matches, all_match_ids = extract_matches(team_name)
    match = st.sidebar.selectbox('Choose', all_matches)
    match_id = all_match_ids[int(match.split('-')[0]) - 1]

    # Phase Length
    st.sidebar.markdown('## Select phase length range')

    team_csv_dir = os.path.join(data_dir, f'{team_name}.csv')
    team_phases_df, match_phases_df, phase_lengths, phase_to_len = read_match_csv_cache(team_csv_dir, match_id)
    phase_len = st.sidebar.select_slider('Phase length', phase_lengths, phase_lengths[0])

    selected_phase_ids = [key for key, value in phase_to_len.items() if value == phase_len]

    # Selecting the current phase
    selected_phases = match_phases_df[match_phases_df['phase_id'].isin(selected_phase_ids)]
    if 'phase_index' in st.session_state:
        phase_index = st.session_state['phase_index'] % len(selected_phase_ids)
    else:
        phase_index = 0
        st.session_state['phase_index'] = 0
    
    phase_df = selected_phases[selected_phases['phase_id'] == selected_phase_ids[phase_index]] 
    phase = Phase(phase_df)
    phase = phase.filter_static_events()
    phase = phase.split_locations(location_columns)

    progress = st.progress((phase_index + 1) / len(selected_phase_ids), text = f'Phase {phase_index + 1} / {len(selected_phase_ids)}')
    st.markdown(f'Phase ID: {selected_phase_ids[phase_index]}')
    st.markdown(f'Match ID: {match_id}')

    col1, col2, col3 = st.columns([1, 10, 1])

    with col1:
        prev_button = st.button('<', use_container_width=True, disabled=False, on_click=on_prev_button_clicked)
    
    with col3:
        next_button = st.button('>', use_container_width=True, disabled=False, on_click=on_next_button_clicked)

    with col2:
        visualizer = PhaseVisualizer(figsize=(16, 11))
        fig, ax = visualizer.plot_phase(phase, events_config)
        st.pyplot(fig)
        st.dataframe(phase.get_summary())