from .entities import Phase
import multiprocessing
from typing import List, Iterable

def phase_generator(df, filter_static_events=True, min_phase_length=2, phase_id_col='phase_id') -> Iterable[Phase]:
    if phase_id_col not in df:
        raise ValueError(f'The dataframe does not have a column named {phase_id_col}')
    
    phase_id = 0
    start = 0
    current = 0
    while current < len(df):
        current_phase_id = df.iloc[current, -1]
        if current_phase_id == phase_id:
            current += 1
        else:
            phase = Phase(df[start:current])
            if filter_static_events:
                phase = phase.filter_static_events()
            if len(phase) >= min_phase_length:
                yield phase
            phase_id += 1
            start = current
    
    if start < len(df):
        if filter_static_events:
            phase = phase.filter_static_events()
        if len(phase) >= min_phase_length:
            yield Phase(df[start: current])

def load_phase_worker(df, filter_static_events=True, min_phase_length=2):
    phase_id = df.iloc[0, -1]
    start = 0
    current = 0
    phases = []
    while current < len(df):
        current_phase_id = df.iloc[current, -1]
        if current_phase_id == phase_id:
            current += 1
        else:
            phase = Phase(df[start:current])
            if filter_static_events:
                phase = phase.filter_static_events()
            if len(phase) >= min_phase_length:
                phases.append(phase)
            phase_id += 1
            start = current
    
    if start < len(df):
        phase = Phase(df[start:current])
        if filter_static_events:
            phase = phase.filter_static_events()
        if len(phase) >= min_phase_length:
            phases.append(phase)
    return phases

def load_phases(df, filter_static_events=True, min_phase_length=2, n_jobs=1, phase_id_col='phase_id') -> List[Phase]:
    if phase_id_col not in df:
        raise ValueError(f'The dataframe does not have a column named {phase_id_col}')
    
    if n_jobs == 1:
        return load_phase_worker(df, filter_static_events, min_phase_length)
    
    slice_length = len(df) // n_jobs
    df_slices = []
    start = 0
    current = 0
    while current < len(df):
        # go at least {slice_length} forward
        current += slice_length
        if current > len(df):
            current = len(df)
        else:
            # go forward until the phase is finished
            current_phase_id = df.iloc[current][phase_id_col]
            while current < len(df) and df.iloc[current][phase_id_col] == current_phase_id: current += 1
        df_slices.append(df[start: current])
        start = current

    filter_statics = [filter_static_events for _ in range(len(df_slices))]
    min_phase_lengths = [min_phase_length for _ in range(len(df_slices))]
    pool = multiprocessing.Pool(processes=n_jobs)
    process_results = pool.starmap(load_phase_worker, zip(df_slices, filter_statics, min_phase_lengths))
    results = []
    for r in process_results:
        results.extend(r)
    return results

def load_phase(df, id, phase_id_col='phase_id') -> Phase:
    if phase_id_col not in df.columns:
        raise ValueError(f'The dataframe does not have a column named {phase_id_col}')
    if id not in df[phase_id_col]:
        raise ValueError(f'No phase with {phase_id_col}={id} exists!')
    return Phase(df[df[phase_id_col]] == id)