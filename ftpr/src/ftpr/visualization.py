from mplsoccer import Pitch
from .entities import Phase
from .utils import get_interpolated_series
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class PhaseVisualizer:

    def __init__(self, figsize,
                pitch_type='statsbomb', 
                pitch_color='#22312b', 
                line_color='#c7d5cc', 
                face_color='#22312b', 
                constrained_layout=False, 
                tight_layout=True,
                annotation_fontsize=15,
                annotation_fontcolor='white',
                legend_loc='upper left',
                legend_labelspacing = 1.5) -> None:
        self.figsize = figsize
        self.pitch_type = pitch_type
        self.pitch_color = pitch_color
        self.line_color = line_color
        self.face_color = face_color
        self.constrained_layout = constrained_layout
        self.tight_layout = tight_layout
        self.annotation_fontsize = annotation_fontsize
        self.annotation_fontcolor = annotation_fontcolor
        self.legend_loc = legend_loc
        self.legend_labelspacing = legend_labelspacing

    def __plot_path(self, pitch, ax, df, event_type, color='#000000', dashed=False):
        if dashed:
            return pitch.lines(df.location_x, df.location_y, df[f'{event_type.lower()}_end_location_x'], 
                            df[f'{event_type.lower()}_end_location_y'], lw=5, transparent=True, 
                            label=f'{event_type.capitalize()}', color=color, ax=ax, linestyle='--')
            
        return pitch.arrows(df.location_x, df.location_y, df[f'{event_type.lower()}_end_location_x'], 
                            df[f'{event_type.lower()}_end_location_y'], label=f'{event_type.capitalize()} Path',
                        color=color, ax=ax, width=3)

    def add_phase(self, pitch, ax, phase, events_config):
        event_types = phase['type'].unique()
        for et in event_types:
            df_event = phase[phase['type'] == et]
            if phase.is_splited():
                pitch.scatter(df_event.location_x, df_event.location_y, s=600, ax=ax, label=et)

        for event in events_config:
            df_event = phase[phase['type'] == event]
            self.__plot_path(pitch, ax, df_event, event, 
                             color=events_config[event].get('color', '#00ffff'),
                             dashed=events_config[event].get('dashed', False)
            )

        # Plot the order of events
        location_x, location_y =  phase.get_location(0)
        pitch.annotate('1', (location_x, location_y), color=self.annotation_fontcolor, fontsize=self.annotation_fontsize, ax=ax,
                       va='center', ha='center')

        for i in range(1, len(phase)):
            new_location_x, new_location_y = phase.get_location(i)
            if new_location_x != location_x or new_location_y != location_y:
                pitch.annotate(f'{i + 1}', (new_location_x, new_location_y), va='center', ha='center',
                                color=self.annotation_fontcolor, fontsize=self.annotation_fontsize, ax=ax)
            location_x, location_y =  new_location_x, new_location_y
        
        ax.legend(loc='upper left', labelspacing=1.5)
        return ax

    def plot_phase(self, phase: Phase, events_config):
        if not phase.is_splited():
            raise ValueError('Phase must be splited before being visualized!')
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        fig.set_facecolor('#22312b')

        ax = self.add_phase(pitch, ax, phase, events_config)

        return fig, ax
    
    
    def plot_heatmap(self, pitch, series, bins, statistic='count', gaussian=True, cbar=False, cbar_outline_color='#efefef', cbar_text_color='#efefef', cmap='hot', edgecolors='#22312b'):
        new_series = []
        for s in series:
            new_series.extend(get_interpolated_series(s))
        x = [t[0] for t in new_series]
        y = [t[1] for t in new_series]
        fig, ax = pitch.draw(figsize=(6.6, 4.125))
        fig.set_facecolor('#22312b')
        bin_statistic = pitch.bin_statistic(x, y, statistic=statistic, bins=bins)
        if gaussian:
            bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, edgecolors=edgecolors)
        if cbar:
            cb = fig.colorbar(pcm, ax=ax, shrink=0.6)
            cb.outline.set_edgecolor(cbar_outline_color)
            cb.ax.yaxis.set_tick_params(color='#efefef')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=cbar_text_color)
        return fig, ax
