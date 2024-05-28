from mplsoccer import Pitch
from .entities import Phase

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

    def plot_phase(self, phase: Phase, events_config):
        # pitch = Pitch(pitch_type=self.pitch_type, pitch_color=self.pitch_color, line_color=self.line_color)
        # fig, ax = pitch.draw(self.figsize, constrained_layout=self.constrained_layout, tight_layout=self.tight_layout)
        # fig.set_facecolor(self.face_color)
        if not phase.is_splited():
            raise ValueError('Phase must be splited before being visualized!')
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=False, tight_layout=True)
        fig.set_facecolor('#22312b')


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

        return fig, ax