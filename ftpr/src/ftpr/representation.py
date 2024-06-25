from .entities import Phase
from typing import List, Tuple, Iterator
from abc import ABC, abstractmethod

class Descretizer(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def apply(self, phase: Phase, index: int):
        pass    

class PhaseDescretizer:

    def __init__(self, descretizers: List[Descretizer]):
        if not isinstance(descretizers, list):
            raise ValueError('Descretizers argument must be a list.')
        if len(descretizers) == 0:
            raise ValueError('Descretizers list must not be empty!')
        self.descretizers = descretizers

    def descretize(self, phases: List[Phase]) -> Iterator[Tuple]:
        for phase in phases:
            sequence = []
            for i in range(len(phase)):
                itemset = self.descretizers[0].apply(phase, i)
                for j in range(1, len(self.descretizers)):
                    itemset = itemset + self.descretizers[j].apply(phase, i)
                sequence.append(itemset)
            yield sequence


class EventDescretizer(Descretizer):

    def __init__(self, events: List[str], event_column='type') -> None:
        super().__init__()
        self.event_to_index = dict()
        for i, event in enumerate(events):
            self.event_to_index[event.lower()] = i
        self.event_column = event_column

    def apply(self, phase: Phase, index: int):
        event = phase.iloc[index][self.event_column].lower()
        if event in self.event_to_index:
            return (self.event_to_index[event], )
        raise ValueError(f'Event ({event}) is not defined in events list.')
    
    def get_index(self, event: str):
        if not event in self.event_to_index:
            raise ValueError(f'Event ({event}) is not defined in events list.')
        return self.event_to_index[event]

class LocationDescretizer(Descretizer):

    index_to_zone = {
        "Left Flank": 1,
        "Right Flank": 2,
        "Own box": 3,
        "Opposition box": 4,
        "Midfield": 5
    }

    def __init__(self, dynamic_events, max_length=120, max_width=80, flanks_width=18, arc_x=22, event_column='type', location_column='location') -> None:
        super().__init__()
        self.flank1 = flanks_width
        self.flank2 = max_width - flanks_width
        self.arc1 = arc_x
        self.arc2 = max_length - arc_x
        self.event_column = event_column
        self.location_column = location_column
        self.dynamic_events = dynamic_events

    def get_zone_index(self, x, y):
        if y < self.flank1: return 1
        if y > self.flank2: return 2
        if x < self.arc1: return 3
        if x > self.arc2: return 4
        return 5
    
    def apply(self, phase: Phase, index: int):
        event = phase.iloc[index][self.event_column]
        locations = phase.get_location(index)
        if event in self.dynamic_events:
            locations = locations + phase.get_location(index, event=event)
        itemset = (self.get_zone_index(locations[0], locations[1]), )
        if len(locations) == 4:
            itemset = itemset + (self.get_zone_index(locations[2], locations[3]), )
        else:
            itemset = itemset + (0, )
        return itemset
