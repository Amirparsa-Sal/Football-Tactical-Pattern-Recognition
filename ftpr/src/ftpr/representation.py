from .entities import Phase
from typing import List, Iterator, Iterable, Dict
from abc import ABC, abstractmethod
import pandas as pd

class Descretizer(ABC):

    def __init__(self, name) -> None:
        if not name:
            raise ValueError('Name parameter must not be None!')
        self.name = name
    
    @abstractmethod
    def encode(self, phase: Phase, index: int, offset=0) -> Iterable:
        pass 
    
    @abstractmethod
    def get_states_num(self) -> int:
        pass

    @abstractmethod
    def get_decode_mapping(self, offset=0) -> int:
        pass
    
    def apply(self, phases: List[Phase], mode='sequential') -> Iterator[Iterable]:
        if mode not in ['sequential', 'parallel']:
            raise ValueError(f'Mode paramter must be in [sequential, parallel]')
        
        if mode == 'parallel':
            for phase in phases:
                sequence = []
                for i in range(len(phase)):
                    sequence.append(self.encode(phase, i))
                yield sequence
        elif mode == 'sequential':
            for phase in phases:
                sequence = []
                for i in range(len(phase)):
                    sequence.extend(self.encode(phase, i))
                yield sequence
    
class Writer(ABC):

    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def write(self, descretizer: Descretizer, phases: List[Phase], filepath: str):
        pass
        
class MultiDescretizer(Descretizer):

    def __init__(self, name: str, descretizers: List[Descretizer]):
        super().__init__(name)
        if not isinstance(descretizers, list):
            raise ValueError('Descretizers argument must be a list.')
        if len(descretizers) == 0:
            raise ValueError('Descretizers list must not be empty!')
        self.descretizers = descretizers

    @abstractmethod
    def encode(self, phase: Phase, index: int, offset=0) -> Iterable:
        pass
    
    def get_decode_mapping(self, offset=0) -> int:
        mapping = self.descretizers[0].get_decode_mapping(offset=0)
        current_offset = self.descretizers[0].get_states_num()
        for i in range(1, len(self.descretizers)):
            mapping = mapping | self.descretizers[i].get_decode_mapping(offset=current_offset)
            current_offset += self.descretizers[i].get_states_num()
        return {key+offset:value for key,value in mapping.items()}
    
    def get_states_num(self) -> int:
        return sum(list(map(self.descretizers, lambda d: d.get_states_num())))   


class MultiSequentialDescritizer(MultiDescretizer):
    
    def __init__(self, name: str, descretizers: List[Descretizer]):
        super().__init__(name, descretizers)
        
    def encode(self, phase: Phase, index: int, offset=0):
        result = []
        result.append(self.descretizers[0].encode(phase, index))
        current_offset = self.descretizers[0].get_states_num()
        for i in range(1, len(self.descretizers)):
            result.append(self.descretizers[i].encode(phase, index, offset=current_offset))
            current_offset += self.descretizers[i].get_states_num()
        for i, itemset in enumerate(result):
            result[i] = tuple(value + offset for value in itemset)
        return result
    
class MultiParallelDescritizer(MultiDescretizer):
    
    def __init__(self, name: str, descretizers: List[Descretizer]):
        super().__init__(name, descretizers)
        
    def encode(self, phase: Phase, index: int, offset=0) -> Iterable:
        itemset = self.descretizers[0].encode(phase, index)
        current_offset = self.descretizers[0].get_states_num()
        for i in range(1, len(self.descretizers)):
            itemset = itemset + self.descretizers[i].encode(phase, index, offset=current_offset)
            current_offset += self.descretizers[i].get_states_num()
        return tuple(value + offset for value in itemset)
    
class EventDescretizer(Descretizer):

    def __init__(self, name: str, events: List[str], event_types:Dict[str, List[str]] = None, event_column='type', event_type_sep='_') -> None:
        super().__init__(name)
        self.all_event_to_index = dict()
        self.just_event_to_index = dict()
        self.event_type_sep = event_type_sep
        self.index_to_event = dict()
        current_index = 0
        for event in events:
            event_low = event.lower()
            self.just_event_to_index[event_low] = current_index
            self.index_to_event[current_index] = event_low
            if event_types and event in event_types:
                current_index += 1
                self.all_event_to_index[event_low] = dict()
                for type in event_types[event]:
                    type_low = type.lower()
                    self.all_event_to_index[event_low][type_low] = current_index
                    self.index_to_event[current_index] = f'{event_low}{self.event_type_sep}{type_low}'
                    current_index += 1
            else:
                self.all_event_to_index[event_low] = current_index
                self.index_to_event[current_index] = event_low
                current_index += 1
                
        self.event_column = event_column
        self.events = events

    def encode(self, phase: Phase, index: int, offset=0):
        event = phase.iloc[index][self.event_column].lower()
        if event in self.just_event_to_index:
            itemset = (self.just_event_to_index[event] + offset, )
            if isinstance(self.all_event_to_index[event], dict):
                type = phase.iloc[index][f'{event}_type']
                if not pd.isna(type):
                    itemset = itemset + (self.all_event_to_index[event][type.lower()] + offset,)
            return itemset
        raise ValueError(f'Event ({event}) is not defined in events list.')
    
    def get_index(self, event: str):
        if not event in self.event_to_index:
            raise ValueError(f'Event ({event}) is not defined in events list.')
        return self.event_to_index[event]

    def get_decode_mapping(self, offset=0) -> int:
        return {key+offset:value for key,value in self.index_to_event.items()}
    
    def get_states_num(self) -> int:
        return len(self.index_to_event.keys())
    
class LocationDescretizer(Descretizer):

    zone_to_index = {
        "Left Flank": 0,
        "Right Flank": 1,
        "Own box": 2,
        "Opposition box": 3,
        "Midfield": 4
    }

    index_to_zone = {value:key for key, value in zone_to_index.items()}

    def __init__(self, name: str, max_length=120, max_width=80, flanks_width=18, arc_x=22, event_column='type', location_column='location') -> None:
        super().__init__(name)
        self.flank1 = flanks_width
        self.flank2 = max_width - flanks_width
        self.arc1 = arc_x
        self.arc2 = max_length - arc_x
        self.event_column = event_column
        self.location_column = location_column

    def get_zone_index(self, x, y):
        if y < self.flank1: return self.zone_to_index['Left Flank']
        if y > self.flank2: return self.zone_to_index['Right Flank']
        if x < self.arc1: return self.zone_to_index['Own box']
        if x > self.arc2: return self.zone_to_index['Opposition box']
        return self.zone_to_index['Midfield']
    
    def encode(self, phase: Phase, index: int, offset=0):
        locations = phase.get_location(index)
        return (self.get_zone_index(locations[0], locations[1]) + offset, )

    def get_decode_mapping(self, offset=0) -> int:
        return {key+offset: value for key, value in self.index_to_zone.items()}
    
    def get_states_num(self) -> int:
        return 5

class CMSPADEWriter(Writer):

    def __init__(self, descretizer_mode='sequential', item_sep=' ', itemset_sep='-1', sequence_sep='-2') -> None:
        super().__init__()
        self.item_sep = item_sep
        self.itemset_sep = itemset_sep
        self.sequence_sep = sequence_sep
        self.descretizer_mode = descretizer_mode
    
    def write(self, descretizer: Descretizer, phases: List[Phase], filepath: str):
        if not descretizer:
            raise ValueError('Descretizer argument can not be none!')
        self.write(descretizer.apply(phases, mode=self.descretizer_mode), filepath)
    
    def write(self, sequence_gen, filepath: str):
        with open(filepath, 'w') as f: 
            for sequence in sequence_gen:
                line = ""
                for itemset in sequence:
                    for item in itemset:
                        line += str(item)
                        line += self.item_sep
                    line += f"{self.itemset_sep} "
                line += f"{self.sequence_sep}\n"
                f.write(line)
