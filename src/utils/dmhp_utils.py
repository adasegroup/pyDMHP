import numpy as np
import pandas as pd
import time
from typing import Dict


def load_sequences_csv(df: pd.DataFrame, domain_names: Dict):
    """
    Load event sequences from a csv file
    :param file_name: the path and name of the target csv file
    :param domain_names: a dictionary contains the names of the key columns
                         corresponding to {'seq_id', 'time', 'event'}
        The format should be
        domain_names = {'seq_id': the column name of sequence name,
                        'time': the column name of timestamps,
                        'event': the column name of events}
    :return: database: a dictionary containing observed event sequences
        database = {'event_features': None,
                    'type2idx': a Dict = {'event_name': event_index}
                    'idx2type': a Dict = {event_index: 'event_name'}
                    'seq2idx': a Dict = {'seq_name': seq_index}
                    'idx2seq': a Dict = {seq_index: 'seq_name'}
                    'sequences': a List  = [seq_1, seq_2, ..., seq_N].
                    }
        For the i-th sequence:
        seq_i = {'times': (N,) float array of timestamps, N is the number of events.
                 'events': (N,) int array of event types.
                 'seq_feature': None.
                 't_start': a float number, the start timestamp of the sequence.
                 't_stop': a float number, the stop timestamp of the sequence.
                 'label': None
                 }
    """
    database = {
        "event_features": None,
        "type2idx": None,
        "idx2type": None,
        "seq2idx": None,
        "idx2seq": None,
        "sequences": [],
    }

    type2idx = {}
    idx2type = {}
    seq2idx = {}
    idx2seq = {}

    start = time.time()
    seq_idx = 0
    type_idx = 0
    for i, row in df.iterrows():
        seq_name = str(row[domain_names["seq_id"]])
        event_type = str(row[domain_names["event"]])
        if seq_name not in seq2idx.keys():
            seq2idx[seq_name] = seq_idx
            seq = {
                "times": [],
                "events": [],
                "seq_feature": None,
                "t_start": 0.0,
                "t_stop": 0.0,
                "label": None,
            }
            database["sequences"].append(seq)
            seq_idx += 1

        if event_type not in type2idx.keys():
            type2idx[event_type] = type_idx
            type_idx += 1

    for seq_name in seq2idx.keys():
        seq_idx = seq2idx[seq_name]
        idx2seq[seq_idx] = seq_name

    for event_type in type2idx.keys():
        type_idx = type2idx[event_type]
        idx2type[type_idx] = event_type

    database["type2idx"] = type2idx
    database["idx2type"] = idx2type
    database["seq2idx"] = seq2idx
    database["idx2seq"] = idx2seq

    for i, row in df.iterrows():
        seq_name = str(row[domain_names["seq_id"]])
        timestamp = float(row[domain_names["time"]])
        event_type = str(row[domain_names["event"]])

        seq_idx = database["seq2idx"][seq_name]
        type_idx = database["type2idx"][event_type]
        database["sequences"][seq_idx]["times"].append(timestamp)
        database["sequences"][seq_idx]["events"].append(type_idx)

    for n in range(len(database["sequences"])):
        database["sequences"][n]["t_start"] = database["sequences"][n]["times"][0]
        database["sequences"][n]["t_stop"] = (
            database["sequences"][n]["times"][-1] + 1e-2
        )
        database["sequences"][n]["times"] = np.asarray(
            database["sequences"][n]["times"]
        )
        database["sequences"][n]["events"] = np.asarray(
            database["sequences"][n]["events"]
        )

    return database


def samples2dict(samples, device, Cs, FCs):
    """
    Convert a batch sampled from dataloader to a dictionary
    :param samples: a batch of data sampled from the "dataloader" defined by EventSampler
    :param device: a string representing usable CPU or GPU
    :param Cs: 'Cs': (num_type, 1) LongTensor containing all events' index
    :param FCs: 'FCs': None or a (num_type, Dc) FloatTensor representing all events' features
    :return:
        ci: events (batch_size, 1) LongTensor indicates each event's type in the batch
        batch_dict = {
            'ci': events (batch_size, 1) LongTensor indicates each event's type in the batch
            'cjs': history (batch_size, memory_size) LongTensor indicates historical events' types in the batch
            'ti': event_time (batch_size, 1) FloatTensor indicates each event's timestamp in the batch
            'tjs': history_time (batch_size, memory_size) FloatTensor represents history's timestamps in the batch
            'sn': sequence index (batch_size, 1) LongTensor
            'fsn': features (batch_size, dim_feature) FloatTensor contains feature vectors of the sequence in the batch
            'fci': current_feature (batch_size, Dc) FloatTensor of current feature
            'fcjs': history_features (batch_size, Dc, memory_size) FloatTensor of historical features
            'Cs': the input Cs
            'FCs': the input FCs
            }
    """
    ti = samples[0].to(device)
    tjs = samples[1].to(device)
    ci = samples[2].to(device)
    cjs = samples[3].to(device)
    sn = samples[4].to(device)
    if len(samples) == 5:
        fsn = None
        fci = None
        fcjs = None
    elif len(samples) == 6:
        fsn = samples[5].to(device)
        fci = None
        fcjs = None
    elif len(samples) == 7:
        fsn = None
        fci = samples[5].to(device)
        fcjs = samples[6].to(device)
    else:
        fsn = samples[5].to(device)
        fci = samples[6].to(device)
        fcjs = samples[7].to(device)

    batch_dict = {'ti': ti,
                  'tjs': tjs,
                  'ci': ci,
                  'cjs': cjs,
                  'sn': sn,
                  'fsn': fsn,
                  'fci': fci,
                  'fcjs': fcjs,
                  'Cs': Cs,
                  'FCs': FCs}
    return ci, batch_dict