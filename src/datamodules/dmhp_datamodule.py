from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import os
import torch
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.utils.dmhp_utils import load_sequences_csv
from src.utils.data_utils import download_unpack_zip, load_data


class EventSampler(Dataset):
    """Load event sequences via minibatch"""

    def __init__(self, database, memorysize):
        """
        :param database: the observed event sequences
            database = {'event_features': None or (C, De) float array of event's static features,
                                      C is the number of event types.
                        'type2idx': a Dict = {'event_name': event_index}
                        'idx2type': a Dict = {event_index: 'event_name'}
                        'seq2idx': a Dict = {'seq_name': seq_index}
                        'idx2seq': a Dict = {seq_index: 'seq_name'}
                        'sequences': a List  = {seq_1, seq_2, ..., seq_N}.
                        }
            For the i-th sequence:
            seq_i = {'times': (N,) float array of timestamps, N is the number of events.
                     'events': (N,) int array of event types.
                     'seq_feature': None or (Ds,) float array of sequence's static feature.
                     't_start': a float number indicating the start timestamp of the sequence.
                     't_stop': a float number indicating the stop timestamp of the sequence.
                     'label': None or int/float number indicating the labels of the sequence}
        :param memorysize: how many historical events remembered by each event
        """
        self.event_cell = []
        self.time_cell = []
        self.database = database
        self.memory_size = memorysize
        for i in range(len(database["sequences"])):
            seq_i = database["sequences"][i]
            times = seq_i["times"]
            events = seq_i["events"]
            t_start = seq_i["t_start"]
            for j in range(len(events)):
                target = events[j]
                # former = np.zeros((memorysize,), dtype=np.int)
                # former = np.random.permutation(len(self.database['type2idx']))
                # former = former[:memorysize]
                former = np.random.choice(len(self.database["type2idx"]), memorysize)
                target_t = times[j]
                former_t = t_start * np.ones((memorysize,))

                if 0 < j < memorysize:
                    former[-j:] = events[:j]
                    former_t[-j:] = times[:j]
                elif j >= memorysize:
                    former = events[j - memorysize : j]
                    former_t = times[j - memorysize : j]

                self.event_cell.append((target, former, i))
                self.time_cell.append((target_t, former_t))

    def __len__(self):
        return len(self.event_cell)

    def __getitem__(self, idx):
        current_time = torch.Tensor([self.time_cell[idx][0]])  # torch.from_numpy()
        current_time = current_time.type(torch.FloatTensor)
        history_time = torch.from_numpy(self.time_cell[idx][1])
        history_time = history_time.type(torch.FloatTensor)

        current_event_numpy = self.event_cell[idx][0]
        current_event = torch.Tensor([self.event_cell[idx][0]])
        current_event = current_event.type(torch.LongTensor)
        history_event_numpy = self.event_cell[idx][1]
        history_event = torch.from_numpy(self.event_cell[idx][1])
        history_event = history_event.type(torch.LongTensor)

        current_seq_numpy = self.event_cell[idx][2]
        current_seq = torch.Tensor([self.event_cell[idx][2]])
        current_seq = current_seq.type(torch.LongTensor)

        if (
            self.database["sequences"][current_seq_numpy]["seq_feature"] is None
            and self.database["event_features"] is None
        ):
            return (
                current_time,
                history_time,
                current_event,
                history_event,
                current_seq,
            )  # 5 outputs

        elif (
            self.database["sequences"][current_seq_numpy]["seq_feature"] is not None
            and self.database["event_features"] is None
        ):
            seq_feature = self.database["sequences"][current_seq_numpy]["seq_feature"]
            seq_feature = torch.from_numpy(seq_feature)
            seq_feature = seq_feature.type(torch.FloatTensor)

            return (
                current_time,
                history_time,
                current_event,
                history_event,
                current_seq,
                seq_feature,
            )  # 6 outputs

        elif (
            self.database["sequences"][current_seq_numpy]["seq_feature"] is None
            and self.database["event_features"] is not None
        ):
            current_event_feature = self.database["event_features"][
                :, current_event_numpy
            ]
            current_event_feature = torch.from_numpy(current_event_feature)
            current_event_feature = current_event_feature.type(torch.FloatTensor)

            history_event_feature = self.database["event_features"][
                :, history_event_numpy
            ]
            history_event_feature = torch.from_numpy(history_event_feature)
            history_event_feature = history_event_feature.type(torch.FloatTensor)

            return (
                current_time,
                history_time,
                current_event,
                history_event,
                current_seq,
                current_event_feature,
                history_event_feature,
            )  # 7 outputs
        else:
            seq_feature = self.database["sequences"][current_seq_numpy]["seq_feature"]
            seq_feature = torch.from_numpy(seq_feature)
            seq_feature = seq_feature.type(torch.FloatTensor)

            current_event_feature = self.database["event_features"][
                :, current_event_numpy
            ]
            current_event_feature = torch.from_numpy(current_event_feature)
            current_event_feature = current_event_feature.type(torch.FloatTensor)

            history_event_feature = self.database["event_features"][
                :, history_event_numpy
            ]
            history_event_feature = torch.from_numpy(history_event_feature)
            history_event_feature = history_event_feature.type(torch.FloatTensor)

            return (
                current_time,
                history_time,
                current_event,
                history_event,
                current_seq,
                seq_feature,
                current_event_feature,
                history_event_feature,
            )  # 8 outputs


class DMHPDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "./",
        data_config_yaml: Union[str, Path] = "./",
        maxsize: Optional[int] = None,
        maxlen: int = -1,
        train_val_split: float = 0.8,
        ext: str = "csv",
        time_col: str = "time",
        event_col: str = "event",
        datetime: bool = False,
        batch_size: int = 128,
        num_workers: int = 4,
        memorysize: int = 3,
    ):
        super().__init__()
        self.data_dir = data_dir
        with open(data_config_yaml, "r") as stream:
            self.data_config = yaml.safe_load(stream)
        data_name = self.data_dir.split("/")[-1]
        self.num_clusters = self.data_config[data_name]["num_clusters"]
        self.num_events = self.data_config[data_name]["num_events"]
        self.maxsize = maxsize
        self.maxlen = maxlen
        self.train_val_split = train_val_split
        self.ext = ext
        self.time_col = time_col
        self.event_col = event_col
        self.datetime = datetime
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.memorysize = memorysize
        self.dataset = None
        self.target = None

    def prepare_data(self):
        """
        Script to download data if necessary
        Data transform for conv1d
        """
        if Path(self.data_dir).exists():
            print("Data is already in place")
        else:
            data_name = self.data_dir.split("/")[-1]
            # dictionary with urls to download sequence data
            download_unpack_zip(self.data_config[data_name], self.data_dir)

        print("Transforming data")
        seq_data = pd.DataFrame([])
        all_files_in_datafolder = os.listdir(self.data_dir)

        for file in all_files_in_datafolder:

            # skipping not relevant files
            if file == "all_users.csv" or file == "info.json" or "(" in file:
                continue

            elif file == "clusters.csv":
                # getting ground truth
                gt_ids = pd.read_csv(f"{self.data_dir}/clusters.csv")
                self.target = gt_ids["cluster_id"].tolist()
                continue

            df_loc = pd.read_csv(f"{self.data_dir}/{file}")
            df_loc["file_name"] = [int(file.replace("." + self.ext, ""))] * len(df_loc)
            df_loc = df_loc.iloc[:, 1:]

            seq_data = pd.concat([seq_data, df_loc])

        self.dataset = seq_data

    def setup(self, stage: Optional[str] = None):
        """
        Assign train/val datasets for use in dataloaders
        """
        domain_names = {"seq_id": "file_name", "time": self.time_col, "event": self.event_col}
        if stage == "fit" or stage is None:
            permutation = np.random.permutation(len(self.dataset))
            split = int(self.train_val_split * len(self.dataset))
            permutation[:split]
            self.train_data = load_sequences_csv(
                self.dataset[permutation[:split]], domain_names
            )
            self.val_data = load_sequences_csv(
                self.dataset[permutation[split:]], domain_names
            )

        # Assign test dataset for use in dataloader
        if stage == "test":
            print(len(self.dataset))
            self.test_data = load_sequences_csv(self.dataset, domain_names)

    def train_dataloader(self):
        return DataLoader(
            EventSampler(self.train_data, self.memorysize),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            EventSampler(self.val_data, self.memorysize),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            EventSampler(self.test_data, self.memorysize),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
