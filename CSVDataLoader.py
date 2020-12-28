import pandas as pd
import numpy as np
import os
import os.path
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchvision import transforms, utils

csv_read_path = r"Data\FaceTracker\preprocessed\csv"

# ein sample ist eine ganze sequence
# get_item sollte eine ganze sequenz zurückgeben!


class AUDataset(Dataset):
    # patents directory angeben anstatt einer file
    # array mit allen datei-namen

    # def __init__(self, parent_folder, transform=None, sequence_length):
    # self.sequence_length = sequnce_length

    # fürs padding
    # def define_sequence_length

    # gibt eine ganze sequence zurück
    # def __getitem__(self, idx):

    def define_longest_sequence(self):
        self.max_sequence_length = 0
        for one_csv in self.csv_names_list:
            df = pd.read_csv(self.csv_directory + "/" + one_csv)
            csv_length = len(df)

            if self.max_sequence_length < csv_length:
                self.max_sequence_length = csv_length
            else:
                self.max_sequence_length = self.max_sequence_length

        return self.max_sequence_length,

    def __init__(self, csv_directory, transform=None):
        """
        :param csv_directory (string): Path to csv-file directory
        :param transform: transforms applied on sample
        """
        self.csv_directory = csv_directory
        self.csv_names_list = os.listdir(self.csv_directory)
        self.transform = transform
        self.all_data_tensor_list = []
        # clean up the list (only use "_fill"-files
        for csv in self.csv_names_list:
            if not "_fill" in csv:
                self.csv_names_list.remove(csv)

        # number of sequences is amount of correct csv_files
        self.number_of_sequences = len(self.csv_names_list)
        self.max_sequence_length = self.define_longest_sequence()

        # for one_csv in csv_names_list:
        #     df = pd.read_csv(csv_directory + "/" + one_csv)
        #     csv_length = len(df)
        #     csv_number_aus = len(df.columns) - 1
        #
        #     self.max_sequence_length = csv_length if csv_length > self.max_sequence_length else self.max_sequence_length = self.max_sequence_length
        #
        #     csv_frames = df.iloc[0:csv_length, 0].values
        #     csv_audata = df.iloc[0:csv_length, 1:].values
        #
        #     sequence_tensor = torch.tensor(csv_audata, dtype=torch.float64)

    # dafür sequence length
    def __getitem__(self, idx):
        current_sequence = self.csv_names_list[idx]
        df = pd.read_csv(self.csv_directory + "/" + current_sequence)
        csv_data = df.iloc[0:len(df), 1:].values
        sequence_tensor = torch.tensor(csv_data, dtype=torch.float32)

        # print("in get item:", sequence_tensor.type())

        # evtl quatsch
        if self.transform is not None:
            sequence_tensor = self.transform(sequence_tensor)


        self.all_data_tensor_list.append(sequence_tensor)
        return sequence_tensor, str(current_sequence)
        # return self.csv_audata[idx]
        # return self.au_tensor[idx]

    def __len__(self):
        return self.number_of_sequences


class PadSequencer:
    def __call__(self, data):
        # sorted_data = sorted(data, key=len, reverse=True)
        # # print("sorted_data:", sorted_data[0].size())
        # padded_data = pad_sequence(sorted_data, batch_first=True)
        # lengths = torch.LongTensor([len(x) for x in sorted_data])
        # # print("Length:", lengths[0])
        # # print("padded_data:", padded_data[0].size())
        # return padded_data, lengths


        # mhm seems to work like this!
        sorted_data = sorted(data, key=lambda x: x[0].shape[0], reverse=True)
        sequences = [x[0] for x in sorted_data]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.LongTensor([len(x) for x in sequences])
        names = [x[1] for x in sorted_data]
        return sequences_padded, lengths, names


if __name__ == "__main__":
    pass
    # Test = AUDataset(csv_read_path)
    # # for i in range(len(Test.csv_names_list)):
    # #     print(f"{Test[i].size()}")
    #
    # # padding = PadSequencer()
    # # padding(Test)
    # # exit()
    #
    # testloader = DataLoader(Test, batch_size=4, collate_fn=PadSequencer(), shuffle=True, num_workers=0)
    #
    # # innerhalb von sequence sind so viele sequences wie in batch_size angegeben
    # # ich denke das soll so sein?
    # # länge beinhaltet alle orginal längen des batches!
    #
    # dataiter = iter(testloader)
    # sequence, lengths = dataiter.next()
    #
    # print("sequence:", sequence)
    # print("sequence.size():", sequence.size())
    # print("lengths:", lengths)

