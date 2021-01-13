import math
import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# if I end up using normalization:
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence as PAD

from CSVDataLoader import AUDataset
from CSVDataLoader import PadSequencer





csv_read_path = r"Data\FaceTracker\preprocessed\csv"
gen_save_pth = r"Data\GeneratedAnims\\"

# Path to save and load models
net_path = "./models/Transition_net.pth"


# set up the divice (GPU or CPU) via input prompt
def set_device():
    cuda_true = input("Use GPU? (y) or (n)?")
    if cuda_true == "y":
        device = "cuda"
    else:
        device = "cpu"
    print("Device:", device)


# save a networks parameters for future use
def save_network(net: nn.Module):
    # save the network?
    save = input("Save net? (y) or (n)?")
    if save == "y":
        version = input("Input the version ID (int): ")
        try:
            int(version)
        except ValueError:
            print("That is not an int Version number!")
            save_network(net)

        path = "./models/Transition_" + str(version) + "_net.pth"
        torch.save(net.state_dict(), path)
    else:
        pass


# load an existing network's parameters and safe them into the just created net
def load_network(net: nn.Module):
    # save the network?
    load = input("Load Network? (y) or (n)?")

    if load == "y":
        version = input("Which model should be loaded? (Version number): ")
        try:
            int(version)
        except ValueError:
            print("That is not a Version number!")
            version = input("Which model should be loaded? (Version number): ")

        try:
            path = "./models/Transition_" + str(version) + "_net.pth"
            net.load_state_dict(torch.load(path))

        except FileNotFoundError:
            print("There is no such Version yet!")
            load_network(net)
    else:
        pass


# Hyperparameters
num_epochs = 4
train_batch_size = 1
test_batch_size = 1
learning_rate = 1e-3

# input_size = ?
# sequence_length = ?

# evtl quatsch
transforms = transforms.ToTensor()

# 42 => disgustsurprise3_fill for sequences[1]
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)


    trainset, testset = torch.utils.data.random_split(dataset, [225, 30])

    # test_before_loader, _ = dataset[0]
    # print("test_before_loader.type():", test_before_loader.type())

    # trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)
    # testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)
    testloader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True)


    class LSTMGenerator(nn.Module):
        def __init__(self, n_features: int, n_hidden: int, n_layers: int, dropout: float):
            super(LSTMGenerator, self).__init__()
            self.n_features = n_features
            self.hidden_size = n_hidden
            self.number_layers = n_layers
            self.dropout = dropout

            self.rnn = nn.LSTM(n_features, n_hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
            # to get back to feature_size (15 for now)
            self.linear = nn.Linear(n_hidden, n_features)


        def forward(self, x):
            # TODO: Was genau ist mein x?
            #  Want to:
            #  Frame(s) eingeben -> Start und Ende, ganze Sequenz ausgeben
            # x ist: ???
            # x war mal: batch_size, seq_len, n_features
            # print("forward before decoding (LSTMGenerator) x.size()", x.size())

            # outputs für die ganze Sequenz ODER ansatz mit Concatination von Tensoren
            outputs = torch.zeros(länge_der_sequenz, batch_size, self.n_features).to(device)

            x_first = x[:, 0, :]              # first anim frame
            x_last = x[:, -1, :]              # last anim frame


            outputs[0] = x              # first frame outputs = first anim frame as this will be known
            x = x.unsqueeze(1)          # for right size
            # print("LSTMGenerator x after unsqueeze x.size()", x.size())

            # x, state = self.decoder(x, lengths, state)
            for t in range(1, lengths[0]):
                x, hidden, cell = self.decoder(x, hidden, cell)

                # print("Decoder output x (which is the second frame):", x.size())
                x.transpose_(0, 1)
                # print("Decoder output x (which is the second frame):", x.size())
                # exit()
                outputs[t] = x[:, 0, :]
                # print("outputs (which is the reconstructed frame for each batch):", outputs.size())
                # print(outputs[0])
                # exit()

            outputs.transpose_(0, 1)
            # x = self.linear(x)
            return outputs


        def init_hidden(self):
            h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            return h_0, c_0




    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model = None
    load_network(model)
    model = model.to(device)

    loss_function = nn.MSELoss()
    # loss_function = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_model(trainloader: DataLoader, testloader: DataLoader, n_Epochs: int):
        history = dict(train=[], test=[])
        print("Training...")

        for epoch in range(1, n_Epochs + 1):
            model.train()
            train_losses = []

            for index, data in enumerate(trainloader):
                optimizer.zero_grad()
                sequences, lengths, _ = data
                sequences = sequences.to(device)
                # lengths = lengths.to(device)

                seq_prediction = model(sequences, lengths, train_batch_size)

                loss = loss_function(seq_prediction, sequences)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            test_losses = []
            model.eval()
            with torch.no_grad():
                for index, data in enumerate(testloader):
                    sequences, lengths, _ = data
                    sequences = sequences.to(device)
                    # lengths = lengths.to(device)

                    seq_prediction = model(sequences, lengths, test_batch_size)

                    loss = loss_function(seq_prediction, sequences)
                    test_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)
            history["train"].append(train_loss)
            history["test"].append(test_loss)

            print(f"Epoch {epoch}: train loss {train_loss} and test loss {test_loss}")

        return model.eval(), history

    # trained_model, history = train_model(trainloader, testloader, num_epochs)
    # safe history dictionary:
    # with open("training_history\history.txt", "w") as file:
    #     file.write(json.dumps(history))
    # or
    # with open('training_history\history.txt', 'w') as f:
    #     print(history, file=f)

    # save_network(model)





