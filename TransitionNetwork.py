import math
import numpy as np
import pandas as pd
import json
import random

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
# ONLY WORKS FOR BATCH = 1 NOW -> no padding so different sequence lengths
train_batch_size = 1
test_batch_size = 1



# evtl quatsch
transforms = transforms.ToTensor()

# 42 => neutralhappy3_fill
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)


    trainset, validationset, testset = torch.utils.data.random_split(dataset, [200, 25, 30])

    # test_before_loader, _ = dataset[0]
    # print("test_before_loader.type():", test_before_loader.type())

    # trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)
    # testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)

    train_loader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=validationset, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True)


    class LSTM(nn.Module):
        def __init__(self, n_features: int, n_hidden: int, n_layers: int, p: float):
            super(LSTM, self).__init__()
            self.n_features = n_features
            self.hidden_size = n_hidden
            self.n_layers = n_layers
            self.dropout = p

            # * 2 wegen concat von current - target
            self.rnn = nn.LSTM(n_features*2, n_hidden, num_layers=n_layers, batch_first=True)
            # to get back to feature_size (15 for now)

            self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            self.cell = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

        def forward(self, x):
            # x ist: concat vector aus current_frame und target_frame - size: [30]
            # print("forward before decoding (LSTM) x.size()", x.size())

            output, (self.hidden, self.cell) = self.rnn(x, (self.hidden, self.cell))
            # h_t has shape batch_size, hidden_size
            # print("lstm module output.size", output.size())               # ([1, 1, 512])
            # print("lstm module hidden.size", self.hidden.size())          # ([1, 1, 512])

            return output

        def init_hidden(self):
            h_0 = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            c_0 = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            return h_0, c_0



    class FeatureDecoder(nn.Module):
        def __init__(self, input_size: int, output_size: int):
            super(FeatureDecoder, self).__init__()
            self.input_size = input_size
            self.output_size = output_size

            self.fc1 = nn.Linear(input_size, output_size*2)
            self.fc2 = nn.Linear(output_size*2, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x



    class FrameGenerator(nn.Module):
        def __init__(self, recurrent: nn.Module, fullyconnected: nn.Module):
            super(FrameGenerator, self).__init__()
            self.lstm = recurrent
            self.fully = fullyconnected


        def forward(self, x):
            """
            :param x: ist ein einziger Frame für's Erste
            :return: den vorhergesagten Frame
            """


            output = self.lstm(x)

            prediction = self.fully(output)
            return prediction









    # Hyperparameters
    num_epochs = 20
    learning_rate = 1e-3
    dropout = 0.5                               # not used right now
    teacher_forcing_ratio = 0.5

    # several models to build 1
    mylstm = LSTM(15, 512, 1, 0.5)
    mydecoder = FeatureDecoder(512, 15)

    model = FrameGenerator(mylstm, mydecoder)
    # load_network(model)
    model = model.to(device)


    # define loss and optimizer
    loss_function = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    def train_model(train: DataLoader, val: DataLoader, n_Epochs: int):
        history = dict(train=[], test=[])
        print("Training...")

        for epoch in range(1, n_Epochs + 1):
            model.train()
            loss = 0

            # generative "loop": - index is number of sequences in this case per loader
            for index, data in enumerate(train):
                batch_data, name = data
                batch_data = batch_data.to(device)
                # print(batch_data.size())      # size = [batch_size, sequence_length, n_features]
                # print(name)                   # csv name
                seq_length = batch_data.size(1)
                number_aus = batch_data.size(2)

                # device?
                first_frame = batch_data[0, 0]
                last_frame = batch_data[0, -1]

                target = torch.cat([first_frame, last_frame])  # target size: [30]

                target = target.unsqueeze(0)        # [1, 30]
                target = target.unsqueeze(0)        # [1, 1, 30]
                # print("target. size() [seq, batch, input_size", target.size())
                target = target.to(device)

                # initial hidden and cell at t=0 -> created in LSTM
                # hidden, cell = model.lstm.init_hidden()
                # print("hidden size() [num_layer*num_directions, batch, input_size", hidden.size())
                # print("cell size() [num_layer*num_directions, batch, input_size", cell.size())

                # create empty sequence tensor for whole anim:
                created_sequence = torch.zeros(1, seq_length, number_aus).to(device)
                # print("created_seq.size() [batch, seq_len, feature_size", created_sequence.size())

                batch_loss = 0

                for t in range(1, seq_length):
                    optimizer.zero_grad()

                    # will ich hidden und cell überhaupt behalten?!
                    prediction = model(target)
                    # print("prediction.size()", prediction.size())               # [1, 1, 15]


                    # prediction has the size: batch, hidden_size
                    real_next_frame = batch_data[0][t]

                    # loss sollte distanz zwischen vorhersage und eigentlichem frame sein!
                    single_loss = loss_function(prediction, real_next_frame)

                    single_loss.backward(retain_graph=True)
                    optimizer.step()

                    batch_loss = batch_loss + single_loss.item()

                    # created_sequence[0][t] = prediction

                    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                    # if use_teacher_forcing:
                    #     # if teacher forced give it the correct next frame as part of target:
                    #     target = torch.cat([real_next_frame, last_frame]).to(device)
                    #     # print("with teacher force", target.size())
                    #     # exit()
                    # else:
                    #     # neues ziel da neuer frame:
                    #     target = torch.cat([prediction, last_frame]).to(device)
                    #     # print("without teacher force", target.size())
                    #     # exit()

                    target = torch.cat([prediction, last_frame]).to(device)
                    target = target.unsqueeze(0)

                    # print("target size: ", target.size())
                    # print(target)
                    # exit()


                print(f"Epoch {epoch} batch{index} of {num_epochs} epochs - loss for whole sequence: {batch_loss}")



    train_model(train_loader, val_loader, num_epochs)

    # trained_model, history = train_model(trainloader, testloader, num_epochs)
    # safe history dictionary:
    # with open("training_history\history.txt", "w") as file:
    #     file.write(json.dumps(history))
    # or
    # with open('training_history\history.txt', 'w') as f:
    #     print(history, file=f)

    # save_network(model)





