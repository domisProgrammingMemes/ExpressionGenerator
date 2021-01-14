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

from torch.utils.tensorboard import SummaryWriter

csv_read_path = r"Data\FaceTracker\preprocessed\csv"
gen_save_pth = r"Data\GeneratedAnims\\"

# Path to save and load models
# net_path = "./models/Transition_net.pth"


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
            path = "./models/ExpressionGenerator_" + str(version) + "_net.pth"
            net.load_state_dict(torch.load(path))

        except FileNotFoundError:
            print("There is no such Version yet!")
            load_network(net)
    else:
        pass


# ONLY WORKS FOR BATCH = 1 NOW -> no padding so different sequence lengths
train_batch_size = 1
test_batch_size = 1

# evtl quatsch
transforms = transforms.ToTensor()

# 42 => ??
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)
    trainset, testset = torch.utils.data.random_split(dataset, [225, 30])

    # test_before_loader, _ = dataset[0]
    # print("test_before_loader.type():", test_before_loader.type())

    # trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)
    # testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)

    train_loader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    # val_loader = DataLoader(dataset=validationset, batch_size=train_batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=testset, batch_size=test_batch_size, shuffle=True, num_workers=0, drop_last=True)


    class ExpressionGenerator(nn.Module):
        def __init__(self, n_features: int, n_output_encoder: int, n_hidden: int, n_layers: int, p: float):
            super(ExpressionGenerator, self).__init__()
            self.n_features = n_features
            self.hidden_size = n_hidden
            self.n_layers = n_layers
            self.dropout = p
            self.output_encoder_size = n_output_encoder

            # hidden and cell for LSTM
            self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            self.cell = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

            # modules
            # linear for encoding of current_frame and target
            self.encoder = nn.Linear(n_features * 2, n_output_encoder)
            # lstm
            self.rnn = nn.LSTM(n_output_encoder, n_hidden, num_layers=n_layers, batch_first=True)
            # decoder -> 3 fc-layers
            self.decoder1 = nn.Linear(n_hidden, n_output_encoder)
            self.decoder2 = nn.Linear(n_output_encoder, n_features * 4)
            self.decoder3 = nn.Linear(n_features * 4, n_features)
            # batch norm
            # self.batch_norm_encoder = nn.BatchNorm1d(n_output_encoder)
            # self.batch_norm_lstm = nn.BatchNorm1d(n_hidden)

        def forward(self, x):
            """
            :param x: ist target besteht aus current_frame and end_frame
            :return: prediction in form eines einzelnen Frames
            """
            # encoding
            encoded = self.encoder(x)
            # encoded = self.batch_norm_encoder(encoded)
            encoded = F.leaky_relu(encoded)

            # temporal dynamics
            frame_encoding, (self.hidden, self.cell) = self.rnn(encoded, (self.hidden, self.cell))
            # frame_encoding = self.batch_norm_lstm(frame_encoding)

            # decoding with 3 fc
            prediction = F.leaky_relu(self.decoder1(frame_encoding))
            prediction = F.leaky_relu(self.decoder2(prediction))
            prediction = self.decoder3(prediction)
            return prediction

        def zero_hidden(self):
            self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            self.cell = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)


    # Hyperparameters
    num_epochs = 100
    learning_rate = 1e-4
    dropout = 0.5  # not used right now
    teacher_forcing_ratio = 0.5

    # model
    model = ExpressionGenerator(15, 256, 512, 1, dropout)
    load_network(model)
    model = model.to(device)

    # define loss(es) and optimizer
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    def train_model(train: DataLoader, test: DataLoader, n_Epochs: int):
        writer = SummaryWriter()
        loss_history = []
        print("Start training...")

        for epoch in range(1, n_Epochs + 1):

            model.train()
            train_loss = 0
            for index, data in enumerate(train):
                batch_data, name = data
                batch_data = batch_data.to(device)
                # print(batch_data.size())      # size = [batch_size, sequence_length, n_features]
                # print(name)                   # csv name
                seq_length = batch_data.size(1)
                number_aus = batch_data.size(2)

                first_frame = batch_data[0, 0]
                last_frame = batch_data[0, -1]

                target = torch.cat([first_frame, last_frame])  # target size: [30]

                target = target.unsqueeze(0)  # [1, 30]
                target = target.unsqueeze(0)  # [1, 1, 30]
                # print("target. size() [seq, batch, input_size", target.size())
                target = target.to(device)

                # initial hidden and cell at t=0
                # hidden, cell = model.lstm.init_hidden()
                # print("hidden size() [num_layer*num_directions, batch, input_size", hidden.size())
                # print("cell size() [num_layer*num_directions, batch, input_size", cell.size())
                model.zero_hidden()

                # create empty sequence tensor for whole anim:
                created_sequence = torch.zeros(1, seq_length, number_aus).to(device)
                # print("created_seq.size() [batch, seq_len, feature_size", created_sequence.size())
                created_sequence[0][0] = first_frame

                # für jede sequenz
                optimizer.zero_grad()

                for t in range(1, seq_length):
                    # will ich hidden und cell überhaupt behalten?!
                    prediction = model(target)
                    prediction_aus = prediction.view(15)
                    # print("prediction.size()", prediction_aus.size())               # [1, 1, 15]

                    # prediction has the size: batch, hidden_size
                    # real_next_frame = batch_data[0][t]
                    # print("real next frame.size()", real_next_frame.size())
                    # loss_per_frame = l1_loss(real und predicted)
                    # sequence_loss += loss_per_frame


                    created_sequence[0][t] = prediction
                    # print("created sequence", created_sequence)

                    # teacher forcing
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

                    # define new target
                    target = torch.cat([prediction_aus, last_frame])
                    target = target.unsqueeze(0)
                    target = target.unsqueeze(0)
                    # print("target size: ", target.size())
                    # print(target[0][0])
                    # exit()

                loss = mse_loss(created_sequence, batch_data)
                loss.backward()
                optimizer.step()

                train_loss = train_loss + loss.item()

            # eval the model on the test set
            model.eval()
            with torch.no_grad():
                test_loss_mse = 0
                test_loss_l1 = 0
                for index, data in enumerate(test):
                    batch_data, name = data
                    batch_data = batch_data.to(device)
                    seq_length = batch_data.size(1)
                    number_aus = batch_data.size(2)
                    first_frame = batch_data[0, 0]
                    last_frame = batch_data[0, -1]

                    target = torch.cat([first_frame, last_frame])  # target size: [30]
                    target = target.unsqueeze(0)  # [1, 30]
                    target = target.unsqueeze(0)  # [1, 1, 30]
                    target = target.to(device)

                    model.zero_hidden()

                    created_sequence = torch.zeros(1, seq_length, number_aus).to(device)
                    created_sequence[0][0] = first_frame

                    for t in range(1, seq_length):
                        prediction = model(target)
                        prediction_aus = prediction.view(15)

                        created_sequence[0][t] = prediction

                        target = torch.cat([prediction_aus, last_frame])
                        target = target.unsqueeze(0)
                        target = target.unsqueeze(0)

                    loss_mse = mse_loss(created_sequence, batch_data)
                    loss_l1 = l1_loss(created_sequence, batch_data)

                    test_loss_mse = test_loss_mse + loss_mse.item()
                    test_loss_l1 = test_loss_l1 + loss_l1.item()

            print(f"Epoch {epoch} of {num_epochs} epochs - Train: {train_loss} -- Test: MSE = {test_loss_mse} | L1 = {test_loss_l1}")

            loss_history.append(train_loss)
            writer.add_scalar("MSE_Loss - train", train_loss, epoch)
            writer.add_scalar("MSE_Loss - test", test_loss_mse, epoch)
            writer.add_scalar("L1_Loss - test", test_loss_l1, epoch)


            # if val loss last worse than new val loss safe model - KOMMT NOCH
            # val loss with L1 Loss (am besten auch MSE einfach zum vgl!)
            torch.save(model.state_dict(), "./models/ExpressionGenerator_0_net.pth")

        # append to txt .. better save than sorry!
        with open('training_history\history.txt', 'a') as f:
            print(loss_history, file=f)

        writer.close()


    train_model(train_loader, test_loader, num_epochs)

    # trained_model, history = train_model(trainloader, testloader, num_epochs)
    # safe history dictionary:
    # with open("training_history\history.txt", "w") as file:
    #     file.write(json.dumps(history))
    # or
    # with open('training_history\history.txt', 'a') as f:
    #     print(losses, file=f)

    # save_network(model)
