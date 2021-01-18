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

        path = "./models/ExGen_" + str(version) + "_net.pth"
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
            path = "./models/normal_dataset/ExGen_" + str(version) + "_net.pth"
            net.load_state_dict(torch.load(path))

        except FileNotFoundError:
            print("There is no such Version yet!")
            load_network(net)
    else:
        pass


# ONLY WORKS FOR BATCH = 1 NOW -> no padding so different sequence lengths
train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

# evtl quatsch
transforms = transforms.ToTensor()

# 42 => ??
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)
    trainset, valset, testset = torch.utils.data.random_split(dataset, [60, 13, 12])

    # trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)
    # testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)

    train_loader = DataLoader(dataset=trainset, batch_size=train_batch_size, shuffle=True, num_workers=0,
                              drop_last=True)
    val_loader = DataLoader(dataset=valset, batch_size=val_batch_size, shuffle=True, num_workers=0, drop_last=True)
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
            self.encoder1 = nn.Linear(n_features * 2, n_output_encoder)
            self.encoder2 = nn.Linear(n_output_encoder, n_output_encoder)
            # lstm
            self.rnn = nn.LSTM(n_output_encoder, n_hidden, num_layers=n_layers, batch_first=True)
            # decoder -> 3 fc-layers
            self.decoder1 = nn.Linear(n_hidden, n_output_encoder)
            self.decoder2 = nn.Linear(n_output_encoder, n_output_encoder / 4)
            self.decoder3 = nn.Linear(n_output_encoder / 4, n_features)
            # batch norm
            # self.batch_norm_encoder = nn.BatchNorm1d(n_output_encoder)
            # self.batch_norm_lstm = nn.BatchNorm1d(n_hidden)

        def forward(self, x):
            """
            :param x: ist target besteht aus current_frame and end_frame
            :return: prediction in Form eines einzelnen Frames
            """
            # some noise??
            # dropout??

            # encoding
            encoded = self.encoder1(x)
            # encoded = self.batch_norm_encoder(encoded)
            encoded = F.leaky_relu(encoded)
            encoded = self.encoder2(encoded)

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
    learning_rate = 1e-3
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
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # best current test error (MSE or L1?):
    best_error = 100                    # none so far
    last_epoch = 0                      # nothing done so far

    # model_safe = 15_256_512 | features, encoded_size, hidden_size

    # training loop
    def train_model(train: DataLoader, val: DataLoader, n_Epochs: int, best_test_error: float):
        writer = SummaryWriter()
        loss_history = []
        best_epoch = 0                  # nothing done so far
        print("Start training...")

        for epoch in range(1 + last_epoch, n_Epochs + 1 + last_epoch):

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

            # eval the model on the val set
            model.eval()
            with torch.no_grad():
                val_loss_mse = 0
                val_loss_l1 = 0
                for index, data in enumerate(val):
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

                    val_loss_mse = val_loss_mse + loss_mse.item()
                    val_loss_l1 = val_loss_l1 + loss_l1.item()

            print(f"Epoch {epoch} of {num_epochs + last_epoch} epochs - Train (MSE): {train_loss:.4f} --- Val: MSE = {val_loss_mse:.4f} | L1 = {val_loss_l1:.4f}")

            # save data to tensorboard and txt!
            loss_history.append(train_loss)
            writer.add_scalar("MSE_Loss - train", train_loss, epoch)
            writer.add_scalar("MSE_Loss - val", val_loss_mse, epoch)
            writer.add_scalar("L1_Loss - val", val_loss_l1, epoch)


            # if val loss last worse than new val loss safe model - KOMMT NOCH
            # val loss with L1 Loss (am besten auch MSE einfach zum vgl!)
            if val_loss_mse < best_test_error:
                torch.save(model.state_dict(), "./models/normal_dataset/ExGen_0_15_256_512_net.pth")
                best_test_error = val_loss_mse
                best_epoch = epoch
                print("new Model was saved!")

            # adjust lr (by hand now for better results?
            # scheduler.step()

        # append to txt .. better save than sorry!
        with open(r'training_history\history_normal_1901_15_256_512.txt', 'a') as f:
            print(loss_history, file=f)

        writer.close()
        print("best test error now (for copy-paste):", best_test_error)
        print("epoch of best test error:", best_epoch)
        print("Finished training!")
        torch.save(model.state_dict(), "./models/normal_dataset/ExGen_1_15_256_512_net.pth")

    train_model(train_loader, val_loader, num_epochs, best_error)

    def test_model(test: DataLoader):
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

        print(f"Test_losses: MSE = {test_loss_mse:.4f} | L1 = {test_loss_l1:.4f}")


    ####### GENERATION #######

    # frown to happy (not in data!)
    start = torch.Tensor([0.06131799283585899,0.08172666935186508,0.03395699517793453,0.04818290386941117,1.1412230401159085,1.1035029605516014,0.113120523744227,1.0597114447811183e-07,0.30072830873532025,3.517824015742137e-05,8.749662808520226e-09,0.0013996257573782305,0.6260813195906696,0.09551505350147324,6.689342889722282e-10])
    end = torch.Tensor([1.1999996314449428,1.1999995870728282,1.0278275208869838,1.0194900942312504,0.2341106233151145,0.0653982846386197,0.8537125658024893,0.8081682829103988,1.4364089873377762e-08,-2.1860498663560296e-08,9.06581770646935e-07,0.006658209848300654,-4.7158831256122355e-09,6.94503947633974e-07,0.9977853517181079])
    # print(start.size())
    # exit()

    def generate_expression(start_frame: torch.Tensor, end_frame: torch.Tensor, sequence_length: int, name: str):
        # eval the model on the test set
        model.eval()
        with torch.no_grad():
            first_frame = start_frame.to(device)
            last_frame = end_frame.to(device)
            number_aus = first_frame.size(0)

            target = torch.cat([first_frame, last_frame])  # target size: [30]
            target = target.unsqueeze(0)  # [1, 30]
            target = target.unsqueeze(0)  # [1, 1, 30]
            target = target.to(device)

            model.zero_hidden()

            created_sequence = torch.zeros(1, sequence_length, number_aus).to(device)
            created_sequence[0][0] = first_frame

            for t in range(1, sequence_length):
                prediction = model(target)
                prediction_aus = prediction.view(15)

                created_sequence[0][t] = prediction

                target = torch.cat([prediction_aus, last_frame])
                target = target.unsqueeze(0)
                target = target.unsqueeze(0)

            # for convenience:
            sequence = created_sequence

            sequence = sequence.cpu()
            sequence = sequence.squeeze(0)
            # print(prediction.size())

            # get right format for columns
            df = pd.read_csv(csv_read_path + "/neutralhappy1_fill.csv")
            header = list(df.drop(["Frame"], axis=1))
            # df.close()
            del df

            # generate new name for the generated animation
            new_name = "Gen_" + name

            # transform predictions to csv
            sequence_np = sequence.numpy()
            sequence_df = pd.DataFrame(sequence_np)
            sequence_df.columns = header
            # prediction_df.columns = ["AU1L","AU1R","AU2L","AU2R","AU4L","AU4R","AU6L","AU6R","AU9","AU10","AU13L","AU13R","AU18","AU22","AU27"]
            sequence_df.to_csv(gen_save_pth + new_name + ".csv")
            del sequence_np
            del sequence_df

    # generate_expression(start, end, 180, "frown_to_happy")


    # custom safe method which can be used to store individual models (name as input during method)
    # save_network(model)
