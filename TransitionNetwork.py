import math
import numpy as np
import pandas as pd

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
num_epochs = 2
train_batch_size = 2
test_batch_size = 2
learning_rate = 5e-5

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

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)
    testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=True)


    class Encoder(nn.Module):
        def __init__(self, n_features: int, hidden_dim: int, num_layers: int):
            super(Encoder, self).__init__()
            self.n_features = n_features
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # batch, seq_len, n_features
            self.encoder = nn.LSTM(n_features, hidden_dim, num_layers=num_layers, batch_first=True)


        def forward(self, x, lengths):
            # TODO: Was soll mir der Encoder zurück geben?
            #  Wie verwende ich den Output des Encoders?
            #  Nur Hidden und Cell oder auch x -> was ist x? Die ganze Sequenz?

            # TODO: only use last hidden and try things out with repeat!?


            # x is ??
            # hidden_n is last hidden state
            # print(x.size())                   batch_size, seq_length, n_features


            x_pack = PACK(x, lengths, batch_first=True)
            x, (last_hidden, last_cell) = self.encoder(x_pack)
            x, _ = PAD(x, batch_first=True)

            # print(x.size())
            # print(last_hidden.size())
            # print(last_cell.size())

            # print("forward (Encoder) - last_hidden.size()", last_hidden.size())            # 1, batch_size, hidden_size

            # last_hidden = last_hidden.repeat(x.size(0), 1, 1)
            # last_hidden = last_hidden.reshape((train_batch_size, self.hidden_dim))
            # print("forward (Encoder) - last_hidden.size() after reshape()", last_hidden.size())            # batch_size, hidden_size

            return last_hidden, last_cell


    class Decoder(nn.Module):
        def __init__(self, n_features: int, hidden_dim: int, num_layers: int):
            super(Decoder, self).__init__()
            self.n_features = n_features
            self.latent_dim = hidden_dim
            self.num_layers = num_layers

            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers)
            self.linear = nn.Linear(hidden_dim, n_features)
            self.linear2 = nn.Linear(n_features, hidden_dim)


        def forward(self, x, hidden, cell):
            # TODO: Was soll mir der Decoder zurück geben?
            #  Wie verwende ich die Inputs des Encoders?
            #  Arbeitet der Decoder Frame by Frame oder wie genau soll alles funktionieren?
            # hidden, cell = state
            # print(x.size())
            # print(hidden.size())
            # print(cell.size())

            # print("forward (Decoder) - x.size()", x.size())

            # x needs to get into shape: batch_size, seq_len, hidden_dim
            # x = x.repeat(lengths[0], 1, 1)
            # print("forward (Decoder) - x.size() after repeat", x.size())

            # x = x.reshape((-1, lengths[0], self.latent_dim))
            # print("forward (Decoder) - x.size() after reshape()", x.size())


            # print("Decoder forward x.size", x.size())
            # print("Decoder forward hidden.size", hidden.size())
            # print("Decoder forward cell.size", cell.size())

            x = self.linear2(x)
            x.transpose_(0, 1)
            # print("Decoder forward x.size", x.size())
            # exit()


            prediction, (hidden, cell) = self.decoder(x, (hidden, cell))
            # print("prediction size (not only last):", prediction.size())

            # prediction = prediction[:, -1, :]

            prediction = self.linear(prediction)

            # # x is ??
            # # hidden_n is last hidden state
            # # state consists of hidden_n and cell_n
            # x_pack = PACK(x, lengths, batch_first=True)
            # x, state = self.decoder(x_pack, last_hidden)
            # x, _ = PAD(x, batch_first=True)

            # hidden and last prediction are the same!
            # print(prediction)
            # print(hidden)
            # exit()

            return prediction, hidden, cell


    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, hidden_size: int):
            super(LSTMAutoencoder, self).__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size

            self.encoder = Encoder(n_features=n_features, hidden_dim=hidden_size, num_layers=1)
            self.decoder = Decoder(n_features=n_features, hidden_dim=hidden_size, num_layers=1)
            self.linear = nn.Linear(n_features, hidden_size)
            self.linear2 = nn.Linear(hidden_size, n_features)


        def forward(self, x, lengths, batch_size):
            # TODO: Was genau ist mein x?
            #  Wie verwende ich den Encoder?
            #  Wie verwende ich den Decoder?
            #  Want to:
            #  ganze Sequenz eingeben, ganze Sequenz ausgeben
            #  später: Ersten + letzten Frame eingeben, ganze Sequenz ausgeben
            # x is: batch_size, seq_len, n_features

            # print(self.encoder)
            # print(self.decoder)
            # exit()

            # x, state = self.encoder(x, lengths)
            # hidden_n, cell_n = state

            # x, last_hidden = self.encoder(x, lengths)
            hidden, cell = self.encoder(x, lengths)

            outputs = torch.zeros(lengths[0], batch_size, self.n_features).to(device)
            # print("Decoder outputs-tensor:", outputs.size())

            # print("forward before decoding (LSTMAutoencoder) x.size()", x.size())
            # print("forward before decoding (LSTMAutoencoder) hidden_n.size()", last_hidden.size())
            # print("forward before decoding (LSTMAutoencoder) cell_n.size()", cell_n.size())

            # x ist die ganze sequenz jedes batches - batch_size, seq_len, n_features
            # in den decoder den ersten frame von x damit der rest der sequenz generiert wird?
            # oder den offset? oder target? oder ne kombi aus allem?

            # x is target_frame = last frame?

            # print("LSTMAutoencoder forward encoding x.size()", x.size())

            # last frame as target? or first frame? or second frame?
            x = x[:, 0, :]              # x is the first anim frame
            # print(x)
            outputs[0] = x              # first frame outputs = first anim frame as this will be known
            x = x.unsqueeze(1)
            # exit()

            # last frame representation
            # x = self.linear(x)
            # print("LSTMAutoencoder forward encoding after linear layer x.size()", x.size())
            # exit()


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

        # needs a sequence first to encode hidden and cell state!!
        # def generate(self, x, length, batch_size=1):
        #     outputs = torch.zeros(length, batch_size, self.n_features).to(device)
        #     # print("outputs.size - generate", outputs.size())
        #     # print("x.size - generate", x.size())
        #     x = x[:, 0, :]
        #     outputs[0] = x
        #     x = x.unsqueeze(1)
        #
        #     # this is the problematic part which would need to be solved first!
        #     hidden = torch.zeros(1, batch_size, 256).to(device)
        #     cell = torch.zeros(1, batch_size, 256).to(device)
        #
        #     for t in range(1, length):
        #         x, hidden, cell = self.decoder(x, hidden, cell)
        #         x.transpose_(0, 1)
        #         outputs[t] = x[:, 0, :]
        #
        #     outputs.transpose_(0, 1)
        #     return outputs


        def init_hidden(self):
            pass


    # Transition Network

    # TODO: Nachdenken!
    #  Tutorials die vll ein wenig helfen:
    #  -https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/
    #  -https://www.youtube.com/watch?v=EoGUlvhRYpk&t=28s&ab_channel=AladdinPersson


    model = LSTMAutoencoder(n_features=15, hidden_size=256)
    load_network(model)
    model = model.to(device)

    loss_function = nn.MSELoss(reduction="sum")
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

                # seq_prediction, state = model(sequences, lengths)
                # hidden, cell = state
                # print(lengths)
                # print(sequences[0])
                # print(sequences[1])
                # exit()

                seq_prediction = model(sequences, lengths, train_batch_size)

                # print(seq_prediction.size())
                # print(sequences.size())
                #
                # print(seq_prediction[:, 1, :])
                # print(sequences[:, 1, :])
                #
                # exit()

                # seq_prediction = seq_prediction.reshape(15, -1)
                # sequences = sequences.reshape(15, -1)

                # print(seq_prediction.size())
                # print(sequences.size())

                # print(hidden.size())
                # print(cell.size())

                # exit()

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

    for index, data in enumerate(trainloader):
        optimizer.zero_grad()
        sequences, lengths, name = data
        print("grab 1 sequence - sequences[0]:", sequences[1].size())
        print("first frame - sequences[0][0]:", sequences[1][0])
        print("last frame - sequences[0][-1]:", sequences[1][-1])
        print("name of sequence", name[1])
        sequences = sequences.to(device)
        a_sequence = sequences[1]
        a_sequence = a_sequence.unsqueeze(0)
        print("a_sequence.size:", a_sequence.size())
        a_sequence_length = lengths[1]
        a_sequence_length = a_sequence_length.unsqueeze(0)
        print("a_sequence_length:", a_sequence_length)
        the_name = name[1]
        break

    print(a_sequence.size())

    # needs a sequence first to encode hidden and cell state!
    def generate_test(model: nn.Module, sequence, length):
        model.eval()
        with torch.no_grad():
            seq_prediction = model(sequence, length, 1)
            return seq_prediction


    prediction = generate_test(model, a_sequence, a_sequence_length)

    prediction = prediction.cpu()
    prediction = prediction.squeeze(0)
    # print(prediction.size())

    # get right format for columns
    df = pd.read_csv(csv_read_path + "/neutralhappy1_fill.csv")
    header = list(df.drop(["Frame"], axis=1))
    # df.close()
    del df

    # generate new name for the generated animation
    new_name = "Test_gen_" + the_name

    # transform predictions to csv
    prediction_np = prediction.numpy()
    prediction_df = pd.DataFrame(prediction_np)
    prediction_df.columns = header
    # prediction_df.columns = ["AU1L","AU1R","AU2L","AU2R","AU4L","AU4R","AU6L","AU6R","AU9","AU10","AU13L","AU13R","AU18","AU22","AU27"]
    prediction_df.to_csv(gen_save_pth + new_name + ".csv")
    del prediction_np
    del prediction_df

    # save_network(model)





    # for index, data in enumerate(trainloader):
    #     batch_features, lengths, names = data

        # print(f"before model - batch_features.size()", batch_features.size())         # batch_size, seq_len, n_features
        # print(f"before model - batch_features[0].size()", batch_features[0].size())   # seq_len, n_features
        # print(f"before model - lengths:", lengths)                                    # size(): batch_size
        # print(f"before model - names: {names}")

        # for batch_idx in batch_features:
        #     print("batch_idx", batch_idx.size())
        # batch is: [batch_size, sequence_length, feature_size]
        # calculate current frame, offset and target

        # print(f"before model - batch_features[0]", batch_features[0])
        # print(f"before model - batch_features[1]", batch_features[1])
        # print(f"before model - batch_features[2]", batch_features[2])

        # print(f"+++++++++++++++++++++++++++++++++++ index (consists of a batch of batch_size): {index} +++++++++++++++++++++++++++++++++++")

        # targets for all batches (hyperparams)
        # target = torch.empty(train_batch_size, batch_features.size(2))
        # for i in range(train_batch_size):
        #     target[i] = batch_features[i][lengths[i]-1][:]

        # init hidden per batch
        # to do

        # print(f"before model - target.size(): {target.size()}")
        # print(f"before model - target: {target}")





