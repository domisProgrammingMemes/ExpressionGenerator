import math
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
def save_network(net: nn.Module, path):
    # save the network?
    save = input("Save net? (y) or (n)?")
    if save == "y":
        torch.save(net.state_dict(), path)
    else:
        pass


# load an existing network's parameters and safe them into the just created net
def load_network(net: nn.Module, net_path):
    # save the network?
    load = input("Load Network? (y) or (n)?")
    if load == "y":
        net.load_state_dict(torch.load(net_path))
    else:
        pass


# Hyperparameters
num_epochs = 4
train_batch_size = 15
test_batch_size = 5
learning_rate = 1e-4

# input_size = ?
# sequence_length = ?

# evtl quatsch
transforms = transforms.ToTensor()

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)


    trainset, testset = torch.utils.data.random_split(dataset, [225, 30])

    # test_before_loader, _ = dataset[0]
    # print("test_before_loader.type():", test_before_loader.type())

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=False)
    testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=False)


    class Encoder(nn.Module):
        def __init__(self, n_features: int, latent_dim: int = 4, num_layers: int = 2):
            super(Encoder, self).__init__()
            self.n_features = n_features
            self.hidden_dim = latent_dim
            self.num_layers = num_layers

            # batch, seq_len, n_features
            self.encoder = nn.LSTM(input_size=n_features, hidden_size=latent_dim, num_layers=2, batch_first=True)


        def forward(self, x, lengths):
            # x is ??
            # hidden_n is last hidden state
            x_pack = PACK(x, lengths, batch_first=True)
            x, (hidden_n, cell_n) = self.encoder(x_pack)
            x, _ = PAD(x, batch_first=True)
            return x, (hidden_n, cell_n)


    class Decoder(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, num_layers: int):
            super(Decoder, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_layers = num_layers

            self.decoder = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, batch_first=True)

        def forward(self, x, lengths):
            # x is ??
            # hidden_n is last hidden state
            x_pack = PACK(x, lengths, batch_first=True)
            x, (hidden_n, cell_n) = self.encoder(x_pack)
            x, _ = PAD(x, batch_first=True)
            return x, (hidden_n, cell_n)


    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, hidden_size: int):
            super(LSTMAutoencoder, self).__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size

            self.encoder = Encoder(n_features, latent_dim=hidden_size, num_layers=1).to(device)
            self.decoder = Decoder(hidden_size, n_features, num_layers=1).to(device)


        def forward(self, x, lengths):
            x, (hidden_n, cell_n) = self.encoder(x, lengths)

            print("forward before decoding (LSTMAutoencoder) x.size()", x.size())
            print("forward before decoding (LSTMAutoencoder) hidden_n.size()", hidden_n.size())
            print("forward before decoding (LSTMAutoencoder) cell_n.size()", cell_n.size())

            exit()

            x, (hidden_n, cell_n) = self.decoder(x, lengths)
            return x, hidden_n, cell_n


        def init_hidden(self):
            pass




    # Transition Network
    model = LSTMAutoencoder(15, 7)
    model = model.to(device)

    loss_function = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), learning_rate=learning_rate)

    def train_model(trainloader: DataLoader, testloader: DataLoader, n_Epochs: int):
        history = dict(train=[], test=[])

        for epoch in range(1,n_Epochs + 1):
            model.train()
            train_losses = []

            for index, data in enumerate(trainloader):
                optimizer.zero_grad()
                sequences, lengths, _ = data
                sequences = sequences.to(device)
                lengths = lengths.to(device)

                seq_prediction = model(sequences, lengths)

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
                    lengths = lengths.to(device)

                    seq_prediction = model(sequences, lengths)

                    loss = loss_function(seq_prediction, sequences)
                    test_losses.append(loss.item())

            train_loss = torch.mean(train_losses)
            test_loss = torch.mean(test_losses)
            history["train"].append(train_loss)
            history["test"].append(test_loss)

            print(f"Epoch {epoch}: train loss {train_loss} and test loss {test_loss}")

        return model.eval(), history

    trained_model, history = train_model(trainloader, testloader, 5)
    save_network(trained_model, path)







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





