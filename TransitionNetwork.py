import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# if I end up using normalization:
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

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
train_batch_size = 5
test_batch_size = 5
learning_rate = 1e-3

# input_size = ?
# sequence_length = ?

# evtl quatsch
transforms = transforms.ToTensor()

torch.manual_seed(0)

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)
    trainset, testset = torch.utils.data.random_split(dataset, [70, 15])

    # test_before_loader, _ = dataset[0]
    # print("test_before_loader.type():", test_before_loader.type())

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=False)
    testloader = DataLoader(dataset=testset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=False)


    class Encoder(nn.Module):
        def __init__(self, input_size: int, hidden_size: int):
            super(Encoder, self).__init__()
            pass

        def forward(self, x):
            pass

    # class TargetEncoder(nn.Module):
    #     def __init__(self, input_size: int, hidden_size: int):
    #         super(TargetEncoder, self).__init__()
    #         self.fc1 = nn.Linear(input_size, hidden_size)
    #         self.fc2 = nn.Linear(hidden_size, hidden_size)
    #
    #
    # class OffsetEncoder(nn.Module):
    #     def __init__(self,  input_size: int, hidden_size: int):
    #         super(OffsetEncoder, self).__init__()
    #         self.fc1 = nn.Linear(input_size, hidden_size)
    #         self.fc2 = nn.Linear(hidden_size, hidden_size)

    class FrameDecoder(nn.Module):
        def __init__(self, input_size: int, hidden1_size: int, hidden2_size: int, output_size: int):
            super(FrameDecoder, self).__init__()
            pass

        def forward(self, x):
            pass


    class RecurrentGenerator(nn.Module):
        def __init__(self, input_size, hidden_size, number_layers=1):
            super(RecurrentGenerator, self).__init__()
            pass

        def forward(self, x, h_t, c_t):
            pass


        def init_hidden(self):
            pass



    class TransitionNetwork(nn.Module):
        def __init__(self, feature_size: int):
            super(TransitionNetwork, self).__init__()



        def forward(self, xt, t, hidden_state, cell_state):
            pass




    # Transition Network
    # TODO: LSTM autoencoder; links für reference are bookmarked + further search if necessary


    for index, data in enumerate(trainloader):
        batch_features, lengths, names = data

        print(f"before model - batch_features.size()", batch_features.size())
        print(f"before model - batch_features[0].size()", batch_features[0].size())
        print(f"before model - lengths:", lengths)
        print(f"before model - names: {names}")
        # for batch_idx in batch_features:
        #     print("batch_idx", batch_idx.size())
        # batch is: [batch_size, sequence_length, feature_size]
        # calculate current frame, offset and target

        print(f"++++++++++++++++++++++++++++++ index (consists of a batch of batch_size): {index} +++++++++++++++++++++++++++++++++")

        # targets for all batches (hyperparams)
        target = torch.empty(train_batch_size, batch_features.size(2))
        for i in range(train_batch_size):
            target[i] = batch_features[i][lengths[i]-1][:]

        # init hidden per batch
        # to do

        print(f"before model - target.size(): {target.size()}")
        print(f"before model - target: {target}")
        print()
        print("PER FRAME CALCULATIONS")
        print()





