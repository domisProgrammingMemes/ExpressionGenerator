import torch
import torch.nn as nn
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
num_epochs = 5
train_batch_size = 2
test_batch_size = 2
learning_rate = 1e-3

# input_size = ?
# sequence_length = ?


if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)
    trainset, testset = torch.utils.data.random_split(dataset, [70, 15])

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0)


    """
    Model consists of several modules
    1. Frame encoder
    at each timestep t the current frame is encoded into a smaller representation ==> hEt  
    2. Future context encoders 
    - target t from encoder F() is the desired expression (transition end) 
        -> constant throughout generation process/sequence ==> hF
    - offset o from encoder O() is composed of the euclidean distances (?) of each AU from target expression 
        -> distance current AUs to target AUs which evolve with respect to last generated frame ==> hOt
    3. Recurrent generator
    single 512 LSTM layer that uses concatenation hF and hO as future-conditioning information along with 
        added corresponding parameters. WHAT HAPPENS WITH hEt? (hF concat hOt)
        ? -> custom LSTM equations ?
        ==> hRt as output
    4. Frame decoder
    each hRt is passed to decoder which outputs offset hDt from current frame xt so that
        final prediction x(t+1) = xt + hDt
    5. Hidden state initializer
    just use zero-initialization for now!
    """

    # just some testing again
    # dataiter = iter(trainloader)
    # sequences, lengths = dataiter.next()
    #
    # print(f"batch size: {train_batch_size}")
    # print()
    # print(f"sequences: {sequences}")
    # print(f"lengths: {lengths}")
    # print()
    # print(f"1st sequence: {sequences[0]}")
    # print(f"1st sequence length: {lengths[0]}")
    #
    # print()
    # print(f"first sequence first frame: {sequences[0][0]}")


    for batch_idx, (sequences, lengths, names) in enumerate(trainloader):
        if batch_idx > 1:
            break
        print(f"name: {names[0]}")
        print(f"length: {lengths[0]}")
        print(f"first sequence: {sequences[0]}")
        print(f"sequence size: {sequences[0].size()}")
        print()
        print(f"name: {names[1]}")
        print(f"length: {lengths[1]}")
        print(f"second sequence: {sequences[1]}")
        print(f"sequence size (which should be as long as the first sequence: {sequences[1].size()}")
        print(f"second sequence last real tensor: {sequences[1][lengths[1]-1]}")

        # prints above seem to print out correct stuff!
        # TODO: TransitionNetwork implementation!






























