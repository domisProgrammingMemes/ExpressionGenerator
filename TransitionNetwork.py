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


    """
    Custom LSTM Cell to represent the work of the paper
    Equations:
    - W stands for weight matrices/weights
    - b stands for biases
    - @ is for matrix multiplication
    - * is for elementwise multiplication
    
    hFOt is concatination of hFt and hOt
    
    (1) i_t = sigmoid(W1 @ hEt + W2 @ hRt-1 + W3 @ hFOt + b) 
    (2) o_t = sigmoid(W1 @ hEt + W2 @ hRt-1 + W3 @ hFOt + b)
    (3) f_t = sigmoid(W1 @ hEt + W2 @ hRt-1 + W3 @ hFOt + b)
    (4) c^_t (g_t) = tanh(W1 @ hEt + W1 @ hRt-1 + W3 @ hFOt + b)
    (5) c_t = f_t * c_t-1 + i_t * tanh(c^_t)
    (6) R(hEt, hRt-1, c_t, hOt, hFt) = o_t+1 * c_t
    (7) hR_t = R(hEt, hRt-1, c_t, hOt, hFt)
        with W1 (W) = feedforward, W2 (U) = recurrent and W3 (C) conditioning weights
    """

    """
    Dimensions:
    hEt = 512
    HFt & hOt = 128 -> concat 256
    hRt => auch 512 
    """

    class Encoder(nn.Module):
        def __init__(self, input_size: int, hidden_size: int):
            super(Encoder, self).__init__()
            # input to hidden 1
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            """
            :param x: input into the encoder which are:
                    (1) current frame AUs
                    (2) target frame AUs
                    (3) offset frame AUs (difference (1) to (2)
            :return: returns input for RecurrentGenerator
            in case of (2) and (3) the tensors need to be concatenated

            """
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            return x

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
            self.fc1 = nn.Linear(input_size, hidden1_size)
            self.fc2 = nn.Linear(hidden1_size, hidden2_size)
            self.fc3 = nn.Linear(hidden2_size, output_size)

        def forward(self, x):
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
            return x


    class RecurrentGenerator(nn.Module):
        def __init__(self, input_size, hidden_size, number_layers=1):
            super(RecurrentGenerator, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = number_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=number_layers, batch_first=True)

        def forward(self, x, h_t, c_t):

            lstm_out, (h_t, c_t) = self.lstm(x, (h_t, c_t))

            return lstm_out, h_t, c_t


        def init_hidden(self):
            h0 = torch.zeros(self.num_layers, train_batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, train_batch_size, self.hidden_size)
            return h0, c0



    class TransitionNetwork(nn.Module):
        def __init__(self, feature_size: int):
            super(TransitionNetwork, self).__init__()
            self.feature_size = feature_size
            self.frame_encoder = Encoder(feature_size, 256)
            self.target_encoder = Encoder(feature_size, 128)
            self.offset_encoder = Encoder(feature_size, 128)

            self.feature_encoder = Encoder(512, 256)

            self.lstm = RecurrentGenerator(256, 256)
            self.decoder = FrameDecoder(256, 128, 64, 15)

            self.count = 0


        def forward(self, xt, t, hidden_state, cell_state):
            """
            :param xt: current frame
            :param t: target
            :param hidden_state: (initial) hidden states
            :param cell_state: (initial) cell states
            :return: next frame with all info necessary
            """

            # print(f"xt[0]: {xt.size()}")
            # print(f"t[0]: {t.size()}")

            # calculate offset:
            ot = torch.subtract(xt, t)

            # print(f"forward (Transition Network) ot: {ot}")

            h_t = self.frame_encoder(xt)
            h_o = self.offset_encoder(ot)
            t = self.target_encoder(t)


            frame_features = torch.cat((h_t, h_o, t), 1)
            encoded_frame = self.feature_encoder(frame_features)

            # print(f"forward (Transition Network) - encoded_frame.size: {encoded_frame.size()}")
            encoded_frame = torch.unsqueeze(encoded_frame, 1)


            # lstm out should be of length sequence_length
            lstm_out, hidden_state, cell_state = self.lstm(encoded_frame, hidden_state, cell_state)

            lstm_out = torch.squeeze(lstm_out, 1)

            # print(f"forward (Transition Network) - lstm_out.size: {lstm_out.size()}")
            # print(f"forward (Transition Network) - hidden_state.size: {hidden_state.size()}")
            # print(f"forward (Transition Network) - cell_state.size: {cell_state.size()}")

            next_frame = self.decoder(lstm_out)

            return next_frame, hidden_state, cell_state


            # self.count += 1
            # return self.count




    # Testing stuff:

    # Transition Network
    MyModel = TransitionNetwork(15)

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

        # TODO: Tell each batch how long the sequence is so it can ignore everything after the last frame somehow?

        print(f"++++++++++++++++++++++++++++++ index (consists of a batch of batch_size): {index} +++++++++++++++++++++++++++++++++")

        # targets for all batches (hyperparams)
        target = torch.empty(train_batch_size, batch_features.size(2))
        for i in range(train_batch_size):
            target[i] = batch_features[i][lengths[i]-1][:]

        # init hidden per batch
        hidden, cell = MyModel.lstm.init_hidden()

        print(f"before model - target.size(): {target.size()}")
        print(f"before model - target: {target}")
        print()
        print("PER FRAME CALCULATIONS")
        print()

        for frame in range(lengths[0]):
            # try:
            #     print(f"====================> frame: {count}")
            # except:
            #     print(f"====================> frame: 0")
            # count = MyModel.forward(batch_features[:, frame, :], target)
            # print(f"----------------------- new frame ----------------------")
            # print(batch_features[:, frame, :])
            next_frame, hidden, cell = MyModel.forward(batch_features[:, frame, :], target, hidden, cell)

            # target
            if frame < (lengths[0] - 1):
                real_next_frame = batch_features[:, frame+1, :]
            else:
                real_next_frame = target
            print(real_next_frame)



            # print(hidden)
            # print(cell)
            # print(next_frame.size())


        # just on batch for now!!!
        break






























