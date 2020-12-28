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
train_batch_size = 1
test_batch_size = 1
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
        def __init__(self, input_size=15, hidden_size=512):
            super(RecurrentGenerator, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        def forward(self, x):
            out, _ = self.lstm(x)
            # print(f"forward (Recurrent Generator)  - out[0].size: {out[0].size()}")
            return out

    # TODO: this class needs work as several changes need to be made:
    #  1: rearrange the overall network so the LSTM gets (padded) Sequences as input (just 1 Input)
    #  2: reimplement the encoder structure so that (1) is possible
    #  Overall: Use standard LSTM-Module; use FutureContext + Frame as Input for LSTM; Padding maybe in forward-method of Network?
    #  p.s. maybe use LSTM as encoder?
    class TransitionNetwork(nn.Module):
        def __init__(self, feature_size: int):
            super(TransitionNetwork, self).__init__()
            self.feature_size = feature_size

            # parameters
            self.t = None
            self.o = None
            self.one_output = None

            # for parameter tuning
            # self.frame_encoder_size = frame_encoder_size
            # self.target_encoder_size = target_encoder_size_size
            # self.offset_encoder_size = offset_encoder_size


            self.FrameEncoder = Encoder(feature_size, 512)
            self.TargetEncoder = Encoder(feature_size, 128)
            self.OffsetEncoder = Encoder(feature_size, 128)

            # self.RecurrentGenerator = RecurrentGenerator(frame_encoder_size=512, target_encoder_size=128, offset_encoder_size=128, hidden_size=512)
            self.RecurrentGenerator = RecurrentGenerator(feature_size, 512)
            self.RecurrentGenerator2 = RecurrentGenerator(512, feature_size)

            self.FrameDecoder = FrameDecoder(input_size=512, hidden1_size=256, hidden2_size=128, output_size=feature_size)


        def forward(self, batch, lengths):
            print("forward (Transition Network) - batch.size()", batch.size())

            test = batch
            print(f"forward (Transition Network) - test1.size: {test.size()}")
            print(f"forward (Transition Network) - lengths: {lengths}")

            """
            lets get to the future context which consists of
            (1) the target frame -> last in sequence
            (2) difference current frame to target frame -> current - (1)
            """

            # # calculate target t
            self.t = torch.empty(train_batch_size, self.feature_size)
            # print(f"t.size(): {t.size()}")
            for i in range(train_batch_size):
                self.t[i] = batch[i, lengths[i]-1, :]
            print(f"forward (Transition Network) - target t.size(): {self.t.size()}")
            # print(f"forward (Transition Network) - target t: {self.t[:]}")
            # # calculate offset o with eucledian distance
            # o = torch.sub(frame - t)
            # self.o = torch.sub(batch[:, , :], t)
            try:
                self.o = torch.sub(self.one_output, self.t)
            except:
                self.o = self.t

            print(f"forward (Transition Network) - offset o.size(): {self.o.size()}")
            # print(f"forward (Transition Network) - offset o: {self.o}")

            # ENCODING



            # RECURRENT CALCULATIONS

            test_pack = nn.utils.rnn.pack_padded_sequence(test, lengths, batch_first=True)
            outputs = self.RecurrentGenerator(test_pack)

            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            self.one_output = torch.empty(train_batch_size, 512)
            # print(f"one output as tensor of shape batch: {one_output.size()}")

            for i in range(len(lengths)):
                self.one_output[i] = outputs[i, lengths[i] - 1, :]

            # DECODING

            test_pack = nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True)
            outputs = self.RecurrentGenerator2(test_pack)

            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            self.one_output = torch.empty(train_batch_size, self.feature_size)
            # print(f"one output as tensor of shape batch: {one_output.size()}")

            for i in range(len(lengths)):
                self.one_output[i] = outputs[i, lengths[i] - 1, :]

            # print(f"forward (Transition Network) - test output.size(): {outputs.size()}")
            # print(f"forward (Transition Network) - test output: {outputs[1][-1][:]}")
            # print("+++++++++++++++++++++++++++")
            # print(f"forward (Transition Network) - one output.size(): {self.one_output.size()}")
            # print(f"forward (Transition Network) - one output: {self.one_output[1][:]}")

            """
            I think I only need the last LSTM output as I only want 1 frame at a time
            one_output always considers the last frame of the sequence with the code above!
            -> one_output [batch_size, hidden_size]
            """
            print()

            return outputs, self.one_output


    # Testing stuff:

    # Transition Network
    model = TransitionNetwork(15)

    for data in trainloader:
        batch_features, lengths, names = data

        print(f"before model - batch_features.size()", batch_features.size())
        print(f"before model - batch_features[0].size()", batch_features[0].size())
        print(f"before model - lengths:", lengths)
        print(f"before model - names: {names}")
        # for batch_idx in batch_features:
        #     print("batch_idx", batch_idx.size())
        print()


        all_outputs, last_output = model(batch_features, lengths)
        print(f"after model - all_outputs.size: {all_outputs.size()}")
        print(f"after model - last_output.size: {last_output.size()}")
        # print(f"after model - last_output[0]: {last_output[0]}")
        break



    # Test = CustomLSTM(512, 256, 512)
    # test_frame_encoder_output = torch.rand(4, 200, 512)
    # test_future_context_encoder_output = torch.rand(4, 200, 256)
    #
    # output, (h, c) = Test(test_frame_encoder_output, test_future_context_encoder_output)
    # print(output.size())
    # print(h.size())
    # print(c.size())


    # just some testing again :)
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

    # MORE TESTING :D
    # for batch_idx, (sequences, lengths, names) in enumerate(trainloader):
    #     if batch_idx > 0:
    #         break
    #     print(f"sequenzes.size(): {sequences.size()}")
    #     print(f"length-tensor: {lengths}")
    #
    #     for i in range(test_batch_size):
    #         print(f"name: {names[i]}")
    #         print(f"length: {lengths[i]}")
    #         print(f"first sequence: {sequences[i]}")
    #         print(f"sequence size: {sequences[i].size()}")
    #         print()
    #         print(f"first frame AUs: {sequences[i][0]}")
    #         print(f"last frame AUs: {sequences[i][lengths[i]-1]}")
    #         print(f"difference first to last frame: {torch.sub(sequences[i][0], sequences[i][lengths[i]-1])}")
    #         break


        # print(f"name: {names[0]}")
        # print(f"length: {lengths[0]}")
        # print(f"first sequence: {sequences[0]}")
        # print(f"sequence size: {sequences[0].size()}")
        # print()
        # print(f"name: {names[1]}")
        # print(f"length: {lengths[1]}")
        # print(f"second sequence: {sequences[1]}")
        # print(f"sequence size (which should be as long as the first sequence: {sequences[1].size()}")
        # print(f"second sequence last real tensor: {sequences[1][lengths[1]-1]}")

        # prints above seem to print out correct stuff!






























