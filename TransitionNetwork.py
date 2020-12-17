import math
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

    class CustomLSTM(nn.Module):
        def __init__(self, frame_encoder_size: int, future_context_encoder_size: int, hidden_size: int):
            """
            :param frame_encoder_size: size of the output of the frame encoder (expected 512)
            :param future_context_encoder_size: size of concatenated Future Context (expected 128 + 128 = 256)
            :param hidden_size: hidden size (expected: 512)
            assumes that later the data has batch_first, hidden_size, encoder1_size, encoder"2"_size
            """
            super(CustomLSTM, self).__init__()
            self.frame_encoder_size = frame_encoder_size
            self.future_context_encoder_size = future_context_encoder_size
            self.hidden_size = hidden_size

            # optimized!
            self.W = nn.Parameter(torch.Tensor(frame_encoder_size, hidden_size * 4))
            self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
            self.C = nn.Parameter(torch.Tensor(future_context_encoder_size, hidden_size * 4))
            self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))

            self.init_weights()


            # not optimized!
            # # i_t (1)
            # self.W_i = nn.Parameter(torch.Tensor(frame_encoder_size, hidden_size))
            # self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            # self.C_i = nn.Parameter(torch.Tensor(future_context_encoder_size, hidden_size))
            # self.b_i = nn.Parameter(torch.Tensor(hidden_size))
            #
            # # o_t (2)
            # self.W_o = nn.Parameter(torch.Tensor(frame_encoder_size, hidden_size))
            # self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            # self.C_o = nn.Parameter(torch.Tensor(future_context_encoder_size, hidden_size))
            # self.b_o = nn.Parameter(torch.Tensor(hidden_size))
            #
            # # f_t (3)
            # self.W_f = nn.Parameter(torch.Tensor(frame_encoder_size, hidden_size))
            # self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            # self.C_f = nn.Parameter(torch.Tensor(future_context_encoder_size, hidden_size))
            # self.b_f = nn.Parameter(torch.Tensor(hidden_size))
            #
            # # c_t (5)
            # self.W_c = nn.Parameter(torch.Tensor(frame_encoder_size, hidden_size))
            # self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            # self.C_c = nn.Parameter(torch.Tensor(future_context_encoder_size, hidden_size))
            # self.b_c = nn.Parameter(torch.Tensor(hidden_size))


        def init_weights(self):
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)


        def forward(self, hEt, hFOt, init_states=None):
            """
            :param hEt:
            :param hFOt:
            :param init_states:
            :return:

            assumes hEt.shape represents (batch_size, sequence_length, frame_encoder_size,
            assumes hFOt.shape represents (batch_size, sequence_length, future_context_encoder_size
            """

            batch_size, sequence_length, _ = hEt.size()
            # batch_size, sequence_length, _ = hFOt.size()
            hidden_sequence = []

            if init_states is None:
                h_t, c_t = (
                    torch.zeros(batch_size, self.hidden_size).to(device=hEt.device),    # to(hEt.device) means same device as hEt I think
                    torch.zeros(batch_size, self.hidden_size).to(device=hFOt.device),   # to make sure devices are all the same!
                )
            else:
                h_t, c_t = init_states


            # optimized!
            HS = self.hidden_size
            for t in range(sequence_length):
                hE_t = hEt[:, t, :]
                hFO_t = hFOt[:, t, :]

                # batch the computations into a single matrix multiplication
                gates = hE_t @ self.W + h_t @ self.U + hFO_t @ self.C + self.bias
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),       # input
                    torch.sigmoid(gates[:, HS:HS*2]),   # forget
                    torch.tanh(gates[:, HS*2:HS*3]),
                    torch.sigmoid((gates[:, HS*3:])),   # output
                )

                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)
                hidden_sequence.append(h_t.unsqueeze(0))

            hidden_sequence = torch.cat(hidden_sequence, dim=0)
            # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
            hidden_sequence = hidden_sequence.transpose(0, 1).contiguous()
            return hidden_sequence, (h_t, c_t)


            # not optimized!
            # for t in range(sequence_length):
            #     hE_t = hEt[:, t, :]
            #     hFO_t = hFOt[:, t, :]
            #
            #     i_t = torch.sigmoid(hE_t @ self.W_i + h_t @ self.U_i + hFO_t @ self.C_i + self.b_i)
            #     o_t = torch.sigmoid(hE_t @ self.W_o + h_t @ self.U_o + hFO_t @ self.C_o + self.b_o)
            #     f_t = torch.sigmoid(hE_t @ self.W_f + h_t @ self.U_f + hFO_t @ self.C_f + self.b_f)
            #     g_t = torch.tanh(hE_t @ self.W_c + h_t @ self.U_c + hFO_t @ self.C_c + self.b_c)
            #
            #     c_t = f_t * c_t + i_t * g_t
            #     h_t = o_t * torch.tanh(c_t)
            #
            #     hidden_sequence.append(h_t.unsqueeze(0))
            #
            # # reshape hidden sequence p/ retornar
            # hidden_sequence = torch.cat(hidden_sequence, dim=0)
            # hidden_sequence = hidden_sequence.transpose(0, 1).contiguous()
            # return hidden_sequence, (h_t, c_t)





    Test = CustomLSTM(512, 256, 512)
    test_frame_encoder_output = torch.rand(4, 200, 512)
    test_future_context_encoder_output = torch.rand(4, 200, 256)

    output, (h, c) = Test(test_frame_encoder_output, test_future_context_encoder_output)
    print(output.size())
    print(h.size())
    print(c.size())



    class FrameEncoder(nn.Module):
        def __init__(self):
            super(FrameEncoder, self).__init__()
            pass


    class TargetEncoder(nn.Module):
        def __init__(self):
            super(TargetEncoder, self).__init__()
            pass


    class OffsetEncoder(nn.Module):
        def __init__(self):
            super(OffsetEncoder, self).__init__()
            pass


    class RecurrentGenerator(nn.Module):
        def __init__(self):
            super(RecurrentGenerator, self).__init__()
            pass


    class FrameDecoder(nn.Module):
        def __init__(self):
            super(FrameDecoder, self).__init__()
            pass



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
    #     if batch_idx > 1:
    #         break
    #     print(f"name: {names[0]}")
    #     print(f"length: {lengths[0]}")
    #     print(f"first sequence: {sequences[0]}")
    #     print(f"sequence size: {sequences[0].size()}")
    #     print()
    #     print(f"name: {names[1]}")
    #     print(f"length: {lengths[1]}")
    #     print(f"second sequence: {sequences[1]}")
    #     print(f"sequence size (which should be as long as the first sequence: {sequences[1].size()}")
    #     print(f"second sequence last real tensor: {sequences[1][lengths[1]-1]}")
    #
    #     # prints above seem to print out correct stuff!
    #     # TODO: TransitionNetwork implementation!






























