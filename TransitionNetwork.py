import math
import random
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


# set up the divice (GPU or CPU) via input prompt
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
num_epochs = 20
train_batch_size = 1
test_batch_size = 1
learning_rate = 5e-5

teacher_forcing_ratio = 0.1

# input_size = ?
# sequence_length = ?

# # evtl quatsch
# tensor(0.2908) tensor(0.1124)
# mean = 0.2908
# std = 0.1124
# transforms = transforms.Compose([
#     transforms.Normalize(mean, std),
# ])


torch.manual_seed(0)


if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)
    trainset, testset = torch.utils.data.random_split(dataset, [70, 15])


    # test_before_loader, _ = dataset[0]
    # print("test_before_loader.type():", test_before_loader.type())

    trainloader = DataLoader(dataset=trainset, batch_size=train_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=False)
    testloader = DataLoader(dataset=testset, batch_size=test_batch_size, collate_fn=PadSequencer(), shuffle=True, num_workers=0, drop_last=False)


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
            self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
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


        def init_hidden(self, batch_size):
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            h0 = h0.to(device=device)
            c0 = c0.to(device=device)
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

    # def transition_network_loss(predicted_frame, real_frame):
    #     loss = F.mse_loss(predicted_frame.view(-1, 15), real_frame.view(-1, 15))
    #     return loss


    # Transition Network
    TransitionAE = TransitionNetwork(15)
    load_network(TransitionAE)

    TransitionAE.to(device=device)


    optimizer = optim.Adam(TransitionAE.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    def train(nEpochs: int):
        TransitionAE.train()
        print("Training...")


        for epoch in range(nEpochs):
            running_loss = 0.0

            for index, data in enumerate(trainloader):
                batch_features, lengths, names = data
                batch_features = batch_features.to(device=device)
                # lengths = lengths.to(device=device)

                # print(f"before model - batch_features.size()", batch_features.size())
                # print(f"before model - batch_features[0].size()", batch_features[0].size())
                # print(f"before model - lengths:", lengths)
                # print(f"before model - names: {names}")

                # for batch_idx in batch_features:
                #     print("batch_idx", batch_idx.size())
                # batch is: [batch_size, sequence_length, feature_size]
                # calculate current frame, offset and target

                # print(
                #     f"++++++++++++++++++++++++++++++ index (consists of a batch of batch_size): {index} +++++++++++++++++++++++++++++++++")


                # targets for all batches (hyperparams)
                target = torch.empty(train_batch_size, batch_features.size(2))
                for i in range(train_batch_size):
                    target[i] = batch_features[i][lengths[i] - 1][:]

                # target to device
                target = target.to(device=device)

                # init hidden per batch
                hidden, cell = TransitionAE.lstm.init_hidden(train_batch_size)


                # print(f"before model - target.size(): {target.size()}")
                # print(f"before model - target: {target}")
                # print()
                # print("PER FRAME CALCULATIONS")
                # print()

                for frame in range(lengths[0]):
                    # try:
                    #     print(f"====================> frame: {count}")
                    # except:
                    #     print(f"====================> frame: 0")
                    # count = TransitionAE.forward(batch_features[:, frame, :], target)
                    # print(f"----------------------- new frame ----------------------")
                    # print(batch_features[:, frame, :])


                    # teacher forcing with prob = 0.2
                    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

                    if frame < (lengths[0] - 1):
                        real_next_frame = batch_features[:, frame + 1, :]
                    else:
                        real_next_frame = target

                    # when teacher forcing true: feed the real next_frame to the network
                    if use_teacher_forcing:
                        predicted_next_frame = real_next_frame

                    predicted_next_frame, hidden, cell = TransitionAE(batch_features[:, frame, :], target, hidden,
                                                                      cell)

                    hidden.detach_()
                    cell.detach_()

                    # print(f"after model - predicted_next_frame: {predicted_next_frame.size()}")
                    # print(f"after model - real_next_frame: {real_next_frame.size()}")


                    loss = criterion(predicted_next_frame, real_next_frame)
                    # loss = transition_network_loss(predicted_next_frame, real_next_frame)
                    # print(f"after model - loss: {loss}")

                    # backprop
                    optimizer.zero_grad()
                    loss.backward()

                    # one step of the optimizer
                    optimizer.step()

                    running_loss += loss.item()

                    # print(hidden)
                    # print(cell)
                    # print(predicted_next_frame.size())

                    # end frame

                # end batch -> all frames
                if index % 20 == 20 - 1:
                    print(f"epoch: {epoch + 1}, mini-batch: {index + 1} - loss: {running_loss / 20}")
                    running_loss = 0.0

            # end epoch
            # print(f"epoch: {epoch + 1}")


        print("Training finished!")
        save_network(TransitionAE)

    def test(loader: DataLoader, model):
        model.eval()

        with torch.no_grad():
            running_loss = 0.0
            all_loss = 0.0

            for index, data in enumerate(loader):
                batch_features, lengths, names = data
                batch_features = batch_features.to(device=device)

                # targets for all batches (hyperparams)
                target = torch.empty(test_batch_size, batch_features.size(2))
                for i in range(test_batch_size):
                    target[i] = batch_features[i][lengths[i] - 1][:]

                # target to device
                target = target.to(device=device)

                # init hidden per batch
                hidden, cell = TransitionAE.lstm.init_hidden(test_batch_size)

                for frame in range(lengths[0]):

                    if frame < (lengths[0] - 1):
                        real_next_frame = batch_features[:, frame + 1, :]
                    else:
                        real_next_frame = target

                    # print(batch_features[:, frame, :].size())
                    # exit()

                    predicted_next_frame, hidden, cell = TransitionAE(batch_features[:, frame, :], target, hidden,
                                                                          cell)


                    loss = criterion(predicted_next_frame, real_next_frame)
                    running_loss += loss.item()
                    # end frame
                
                # end batch
                print(f"Running Loss: {running_loss}")
                all_loss += running_loss
                running_loss = 0.0

            print(f"all losses: {all_loss}")


    # def generate_sequence(model, start_frame, target_frame, duration):
    #     model.eval()
    #     transition = []
    #
    #     start_frame = torch.unsqueeze(start_frame, 0).to(device=device)
    #     target_frame = torch.unsqueeze(target_frame, 0).to(device=device)
    #
    #     hidden = None
    #     cell = None
    #
    #     with torch.no_grad():
    #
    #         for i in range(duration-1):
    #             predicted_next_frame, hidden, cell = TransitionAE(start_frame, target_frame, hidden, cell)
    #             transition.append(predicted_next_frame)
    #         return transition

    train(num_epochs)
    test(testloader, TransitionAE)

    start = torch.tensor([1.1735114637781786e-08,7.241975343010249e-08,0.018730973801541314,0.09275460610676707,1.1999998418906612,1.176913967268185,0.1643039178064644,0.18280748342566253,0.17407648701965087,0.2411355052111981,0.229750059760292,3.626090594941871e-08,1.2085446863230588e-09,8.463887096639834e-11,2.5167801468923616e-10])
    end = torch.tensor([1.1999996482175648,1.199999459762757,1.199999349722415,1.1999967071002502,3.730670327388917e-07,2.192692481468664e-05,0.8932812902538821,0.9212007896416295,3.333123421165009e-07,1.7824243257347232e-07,3.263498064039033e-07,3.45552188066782e-07,4.02158899625008e-06,0.05982334779796216,1.1986696864049995])
    #print(start.size())
    #print(end.size())
    t = 10

    # transition = generate_sequence(TransitionAE, start, end, t)
    # print(transition)

































