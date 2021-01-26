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

csv_read_path = r"Data\FaceTracker\preprocessed\new_csv"
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

        path = "./models/ExGen_Best" + str(version) + "_net.pth"
        torch.save(net.state_dict(), path)
    else:
        pass


# load an existing network's parameters and safe them into the just created net
def load_network(net: nn.Module):
    # save the network?
    load = input("Load Network? (y) or (n)?")

    if load == "y":
        # version = input("Which model should be loaded? (Version number): ")
        try:
            # path = "./models/triple_dataset/ExGen_" + str(version) + "_net.pth"
            # absolute path
            # path = "./models/last_try_s2s_balanced_ds/ExGen_Bo300_net.pth"
            # best model!!
            path = "./models/working_model/ExGen_Bo355_net.pth"

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


# 42 => ??
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    dataset = AUDataset(csv_read_path)
    trainset, valset, testset = torch.utils.data.random_split(dataset, [151, 19, 19])   # 189 -> 80% 10% 10% split!

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
            # encoding lstm
            self.encoding = nn.LSTM(input_size=n_features, hidden_size=n_output_encoder, num_layers=n_layers, dropout=p)
            self.linear = nn.Linear(n_output_encoder, n_features)

            # lstm - temporal information
            self.rnn = nn.LSTM(n_output_encoder + n_features, n_hidden, num_layers=n_layers, batch_first=True)

            # decoder ->
            self.decoding = nn.Linear(n_hidden, n_features)


        def forward(self, x, target, hidden, cell):
            """
            :param x: ist target besteht aus 10frames
            :param target: ist das ende der sequenz
            :return: prediction in Form eines einzelnen Frames
            """
            # some noise??
            # dropout??

            # difference target current frame??

            # encoding
            _, (hidden, cell) = self.encoding(x, (hidden, cell))
            # encoding = self.linear(hidden)                      # encoding size [1, 1, 15]
            feature = torch.cat([hidden, target], dim=2)

            # temporal dynamics
            frame_encoding, (self.hidden, self.cell) = self.rnn(feature, (self.hidden, self.cell))

            # decoding with fc
            prediction = self.decoding(frame_encoding)

            return prediction, hidden, cell

        def zero_hidden(self):
            self.hidden = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
            self.cell = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)

        def zero_hidden_encoding(self):
            hidden = torch.zeros(self.n_layers, 1, self.output_encoder_size).to(device)
            cell = torch.zeros(self.n_layers, 1, self.output_encoder_size).to(device)
            return hidden, cell


    # Hyperparameters
    num_epochs = 50
    learning_rate = 1e-5
    dropout = 0.5  # not used right now
    teacher_forcing_ratio = 0.5

    # safe path for models (best and last)
    save_best = "./models/last_try_s2s_balanced_ds/ExGen_Bo100_net.pth"
    save_last = "./models/last_try_s2s_balanced_ds/ExGen_B100_net.pth"

    # model
    model = ExpressionGenerator(15, 256, 512, 1, dropout)
    load_network(model)
    model = model.to(device)


    # define loss(es) and optimizer
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5, amsgrad=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # best current test error (MSE):
    best_error = 0.33288518129847944
    last_epoch = 300
    # model_safe = ExGen_0_B0_net - B for batch | 0_ is best model (MSE) from train, 1_ is last model from train

    def fifo(tensor, x):
        # print(tensor[1:])
        # print(x)
        return torch.cat((tensor[1:], x))

    # training loop
    def train_model(train: DataLoader, val: DataLoader, n_Epochs: int, best_test_error: float):
        writer = SummaryWriter()
        loss_history = []
        best_epoch = 293
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

                # create empty sequence tensor for whole anim:
                created_sequence = torch.zeros(1, seq_length, number_aus).to(device)

                sequence = batch_data[0, 0:20]
                # first 10 frames are known and should be copied
                for i in range(0, 20):
                    created_sequence[0][i] = sequence[i]

                sequence = sequence.unsqueeze(1)                # add dimension for batch [frames_given, 1, 15]
                # first_frame = batch_data[0, 0]
                last_frame = batch_data[0, -1]
                last_frame = last_frame.unsqueeze(0)
                last_frame = last_frame.unsqueeze(0)

                # für jede sequenz
                optimizer.zero_grad()
                model.zero_hidden()
                hidden, cell = model.zero_hidden_encoding()

                for t in range(20, seq_length):
                    prediction, hidden, cell = model(sequence, last_frame, hidden, cell)

                    created_sequence[0][t] = prediction

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
                    sequence = fifo(sequence, prediction)

                # loss calculation
                loss = mse_loss(created_sequence, batch_data)
                loss.backward()

                # grad clipping -> caused cuda-error
                # nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # upgrade gradients
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
                    # print(batch_data.size())      # size = [batch_size, sequence_length, n_features]
                    # print(name)                   # csv name
                    seq_length = batch_data.size(1)
                    number_aus = batch_data.size(2)

                    # create empty sequence tensor for whole anim:
                    created_sequence = torch.zeros(1, seq_length, number_aus).to(device)

                    sequence = batch_data[0, 0:20]
                    # first 10 frames are known and should be copied
                    for i in range(0, 20):
                        created_sequence[0][i] = sequence[i]

                    sequence = sequence.unsqueeze(1)  # add dimension for batch [10, 1, 15]
                    # first_frame = batch_data[0, 0]
                    last_frame = batch_data[0, -1]
                    last_frame = last_frame.unsqueeze(0)
                    last_frame = last_frame.unsqueeze(0)

                    # für jede sequenz
                    optimizer.zero_grad()
                    model.zero_hidden()
                    hidden, cell = model.zero_hidden_encoding()

                    for t in range(20, seq_length):
                        prediction, hidden, cell = model(sequence, last_frame, hidden, cell)
                        created_sequence[0][t] = prediction
                        # define new target
                        sequence = fifo(sequence, prediction)

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
                torch.save(model.state_dict(), save_best)
                best_test_error = val_loss_mse
                best_epoch = epoch
                print("New Model had been saved!")

            # adjust lr (by hand now for better results?
            # scheduler.step()

        # append to txt .. better save than sorry!
        with open(r'training_history\seq2seq_last.txt', 'a') as f:
            print(loss_history, file=f)

        writer.close()
        print("Best test error (for copy-paste):", best_test_error)
        print("Epoch (best test error):", best_epoch)
        print("Finished training!")
        torch.save(model.state_dict(), save_last)

    def test_model(test: DataLoader):
        model.eval()
        with torch.no_grad():
            test_loss_mse = 0
            test_loss_l1 = 0
            for index, data in enumerate(test):
                batch_data, name = data
                batch_data = batch_data.to(device)
                # print(batch_data.size())      # size = [batch_size, sequence_length, n_features]
                # print(name)                   # csv name
                seq_length = batch_data.size(1)
                number_aus = batch_data.size(2)

                # create empty sequence tensor for whole anim:
                created_sequence = torch.zeros(1, seq_length, number_aus).to(device)

                sequence = batch_data[0, 0:20]
                # first 10 frames are known and should be copied
                for i in range(0, 20):
                    created_sequence[0][i] = sequence[i]

                sequence = sequence.unsqueeze(1)  # add dimension for batch [10, 1, 15]
                # first_frame = batch_data[0, 0]
                last_frame = batch_data[0, -1]
                last_frame = last_frame.unsqueeze(0)
                last_frame = last_frame.unsqueeze(0)

                # für jede sequenz
                optimizer.zero_grad()
                model.zero_hidden()
                hidden, cell = model.zero_hidden_encoding()

                for t in range(20, seq_length):
                    prediction, hidden, cell = model(sequence, last_frame, hidden, cell)
                    created_sequence[0][t] = prediction
                    # define new target
                    sequence = fifo(sequence, prediction)

                loss_mse = mse_loss(created_sequence, batch_data)
                loss_l1 = l1_loss(created_sequence, batch_data)

                test_loss_mse = test_loss_mse + loss_mse.item()
                test_loss_l1 = test_loss_l1 + loss_l1.item()

        print(f"Test_losses: MSE = {test_loss_mse:.4f} | L1 = {test_loss_l1:.4f}")
        with open(r'test_history\seq2seq_last.txt', 'a') as f:
            print(f"MSE:{test_loss_mse}, L1:{test_loss_l1}", file=f)

    # train_model(train_loader, val_loader, num_epochs, best_error)
    # test_model(test_loader)

    # GENERATION

    def sequenize_start(start_frame: torch.Tensor):
        sequence = torch.empty(20, 15)
        for i in range(0, 20):
            sequence[i] = start_frame

        return sequence


    def generate_expression(start_frame: torch.Tensor, end_frame: torch.Tensor, sequence_length: int, anim_name: str):
        # eval the model on the test set
        model.eval()
        with torch.no_grad():


            last_frame = end_frame.to(device)
            last_frame = last_frame.unsqueeze(0)
            last_frame = last_frame.unsqueeze(0)
            number_aus = end_frame.size(0)

            created_sequence = torch.zeros(1, sequence_length, number_aus).to(device)

            sequence = sequenize_start(start_frame).to(device)
            # first 10 frames are known and should be copied
            for i in range(0, 20):
                created_sequence[0][i] = sequence[i]

            sequence = sequence.unsqueeze(1)  # add dimension for batch [20, 1, 15]

            model.zero_hidden()
            hidden, cell = model.zero_hidden_encoding()

            for t in range(20, sequence_length):
                prediction, hidden, cell = model(sequence, last_frame, hidden, cell)
                created_sequence[0][t] = prediction
                # define new target
                sequence = fifo(sequence, prediction)

            # for convenience:
            sequence = created_sequence

            sequence = sequence.cpu()
            sequence = sequence.squeeze(0)
            # print(prediction.size())

            # get right format for columns
            df = pd.read_csv(csv_read_path + "/neutralhappy2_lfill.csv")
            header = list(df.drop(["Frame"], axis=1))
            # df.close()
            del df

            # generate new name for the generated animation
            new_name = "ExGen_" + str(anim_name)

            # transform predictions to csv
            sequence_np = sequence.numpy()
            sequence_df = pd.DataFrame(sequence_np)
            sequence_df.columns = header
            # prediction_df.columns = ["AU1L","AU1R","AU2L","AU2R","AU4L","AU4R","AU6L","AU6R","AU9","AU10","AU13L","AU13R","AU18","AU22","AU27"]
            sequence_df.to_csv(gen_save_pth + new_name + ".csv")
            del sequence_np
            del sequence_df

    ####### GENERATION #######

    # frown to suprised (not in data!)
    # start = torch.Tensor([0.06131799283585899,0.08172666935186508,0.03395699517793453,0.04818290386941117,1.1412230401159085,1.1035029605516014,0.113120523744227,1.0597114447811183e-07,0.30072830873532025,3.517824015742137e-05,8.749662808520226e-09,0.0013996257573782305,0.6260813195906696,0.09551505350147324,6.689342889722282e-10])
    # end = torch.Tensor([1.1999996314449428,1.1999995870728282,1.0278275208869838,1.0194900942312504,0.2341106233151145,0.0653982846386197,0.8537125658024893,0.8081682829103988,1.4364089873377762e-08,-2.1860498663560296e-08,9.06581770646935e-07,0.006658209848300654,-4.7158831256122355e-09,6.94503947633974e-07,0.9977853517181079])

    # disgust1 to happy5 (more or less in data)
    # start = torch.Tensor([6.716317271550969e-08, 0.02935825960378735, 0.021760095040568358, 0.08411739822017017, 1.1999998885514678, 1.1846442213750967, 0.2946981952170156, 0.14367314787076094, 0.0014814104832844687, 0.30272492210729074, 0.3789290959138957, 1.1742008603708393e-08, 4.753175528420333e-09, 4.965092547141178e-09, 1.8304739182918754e-09])
    # end = torch.Tensor([0.06125034864614825,0.02625214569476484,4.114325553268089e-09,0.02274558697252165,0.17967660446464898,0.13432276325223844,0.15313606416869646,0.12118201135523902,3.097393968560188e-08,9.003948174507734e-10,0.1875999910930843,-3.096835818564804e-11,-1.6041286239695211e-10,2.658632689948891e-10,0.041449114407351585])

    # disgust_suprised5 end
    # end = torch.Tensor([1.2000000063515006,1.2000000037242264,1.20000000988556,1.20000000517873,-6.258249112550545e-09,-5.344243176490128e-09,0.8856496208847644,0.7621254478783975,2.209424007560319e-09,-1.967204045131492e-08,9.93416993606448e-07,0.31120050895524043,3.8454029649169626e-07,0.18082177539922012,0.9725068273045891])

    # sup_neutral2 end
    # end = torch.Tensor([0.07738269721417339,0.08807293900396353,0.011010792242784829,0.06143003939298607,0.029883286893899432,1.3437291391474005e-10,6.630145104358612e-10,1.0842664182092701e-10,4.41634207694271e-11,1.8048909959586743e-11,1.7592160096523457e-10,1.8230500751428678e-11,0.19040087231317004,7.341144061507404e-11,1.1859499592224213e-09])

    # start happy_neutral3
    # start = torch.Tensor([1.7312022381517743e-07,9.942494218012084e-07,5.5279378574193394e-08,7.275862012056518e-08,0.1614868551455088,0.10531355676472627,1.199833379749608,1.1999999171354383,6.253415944733785e-08,0.25089657727780995,0.5664661956248268,1.0567180085991217,5.8532759285127815e-09,4.522632508438027e-09,1.2681872116221288e-07])

    # suprise_disgust3
    # start = torch.Tensor([1.199998046094369,1.1999726704973803,1.1999998118766648,0.9563605766628656,1.5585242134849914e-07,7.692463816309587e-07,6.250386339331801e-08,6.782585383713505e-08,9.492120125413018e-09,4.076888149768646e-09,9.729325069729388e-09,5.925533415555281e-09,1.9051411842377304e-09,3.874445721084235e-09,1.1999999482368842])

    # suprise_happy5
    # end = torch.Tensor([1.199998046094369,1.1999726704973803,1.1999998118766648,0.9563605766628656,1.5585242134849914e-07,7.692463816309587e-07,6.250386339331801e-08,6.782585383713505e-08,9.492120125413018e-09,4.076888149768646e-09,9.729325069729388e-09,5.925533415555281e-09,1.9051411842377304e-09,3.874445721084235e-09,1.1999999482368842])

    # happy_sad5
    # end = torch.Tensor([0.17688237604716348,0.19866736744246646,0.11185978632743752,0.09487987030174973,1.0548543034029183,1.1542045333747448,0.0870861610479417,0.22915305471376285,0.01832295349686015,2.090484285219315e-08,0.03243690786750332,0.006573015681693785,0.20954841526969845,1.2414312200954819e-09,1.559739671024413e-09])
    # start from that anim!!
    # end = torch.Tensor([2.6904757304193483e-07,3.2651399899997965e-07,1.1110361348045573e-07,-4.68313755856502e-08,0.17775043528894266,0.07765913019549357,1.0402484487911563,1.123246129709042,8.839867718517393e-09,0.28844375448016346,0.8882674808572604,1.199999890009607,-5.074225205649736e-08,-5.6006952442998216e-08,-5.990424392451736e-08])

    # happy_frown5 -> frown
    frown = torch.Tensor(
        [0.11600816761214412, 0.1284655902491436, 0.04555017236853346, 0.07891883398256783, 1.1624704923767863,
         1.1292259224988748, 4.1179628134090425e-09, 1.3139848525048626e-08, 0.3780972373525189,
         0.002627721186276977, 0.4052149252538391, 0.006900503041645656, 2.656951757012366e-10,
         -8.446394673406755e-10, -9.975166643120387e-10])

    # surprise_disgust2 -> surprise
    surprise = torch.Tensor([1.1999999112205617,1.1999999159976815,1.1999999328538695,1.1488722636177362,4.595641601005289e-08,1.7807415136001172e-07,4.928640608924983e-05,3.610486793096247e-05,9.383047183030742e-08,-1.3288099099324539e-08,2.009467533535717e-08,-1.4430244360720015e-09,-2.4141230449188912e-08,5.907091004922681e-09,1.2000000099788897])

    # happy_sad5 -> sad
    sad = torch.Tensor([0.1469748339078054, 0.1645827189561475, 0.09715293786052676, 0.09448502011811992,
                          1.0443333135088018,1.1510849319027174, 0.045217939313237115, 0.20751686582277729,
                          0.0356037971287012, 2.5040262983977084e-08,0.024025351834480672, 0.00011304006851385568,
                          0.17286549919783545, 4.5758218435735396e-08,1.1034485458071822e-08])

    # happy_neutral -> neutral
    neutral = torch.Tensor([1.3183410608020914e-09,0.007341170604253437,6.560640186486611e-10,6.156423513747244e-10,0.05534238495332509,0.03196604997678923,4.029073564969686e-07,0.05242904922403188,2.121624087550293e-09,3.935684714177853e-09,-4.39079980658653e-12,0.06648207797258246,0.08107495649839419,4.873076503851266e-10,0.02053503309248228])

    # happy_neutral -> happy
    happy = torch.Tensor([1.7312022381517743e-07,9.942494218012084e-07,5.5279378574193394e-08,7.275862012056518e-08,0.1614868551455088,0.10531355676472627,1.199833379749608,1.1999999171354383,6.253415944733785e-08,0.25089657727780995,0.5664661956248268,1.0567180085991217,5.8532759285127815e-09,4.522632508438027e-09,1.2681872116221288e-07])


    # surprise_disgust -> disgust
    disgust = torch.Tensor([1.0670191593715594e-07,3.2528796729084693e-07,0.12982850058516537,0.06965097196072645,1.199999695066714,1.1706918609055257,0.2146797932278581,0.046555729206558386,0.042598584997725494,0.4542929713122771,0.401825638510415,8.596737489378636e-07,4.7000663724022786e-08,1.2655896810496993e-08,1.0064052277759405e-08])

    # generate_expression(start, end, DURATION, "Test_triple_DURATION_WHICHMODEL_FROM_to_TO")
    generate_expression(neutral, frown, 250, "neutral2frown")

    # custom safe method which can be used to store individual models (name as input during method)
    # save_network(model)

###### TRAIN LIKE NOW and then for generation just copy 1 Frame 20 times and go from there!
# Try to train it with 20 copied frame maybe (-> Time!) => Outlook

# Diskussion/Ergebnisse
# Discuss differences with generating with 20 real frame or 20 copied
# questionnaire as an addendum

# Presentation
### Quick why my thesis is there, what is the goal, what is my data -> Baseline (2 Slides)
### Start with System (see where I stopped last time so to say!)

