import torch
import pandas as pd
import logging
import os
import sys
from data_reader import DataReader
from constants import *
from aggregator import *
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_logger(name, save_dir, save_filename):
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=DATE_FORMAT)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def select_by_threshold(to_share: torch.Tensor, select_fraction: float, select_threshold: float = 1):
    """
    Apply the privacy-preserving method following selection-by-threshold approach
    """
    threshold_count = round(to_share.size(0) * select_threshold)
    selection_count = round(to_share.size(0) * select_fraction)
    indices = to_share.topk(threshold_count).indices
    perm = torch.randperm(threshold_count).to(DEVICE)
    indices = indices[perm[:selection_count]]
    rei = torch.zeros(to_share.size()).to(DEVICE)
    rei[indices] = to_share[indices].to(DEVICE)
    to_share = rei.to(DEVICE)
    return to_share, indices


class ModelPurchase100(torch.nn.Module):
    """
    The model handling purchase-100 data set
    """

    def __init__(self):
        super(ModelPurchase100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(600, 1024),
            torch.nn.ReLU()
        )
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out


class ModelPreTrainedCIFAR10(torch.nn.Module):
    """
    The model to support pre-trained CIFAR-10 data set
    """

    def __init__(self):
        super(ModelPreTrainedCIFAR10, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(64, 1024),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelLocation30(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelLocation30, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(446, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelTexas100(torch.nn.Module):
    """
    The model to handel Texas10 dataset
    """

    def __init__(self):
        super(ModelTexas100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(6169, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelMNIST(torch.nn.Module):
    """
    The model handling MNIST dataset
    """

    def __init__(self):
        super(ModelMNIST, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(784, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelGnome(torch.nn.Module):
    """
    The model handling Gnome dataset
    """

    def __init__(self):
        super(ModelGnome, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(5547, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 5)
        )

    def forward(self, x):
        return self.network(x)


class TargetModel:
    """
    The model to attack against, the target for attacking
    """

    def __init__(self, data_reader: DataReader, participant_index=0, model=DEFAULT_SET):
        # initialize the model
        if model == PURCHASE100:
            self.model = ModelPurchase100()
        elif model == CIFAR_10:
            self.model = ModelPreTrainedCIFAR10()
        elif model == LOCATION30:
            self.model = ModelLocation30()
        elif model == TEXAS100:
            self.model = ModelTexas100()
        elif model == MNIST:
            self.model = ModelMNIST()
        elif model == GNOME:
            self.model = ModelGnome()
        else:
            raise NotImplementedError("Model not supported")

        self.model = self.model.to(DEVICE)

        # initialize the data
        self.test_set = None
        self.train_set = None
        self.data_reader = data_reader
        self.participant_index = participant_index
        self.load_data()

        # initialize the loss function and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Initialize recorder
        self.train_loss = -1
        self.train_acc = -1

        # Initialize confidence recorder
        self.mask = torch.ones(BATCH_SIZE)
        self.defend = False
        self.defend_count_down = 0
        self.defend_loss_checker = self.train_loss
        self.drop_out = BATCH_SIZE // 4

    def load_data(self):
        """
        Load batch indices from the data reader
        :return: None
        """
        self.train_set = self.data_reader.get_train_set(self.participant_index).to(DEVICE)
        self.test_set = self.data_reader.get_test_set(self.participant_index).to(DEVICE)

    def activate_defend(self):
        """
        Activate defend function for this participant
        """
        self.defend = True
        self.defend_count_down = 5
        self.generate_mask()
        self.defend_loss_checker = self.train_loss

    def generate_mask(self):
        """
        Generate a random mask
        """
        if self.defend_loss_checker > self.train_loss:
            self.drop_out -= 1
            # print("Dropout decrease, current = {}".format(self.drop_out))
        else:
            self.drop_out += 1
            # print("Dropout increase, current = {}".format(self.drop_out))
        if self.drop_out > BATCH_SIZE // 2:
            self.drop_out = BATCH_SIZE // 2
        if self.drop_out < 8:
            self.drop_out = 8
        temp_mask = torch.zeros(BATCH_SIZE)
        rand = torch.randperm(BATCH_SIZE)
        rand = rand[:BATCH_SIZE - self.drop_out]
        temp_mask[rand] = 1
        self.mask = temp_mask

    def normal_epoch(self, print_progress=False, by_batch=BATCH_TRAINING):
        """
        Train a normal epoch with the given dataset
        :param print_progress: if print the training progress or not
        :param by_batch: True to train by batch, False otherwise
        :return: the training accuracy and the training loss value
        """
        if self.defend:
            if self.defend_count_down > 0:
                self.defend_count_down -= 1
            else:
                self.generate_mask()
                self.defend_loss_checker = self.train_loss
                self.defend_count_down = 5
        train_loss = 0
        train_acc = 0
        batch_counter = 0
        if by_batch:
            for batch_indices in self.train_set:
                batch_counter += 1
                if print_progress and batch_counter % 100 == 0:
                    print("Currently training for batch {}, overall {} batches"
                          .format(batch_counter, self.train_set.size(0)))
                if self.defend:
                    batch_indices = batch_indices[self.mask == 1]
                batch_x, batch_y = self.data_reader.get_batch(batch_indices)
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                out = self.model(batch_x).to(DEVICE)
                # print("The size of output = {}, the size of label = {}".format(out.size(), batch_y.size()))
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                # if self.defend:
                #     confidence = torch.nn.Softmax(1)(out)
                #     # confidence = out
                #     max_conf = torch.max(confidence, 1).values.to(DEVICE)
                #     for i in range(confidence.size(0)):
                #         sample = batch_indices[i]
                #         sample = sample.item()
                #         self.record_confidence(sample, max_conf[i])
                batch_acc = (prediction == batch_y).sum()
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
        else:
            batch_x, batch_y = self.data_reader.get_batch(self.train_set)
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            out = self.model(batch_x).to(DEVICE)
            # print("The size of output = {}, the size of label = {}".format(out.size(), batch_y.size()))
            batch_loss = self.loss_function(out, batch_y)
            train_loss += batch_loss.item()
            prediction = torch.max(out, 1).indices.to(DEVICE)
            batch_acc = (prediction == batch_y).sum()
            train_acc += batch_acc.item()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        self.train_acc = train_acc / (self.train_set.flatten().size(0))
        self.train_loss = train_loss / (self.train_set.flatten().size(0))
        if print_progress:
            print("Epoch complete for participant {}, train acc = {}, train loss = {}"
                  .format(self.participant_index, train_acc, train_loss))
        return self.train_loss, self.train_acc

    def test_outcome(self, by_batch=BATCH_TRAINING):
        """
        Test through the test set to get loss value and accuracy
        :return: the test accuracy and test loss value
        """
        test_loss = 0
        test_acc = 0
        if by_batch:
            for batch_indices in self.test_set:
                batch_x, batch_y = self.data_reader.get_batch(batch_indices)
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                # print(batch_x)
                with torch.no_grad():
                    out = self.model(batch_x).to(DEVICE)
                    batch_loss = self.loss_function(out, batch_y).to(DEVICE)
                    test_loss += batch_loss.item()
                    prediction = torch.max(out, 1).indices.to(DEVICE)
                    batch_acc = (prediction == batch_y).sum().to(DEVICE)
                    test_acc += batch_acc.item()
        else:
            batch_x, batch_y = self.data_reader.get_batch(self.test_set)
            with torch.no_grad():
                out = self.model(batch_x)
                batch_loss = self.loss_function(out, batch_y)
                test_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices
                batch_acc = (prediction == batch_y).sum()
                test_acc += batch_acc.item()
        test_acc = test_acc / (self.test_set.flatten().size(0))
        test_loss = test_loss / (self.test_set.flatten().size(0))
        return test_loss, test_acc

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to(DEVICE)
        with torch.no_grad():
            for parameter in self.model.parameters():
                out = torch.cat([out, parameter.flatten()]).to(DEVICE)
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.model.parameters():
            length = len(param.flatten())
            to_load = parameters[start_index: start_index + length].to(DEVICE)
            to_load = to_load.reshape(param.size()).to(DEVICE)
            with torch.no_grad():
                param.copy_(to_load).to(DEVICE)
            start_index += length

    def get_epoch_gradient(self, apply_gradient=True):
        """
        Get the gradient for the current epoch
        :param apply_gradient: if apply the gradient or not
        :return: the tensor contains the gradient
        """
        cache = self.get_flatten_parameters().to(DEVICE)
        self.normal_epoch()
        gradient = self.get_flatten_parameters() - cache.to(DEVICE)
        if not apply_gradient:
            self.load_parameters(cache)
        return gradient

    def init_parameters(self, mode=INIT_MODE):
        """
        Initialize the parameters according to given mode
        :param mode: the mode to init with
        :return: None
        """
        if mode == NORMAL:
            to_load = torch.randn(self.get_flatten_parameters().size())
            self.load_parameters(to_load)
        elif mode == UNIFORM:
            to_load = torch.rand(self.get_flatten_parameters().size())
            self.load_parameters(to_load)
        elif mode == PYTORCH_INIT:
            return
        else:
            raise ValueError("Invalid initialization mode")

    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        """
        cache = self.get_flatten_parameters()
        test_param = cache + gradient
        self.load_parameters(test_param)
        loss, acc = self.test_outcome()
        self.load_parameters(cache)
        return loss, acc


class FederatedModel(TargetModel):
    """
    Representing the class of federated learning members
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator, participant_index=0):
        super(FederatedModel, self).__init__(reader, participant_index)
        self.aggregator = aggregator

    def update_aggregator(self,aggregator):
        self.aggregator = aggregator

    def get_aggregator(self):
        return self.aggregator

    def init_global_model(self):
        """
        Initialize the current model as the global model
        :return: None
        """
        self.init_parameters()
        if DEFAULT_DISTRIBUTION is None:
            self.test_set = self.data_reader.test_set.to(DEVICE)
        elif DEFAULT_DISTRIBUTION == CLASS_BASED:
            length = self.data_reader.test_set.size(0)
            length -= length % BATCH_SIZE
            self.test_set = self.data_reader.test_set[:length].reshape((-1, BATCH_SIZE)).to(DEVICE)
        self.train_set = None

    def init_participant(self, global_model: TargetModel, participant_index):
        """
        Initialize the current model as a participant
        :return: None
        """
        self.participant_index = participant_index
        self.load_parameters(global_model.get_flatten_parameters())
        self.load_data()

    def share_gradient(self, noise_scale=NOISE_SCALE, agr=False):
        """
        Participants share gradient to the aggregator
        :return: None
        """
        gradient = self.get_epoch_gradient()
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        noise = torch.randn(gradient.size()).to(DEVICE)
        noise = (noise/noise.norm()) * noise_scale * gradient.norm()
        # print("gradient norm before add noise {}".format(gradient.norm()), end = "")
        gradient += noise
        # print("gradient norm after add noise {}".format(gradient.norm()))
        if agr:
            self.aggregator.agr_loss_gradient_collect(gradient, indices)
        else:
            self.aggregator.collect(gradient, indices=indices, source=self.participant_index)
        return gradient

    def apply_gradient(self):
        """
        Global model applies the gradient
        :return: None
        """
        parameters = self.get_flatten_parameters()
        parameters += self.aggregator.get_outcome(reset=True)
        self.load_parameters(parameters)

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Participants collect parameters from the global model
        :param parameter: the parameters shared by the global model
        :return: None
        """
        to_load = self.get_flatten_parameters().to(DEVICE)
        parameter, indices = select_by_threshold(parameter, PARAMETER_EXCHANGE_RATE, PARAMETER_SAMPLE_THRESHOLD)
        to_load[indices] = parameter[indices]
        self.load_parameters(to_load)


class BlackBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to perform a black-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        super(BlackBoxMalicious, self).__init__(reader, aggregator)
        self.attack_samples, self.members, self.non_members = reader.get_black_box_batch()
        self.member_count = 0
        self.batch_x, self.batch_y = self.data_reader.get_batch(self.attack_samples)
        self.shuffled_y = self.shuffle_label(self.batch_y)
        for i in self.attack_samples:
            if i in reader.train_set:
                self.member_count += 1

    def shuffle_label(self, ground_truth):
        result = ground_truth[torch.randperm(ground_truth.size()[0])]
        for i in range(ground_truth.size()[0]):
            while result[i].eq(ground_truth[i]):
                result[i] = torch.randint(ground_truth.max(), (1,))
        return result

    def train(self, agr=False, attack=False):
        """
        Perform one round of gradient ascent attack using the given attack samples
        """
        cache = self.get_flatten_parameters()
        # batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        loss = None
        out = self.model(self.batch_x)
        if attack:
            loss = self.loss_function(out, self.shuffled_y)
        else:
            loss = self.loss_function(out, self.batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache

        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if attack:
            if agr:
                self.aggregator.agr_loss_gradient_collect(gradient, indices)
            else:
                self.aggregator.collect(gradient, indices)
        return gradient

    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member

    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate.cpu().numpy() / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate.cpu().numpy() / len(batch_y)


class GreyBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to perform a grey-box membership inference attack
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        super(GreyBoxMalicious, self).__init__(reader, aggregator)
        self.attack_samples, self.members, self.non_members = reader.get_black_box_batch()
        self.b = 0
        self.step = 0.05
        self.lamda = 0.2
        self.member_count = 0
        self.pred_history = []
        self.pred_history.append([])
        self.pred_history.append([])
        self.confidence_history = []
        self.confidence_history.append([])
        self.confidence_history.append([])
        for i in self.attack_samples:
            if i in reader.train_set:
                self.member_count += 1

    def prune_data(self, label=KEEP_CLASS):
        if label == None:
            return None
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        index = []
        for i in range(len(batch_y)):
            if batch_y[i] == label:
                #self.attack_samples=torch.cat([self.attack_samples[0:i],self.attack_samples[i+1:]])
                index.append(i)
                
        self.attack_samples = self.attack_samples[index].to(DEVICE)

    def get_single_member(self, partIndex=0):
        self.attack_samples = self.data_reader.get_train_set(self, participant_index=partIndex)

    def init_attacker(self):
        self.train_set = self.attack_samples
        # self.test_set = self.data_reader.get_test_set(self.participant_index)

    def record_pred_before_attack(self):
        """
        Save prediction 
        """
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = torch.nn.functional.softmax(self.model(batch_x))
        # out = self.model(batch_x)
        self.pred_history[0].append(torch.max(out, 1).indices)
        self.confidence_history[0].append(out)

    def record_pred_after_attack(self):
        """
        Save prediction 
        """
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = torch.nn.functional.softmax(self.model(batch_x))
        # out = self.model(batch_x)
        self.pred_history[1].append(torch.max(out, 1).indices)
        self.confidence_history[1].append(out)

    # def drop(self,b = 0,step = 0.5):
    #     deviation = np.std(all_updates, 0)  # std
    #     # deviation = np.mean(all_updates, 0)
    #
    #     if pred[i] == batch_y[i]:
    #         b = b + step
    #     else:
    #         if b > 0:
    #             b = b - step
    #
    #     step = step / 2
    #     lamda = lamda + b
    #
    #     return lamda * deviation,

    def optimized_gradient_ascent(self, i):
        """
        Perform one round of gradient ascent attack using the given attack samples
        """
        cache = self.get_flatten_parameters()
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        deviation = torch.std(self.get_flatten_parameters(), 0)
        print("b", self.b)
        print("step", self.step)
        if prediction[i] == batch_y[i]:
            print(prediction[i], batch_y[i])
            # print(prediction[len(self.attack_samples[i])],batch_y[len(self.attack_samples[i])])
            self.b = self.b + self.step
            self.step = self.step
        else:
            if self.b > 0:
                self.b = self.b - self.step
                self.step = self.step / 2

        self.lamda = self.lamda + self.b
        print(self.lamda, "lambda")
        gradient = - (gradient - self.lamda * deviation)
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)

        return gradient

    def gradient_ascent(self, ascent_factor=ASCENT_FACTOR, agr=False, attack=False):
        """
        Perform one round of gradient ascent attack using the given attack samples
        """
        cache = self.get_flatten_parameters()
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        if attack:
            gradient = - ascent_factor * gradient
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if agr:
            self.aggregator.agr_loss_gradient_collect(gradient, indices)
        else:
            self.aggregator.collect(gradient, indices)
        return gradient

    def partial_gradient_ascent(self, ascent_factor=ASCENT_FACTOR, agr=False, ascent_fraction=0.5):
        """
        Perform gradient ascent on only a subset of the attack samples
        """
        cache = self.get_flatten_parameters()
        rand_perm = torch.randperm(len(self.attack_samples))
        ascent_count = round(len(self.attack_samples) * ascent_fraction)
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples[rand_perm[:ascent_count]])
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient = - ascent_factor * gradient
        to_load = cache + gradient
        self.load_parameters(to_load)
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples[rand_perm[ascent_count:]])
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if agr:
            self.aggregator.agr_loss_gradient_collect(gradient, indices)
        else:
            self.aggregator.collect(gradient, indices)

    def in_participant(self, dataindex):
        result = "No"
        for i in range(NUMBER_OF_PARTICIPANTS):
            if dataindex in self.data_reader.get_train_set(i):
                result = i
        return result

    def get_true_member(self):
        true_member = {}
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)

        for i in range(len(self.attack_samples)):
            for j in range(NUMBER_OF_PARTICIPANTS):
                if self.attack_samples[i] in self.data_reader.get_train_set(j):
                    true_member[i] = [batch_y[i],j]
                    break

        return true_member

    def find_member(self):
        pred_member = {}
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        for i in range(len(self.attack_samples)):
            if self.pred_history[0][-1][i] == batch_y[i]:
                pred_member[i] = [batch_y[i]]
                # truth.append(batch_y[i])
        true_member_set = {}
        for i in range(NUMBER_OF_PARTICIPANTS):
            true_member_set[i] = self.data_reader.get_train_set(i)
            # truth_member = self.data_reader.get_train_set(i)

        for member in pred_member.keys():
            for i in true_member_set.keys():
                if member in true_member_set[i]:
                    pred_member[member].append(i)
                    break
            if len(pred_member[member]) != 2:
                pred_member[member].append("No")

        return pred_member

    def get_pred_member(self):
        pred_member = {}
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)

        for i in range(len(self.attack_samples)):
            if self.pred_history[0][TRAIN_EPOCH][i] == batch_y[i]:
                for j in range(NUMBER_OF_PARTICIPANTS):
                    if self.attack_samples[i] in self.data_reader.get_train_set(j):
                        pred_member[i] = [batch_y[i],j]
                        break
                    pred_member[i] = [batch_y[i],"No"]

        return pred_member

    def target_member(self, pred_member, target_counter, epochs_in_round, rounds):
        # pred_member = self.find_member()
        #pred_member = self.get_pred_member()
        true_target = 0
        true_member = 0
        #total_member = len(self.get_true_member())
        #total_sample = len(self.attack_samples)
        total_pred_member = len(pred_member)
        for i in pred_member.keys():
            if pred_member[i][1] == "No":
                continue
            truth_label = pred_member[i][0]
            if i not in target_counter:
                target_counter[i] = {}
                target_counter[i][0] = 0
                target_counter[i][1] = 0
                target_counter[i][2] = 0
                target_counter[i][3] = 0
                target_counter[i][4] = 0
            #max_confi_participant = {}  # epoch that has max confidence in a round
            #max_confi_participant[0] = 0
            #max_confi_participant[1] = 0
            #max_confi_participant[2] = 0
            #max_confi_participant[3] = 0
            #max_confi_participant[4] = 0
            #attack_pred = None  # save the attacked prediction from last round
            #attack_confi = None
            #total_attack_epoch = MAX_EPOCH - TRAIN_EPOCH
            #epochs_in_round = (NUMBER_OF_PARTICIPANTS * 2)
            #total_round = total_attack_epoch // epochs_in_round
            # each round: a 0 a 1 a 2 a 3 a 4
            #for round in range(rounds):
            start_epoch = rounds * epochs_in_round + TRAIN_EPOCH +1  # add offset
            end_epoch = (rounds + 1) * epochs_in_round + TRAIN_EPOCH +2
            round_prediction_history = self.pred_history[0][start_epoch:end_epoch]
            round_confidence_history = self.confidence_history[0][start_epoch:end_epoch]
            for epoch_index in range(epochs_in_round):
                if epoch_index % 2 == 0:
                    continue
                current_attendee = (epoch_index - 1) // 2
                last_prediction = round_prediction_history[epoch_index - 1][i]
                last_confidence = round_confidence_history[epoch_index - 1][i][last_prediction]
                next_prediction = round_prediction_history[epoch_index + 1][i]
                next_confidence = round_confidence_history[epoch_index + 1][i][last_prediction]
                current_prediction = round_prediction_history[epoch_index][i]
                current_confidence = round_confidence_history[epoch_index][i][current_prediction]
                last_truth_label_confidence = round_confidence_history[epoch_index - 1][i][truth_label]
                current_truth_label_confidence = round_confidence_history[epoch_index][i][truth_label]
                next_truth_label_confidence = round_confidence_history[epoch_index + 1][i][truth_label]
                if current_prediction == truth_label:
                    #max_confi_participant[current_attendee] += 2
                    target_counter[i][current_attendee] += 2
                if last_prediction!=current_prediction:
                    #max_confi_participant[current_attendee] += 1
                    target_counter[i][current_attendee] += 1
       
                #if current_truth_label_confidence > last_truth_label_confidence and current_prediction == truth_label:
                #    max_confi_participant[current_attendee] += 3
                #if (current_prediction == truth_label and last_confidence < current_confidence):
                #    max_confi_participant[current_attendee] += 2
                #if (current_prediction != truth_label and last_confidence > current_confidence):
                #    max_confi_participant[current_attendee] += 1
                """
                if (current_prediction == truth_label and last_confidence < current_confidence) or \
                    (current_prediction != truth_label and last_confidence > current_confidence):
                    max_confi_participant[current_attendee] += 1
                """
            pred_participant = max(target_counter[i], key=lambda x: target_counter[i][x])
            if pred_participant == pred_member[i][1]:
                true_target += 1
            true_member += 1
            #print("SampleIndex={} Class={} InParticipant={} Predict={} TargetCount={} ".format(i, truth_label, pred_member[i][1],
            #                                                                          pred_participant,
            #                                                                          max_confi_participant))

        #print("UsedRounds = {} TruePredict={} TrueMember={} TotalPredMemebr={}"\
        #    .format(rounds+1, true_target, true_member, total_pred_member))
        return (true_target, true_member, total_pred_member)

    def target_member_once_per_round(self):
        # pred_member = self.find_member()
        pred_member = self.get_pred_member()
        true_target = 0
        true_member = 0
        total_member = len(self.get_true_member())
        total_sample = len(self.attack_samples)
        total_pred_member = len(pred_member)
        for i in pred_member.keys():
            if pred_member[i][1] == "No":
                continue
            truth_label = pred_member[i][0]
            max_confi_participant = {}  # epoch that has max confidence in a round
            max_confi_participant[0] = 0
            max_confi_participant[1] = 0
            max_confi_participant[2] = 0
            max_confi_participant[3] = 0
            max_confi_participant[4] = 0
            attack_pred = None  # save the attacked prediction from last round
            attack_confi = None
            total_attack_epoch = MAX_EPOCH - TRAIN_EPOCH
            epochs_in_round = (NUMBER_OF_PARTICIPANTS * 2)
            total_round = total_attack_epoch // epochs_in_round
            # each round: a 0 a 1 a 2 a 3 a 4
            for round in range(total_round):
                start_epoch = round * epochs_in_round + TRAIN_EPOCH +1  # add offset
                end_epoch = (round + 1) * epochs_in_round + TRAIN_EPOCH +1
                round_prediction_history = self.pred_history[0][start_epoch:end_epoch]
                round_confidence_history = self.confidence_history[0][start_epoch:end_epoch]
                for epoch_index in range(epochs_in_round):
                    if epoch_index % 2 == 0:
                        continue
                    current_attendee = (epoch_index - 1) // 2
                    last_prediction = round_prediction_history[epoch_index - 1][i]
                    last_confidence = round_confidence_history[epoch_index - 1][i][last_prediction]
                    current_prediction = round_prediction_history[epoch_index][i]
                    current_confidence = round_confidence_history[epoch_index][i][current_prediction]
                    last_truth_label_confidence = round_confidence_history[epoch_index - 1][i][truth_label]
                    current_truth_label_confidence = round_confidence_history[epoch_index][i][truth_label]
                    if current_prediction == truth_label and last_prediction!=current_prediction:
                        max_confi_participant[current_attendee] += 1




                    #if current_truth_label_confidence > last_truth_label_confidence and current_prediction == truth_label:
                    #    max_confi_participant[current_attendee] += 3
                    #if (current_prediction == truth_label and last_confidence < current_confidence):
                    #    max_confi_participant[current_attendee] += 2
                    #if (current_prediction != truth_label and last_confidence > current_confidence):
                    #    max_confi_participant[current_attendee] += 1
                    """
                    if (current_prediction == truth_label and last_confidence < current_confidence) or \
                        (current_prediction != truth_label and last_confidence > current_confidence):
                        max_confi_participant[current_attendee] += 1
                    """
            pred_participant = max(max_confi_participant, key=lambda x: max_confi_participant[x])
            if pred_participant == pred_member[i][1]:
                true_target += 1
            true_member += 1
            print("SampleIndex={} Class={} InParticipant={} Predict={} TargetCount={} ".format(i, truth_label, pred_member[i][1],
                                                                                      pred_participant,
                                                                                      max_confi_participant))

        print("TruePredict={} TrueMember={} TotalPredMemebr={} TotalMember={} TotalSample={}"\
            .format(true_target, true_member, total_pred_member, total_member, total_sample))
        return None

    def target_member_once_absent(self):
        # pred_member = self.find_member()
        pred_member = self.get_pred_member()
        true_target = 0
        true_member = 0
        total_member = len(self.get_true_member())
        total_sample = len(self.attack_samples)
        total_pred_member = len(pred_member)
        for i in pred_member.keys():
            if pred_member[i][1] == "No":
                continue
            truth_label = pred_member[i][0]
            max_confi_participant = {}  # epoch that has max confidence in a round
            max_confi_participant[0] = 0
            max_confi_participant[1] = 0
            max_confi_participant[2] = 0
            max_confi_participant[3] = 0
            max_confi_participant[4] = 0
            attack_pred = None  # save the attacked prediction from last round
            attack_confi = None
            total_attack_epoch = MAX_EPOCH - TRAIN_EPOCH
            epochs_in_round = NUMBER_OF_PARTICIPANTS
            total_round = total_attack_epoch // epochs_in_round
            # each round: a1234 a0234 a0134 a0124 a0123
            for round in range(total_round):
                start_epoch = round * epochs_in_round + TRAIN_EPOCH +1  # add offset
                end_epoch = (round + 1) * epochs_in_round + TRAIN_EPOCH +1
                round_prediction_history = self.pred_history[0][start_epoch:end_epoch]
                round_confidence_history = self.confidence_history[0][start_epoch:end_epoch]
                for epoch_index in range(epochs_in_round):
                    
                    current_absent = epoch_index
                    last_prediction = round_prediction_history[epoch_index - 1][i]
                    last_confidence = round_confidence_history[epoch_index - 1][i][last_prediction]
                    current_prediction = round_prediction_history[epoch_index][i]
                    current_confidence = round_confidence_history[epoch_index][i][current_prediction]
                    last_truth_label_confidence = round_confidence_history[epoch_index - 1][i][truth_label]
                    current_truth_label_confidence = round_confidence_history[epoch_index][i][truth_label]
                    if current_prediction != truth_label and last_prediction==current_prediction:
                        max_confi_participant[current_absent] += 1




                    #if current_truth_label_confidence > last_truth_label_confidence and current_prediction == truth_label:
                    #    max_confi_participant[current_attendee] += 3
                    #if (current_prediction == truth_label and last_confidence < current_confidence):
                    #    max_confi_participant[current_attendee] += 2
                    #if (current_prediction != truth_label and last_confidence > current_confidence):
                    #    max_confi_participant[current_attendee] += 1
                    """
                    if (current_prediction == truth_label and last_confidence < current_confidence) or \
                        (current_prediction != truth_label and last_confidence > current_confidence):
                        max_confi_participant[current_attendee] += 1
                    """
            pred_participant = max(max_confi_participant, key=lambda x: max_confi_participant[x])
            if pred_participant == pred_member[i][1]:
                true_target += 1
            true_member += 1
            print("SampleIndex={} Class={} InParticipant={} Predict={} TargetCount={} ".format(i, truth_label, pred_member[i][1],
                                                                                      pred_participant,
                                                                                      max_confi_participant))

        print("TruePredict={} TrueMember={} TotalPredMemebr={} TotalMember={} TotalSample={}"\
            .format(true_target, true_member, total_pred_member, total_member, total_sample))
        return None


    def evaluate_attack_result(self):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        # print(batch_x)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member

    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)


class WhiteBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to collect data for a white-box membership inference attack
    #TODO
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        super(WhiteBoxMalicious, self).__init__(reader, aggregator, 0)
        self.members = None
        self.non_members = None
        self.all_samples = self.get_attack_sample()
        self.attack_samples = self.all_samples
        self.descending_samples = None
        self.shuffled_labels = {}
        self.shuffle_labels()
        self.global_gradient = torch.zeros(self.get_flatten_parameters().size())
        self.last_round_shared_grad = None
        self.pred_history = []
        self.pred_history.append([])
        self.pred_history.append([])
        self.pred_history_new = {}
        self.confidence_history = []
        self.confidence_history.append([])
        self.confidence_history.append([])
        self.member_prediction = None
        self.member_intersections = {}

    def prune_data(self, label=KEEP_CLASS):
        if label == None:
            return None
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        index = []
        for i in range(len(batch_y)):
            if batch_y[i] == label:
                #self.attack_samples=torch.cat([self.attack_samples[0:i],self.attack_samples[i+1:]])
                index.append(i)
                
        self.attack_samples = self.attack_samples[index].to(DEVICE)

    def record_pred_before_attack(self):
        """
        Save prediction 
        """
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = torch.nn.functional.softmax(self.model(batch_x))
        # out = self.model(batch_x)
        self.pred_history[0].append(torch.max(out, 1).indices)
        self.confidence_history[0].append(out)

    def record_pred_after_attack(self):
        """
        Save prediction 
        """
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = torch.nn.functional.softmax(self.model(batch_x))
        # out = self.model(batch_x)
        self.pred_history[1].append(torch.max(out, 1).indices)
        self.confidence_history[1].append(out)

    def get_attack_sample(self, attack_samples=NUMBER_OF_ATTACK_SAMPLES, member_rate=BLACK_BOX_MEMBER_RATE):
        """
        Randomly select a sample from the data set
        :return: shuffled data of attacker samples
        """
        member_count = round(attack_samples * member_rate)
        non_member_count = attack_samples - member_count
        self.members = self.data_reader.train_set.flatten()[
            torch.randperm(len(self.data_reader.train_set.flatten()))[:member_count]]
        self.non_members = self.data_reader.test_set.flatten()[
            torch.randperm(len(self.data_reader.test_set.flatten()))[:non_member_count]]
        return torch.cat([self.members, self.non_members])[torch.randperm(attack_samples)]

    def shuffle_labels(self, iteration=WHITE_BOX_SHUFFLE_COPIES):
        """
        Shuffle the labels in several random permutation, to be used as misleading labels
        it will repeat the given iteration times denote as k, k different copies will be saved
        """
        max_label = torch.max(self.data_reader.labels).item()
        for i in range(iteration):
            shuffled = self.data_reader.labels[torch.randperm(len(self.data_reader.labels))]
            for j in torch.nonzero(shuffled == self.data_reader.labels):
                shuffled[j] = (shuffled[j] + torch.randint(max_label, [1]).item()) % max_label
            self.shuffled_labels[i] = shuffled

    def target_member(self, pred_member, target_counter, epochs_in_round, rounds):
        # pred_member = self.find_member()
        #pred_member = self.get_pred_member()
        true_target = 0
        true_member = 0
        #total_member = len(self.get_true_member())
        #total_sample = len(self.attack_samples)
        total_pred_member = len(pred_member)
        for i in pred_member.keys():
            if pred_member[i][1] == "No":
                continue
            truth_label = pred_member[i][0]
            if i not in target_counter:
                target_counter[i] = {}
                target_counter[i][0] = 0
                target_counter[i][1] = 0
                target_counter[i][2] = 0
                target_counter[i][3] = 0
                target_counter[i][4] = 0
            #max_confi_participant = {}  # epoch that has max confidence in a round
            #max_confi_participant[0] = 0
            #max_confi_participant[1] = 0
            #max_confi_participant[2] = 0
            #max_confi_participant[3] = 0
            #max_confi_participant[4] = 0
            #attack_pred = None  # save the attacked prediction from last round
            #attack_confi = None
            #total_attack_epoch = MAX_EPOCH - TRAIN_EPOCH
            #epochs_in_round = (NUMBER_OF_PARTICIPANTS * 2)
            #total_round = total_attack_epoch // epochs_in_round
            # each round: a 0 a 1 a 2 a 3 a 4
            #for round in range(rounds):
            start_epoch = rounds * epochs_in_round + TRAIN_EPOCH +1  # add offset
            end_epoch = (rounds + 1) * epochs_in_round + TRAIN_EPOCH +2
            round_prediction_history = self.pred_history[0][start_epoch:end_epoch]
            round_confidence_history = self.confidence_history[0][start_epoch:end_epoch]
            for epoch_index in range(epochs_in_round):
                if epoch_index % 2 == 0:
                    continue
                current_attendee = (epoch_index - 1) // 2
                last_prediction = round_prediction_history[epoch_index - 1][i]
                last_confidence = round_confidence_history[epoch_index - 1][i][last_prediction]
                next_prediction = round_prediction_history[epoch_index + 1][i]
                next_confidence = round_confidence_history[epoch_index + 1][i][last_prediction]
                current_prediction = round_prediction_history[epoch_index][i]
                current_confidence = round_confidence_history[epoch_index][i][current_prediction]
                last_truth_label_confidence = round_confidence_history[epoch_index - 1][i][truth_label]
                current_truth_label_confidence = round_confidence_history[epoch_index][i][truth_label]
                next_truth_label_confidence = round_confidence_history[epoch_index + 1][i][truth_label]
                if current_prediction == truth_label:
                    #max_confi_participant[current_attendee] += 2
                    target_counter[i][current_attendee] += 2
                if last_prediction!=current_prediction:
                    #max_confi_participant[current_attendee] += 1
                    target_counter[i][current_attendee] += 1
       
                if current_truth_label_confidence > last_truth_label_confidence and current_prediction == truth_label:
                    target_counter[i][current_attendee] += 3
                if (current_prediction == truth_label and last_confidence < current_confidence):
                    target_counter[i][current_attendee] += 2
                if (current_prediction != truth_label and last_confidence > current_confidence):
                    target_counter[i][current_attendee] += 1
                """
                if (current_prediction == truth_label and last_confidence < current_confidence) or \
                    (current_prediction != truth_label and last_confidence > current_confidence):
                    max_confi_participant[current_attendee] += 1
                """
            pred_participant = max(target_counter[i], key=lambda x: target_counter[i][x])
            if pred_participant == pred_member[i][1]:
                true_target += 1
            true_member += 1
            #print("SampleIndex={} Class={} InParticipant={} Predict={} TargetCount={} ".format(i, truth_label, pred_member[i][1],
            #                                                                          pred_participant,
            #                                                                          max_confi_participant))

        print("UsedRounds = {} TruePredict={} TrueMember={} TotalPredMemebr={}"\
            .format(rounds+1, true_target, true_member, total_pred_member))
        return (rounds+1, true_target, true_member, total_pred_member)

    def gradient_ascent(self, ascent_factor=ASCENT_FACTOR, batch_size=BATCH_SIZE,
                        adaptive_factor=FRACTION_OF_ASCENDING_SAMPLES, mislead=False, mislead_factor=1):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :return: gradient generated
        """
        cache = self.get_flatten_parameters()
        threshold = round(len(self.all_samples) * adaptive_factor)
        ascending_samples = self.all_samples[:threshold]
        self.attack_samples = ascending_samples
        descending_samples = self.all_samples[threshold:]
        self.descending_samples = descending_samples
        # Perform gradient ascent for ascending samples
        i = 0
        while i * batch_size < len(ascending_samples):
            batch_index = ascending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        asc_gradient = self.get_flatten_parameters() - cache
        self.load_parameters(cache)
        # Perform gradient descent for the rest of samples
        i = 0
        while i * batch_size < len(descending_samples):
            batch_index = descending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        desc_gradient = self.get_flatten_parameters() - cache
        if not mislead:
            return desc_gradient - asc_gradient * ascent_factor

        # mislead labels
        self.load_parameters(cache)
        mislead_gradients = []
        for k in range(len(self.shuffled_labels)):
            i = 0
            while i * batch_size < len(ascending_samples):
                batch_index = ascending_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                y = self.shuffled_labels[k][batch_index]
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            mislead_gradients.append(self.get_flatten_parameters() - cache)
            self.load_parameters(cache)

        # select the best misleading gradient
        selected_k = 0
        largest_gradient_diff = 0
        for k in range(len(mislead_gradients)):
            diff = (mislead_gradients[k] - asc_gradient).norm()
            if diff > largest_gradient_diff:
                largest_gradient_diff = diff
                selected_k = k
        return desc_gradient - asc_gradient * ascent_factor + mislead_factor * mislead_gradients[selected_k]
    

    def train(self, attack=False, mislead=True,
              ascent_factor=ASCENT_FACTOR, ascent_fraction=FRACTION_OF_ASCENDING_SAMPLES, white_box_optimize=False):
        """
        Start a white-box training
        """
        gradient = self.gradient_ascent(ascent_factor=ascent_factor, adaptive_factor=ascent_fraction, mislead=mislead)
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if attack:
            random_key = torch.randint(1, 10, [1]).item()
            if self.global_gradient is not None and white_box_optimize and random_key <8:
                norm = self.global_gradient.norm()
                gradient += self.global_gradient
                gradient = gradient * norm / gradient.norm()
            self.last_round_shared_grad = gradient
            self.aggregator.collect(gradient, indices)
        return gradient

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Save the parameters from last round before collect new parameters
        """
        cache = self.get_flatten_parameters()
        super(WhiteBoxMalicious, self).collect_parameters(parameter)
        self.global_gradient = self.get_flatten_parameters() - cache

    def examine_sample_gradients(self, monitor_window=512):
        """
        Examine the gradients for each sample, used to compare
        """
        cache = self.get_flatten_parameters()
        monitor = {}
        for i in range(len(self.attack_samples)):
            print("\r Evaluating monitor window for attack sample {}/{}".format(i, len(self.attack_samples)), end="")
            sample = self.attack_samples[i]
            x = self.data_reader.data[sample]
            y = self.data_reader.labels[sample]
            x = torch.vstack([x] * BATCH_SIZE)
            y = torch.hstack([y] * BATCH_SIZE)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            gradient = self.get_flatten_parameters() - cache
            sorted = torch.sort(gradient, descending=True)
            monitor[i] = sorted.indices[:monitor_window], \
                         sorted.values[:monitor_window]
            self.load_parameters(cache)
        print(" Monitor window generated.")
        return monitor

    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def optimized_evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        selected_participants = RobustMechanism.appearence_list
        union = []
        for i in selected_participants:
            if i !=5:
                union.append(self.member_intersections[i])
            else:
                pass

        effective_members = torch.unique(torch.cat(union,0).cpu())
        if len(effective_members) > 0:
            batch_x, batch_y = self.data_reader.get_batch(effective_members)
            with torch.no_grad():
                out = self.model(batch_x).to(DEVICE)
            prediction = torch.max(out, 1).indices
            accurate = (prediction == batch_y).sum()
            return accurate / len(batch_y)
        return 0

    def optimized_evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        selected_participants = RobustMechanism.appearence_list
        to_union = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            if i not in selected_participants:
                if i !=5:
                    to_union.append(i)
                else:
                    pass
        union = [self.non_members.cpu()]
        for i in to_union:
            union.append(self.member_intersections[i])


        effective_non_members = torch.unique(torch.cat(union,0).cpu())
        if len(effective_non_members) > 0:
            batch_x, batch_y = self.data_reader.get_batch(effective_non_members)
            with torch.no_grad():
                out = self.model(batch_x)
            prediction = torch.max(out, 1).indices.to(DEVICE)
            accurate = (prediction == batch_y).sum()
            return accurate / len(batch_y)
        return 0

    def optimized_evaluation_init(self):
        """
        Calculate the intersection of self.members and the train set of each participant
        """
        for i in range(NUMBER_OF_PARTICIPANTS):
            self.member_intersections[i] = \
                torch.tensor(np.intersect1d(self.data_reader.get_train_set(i).detach().cpu().numpy(), self.attack_samples.detach().cpu().numpy()))



    def get_pred_member(self, rounds):
        pred_member = {}
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        for i in range(len(self.member_prediction)):
            if self.pred_history_new[rounds][i] == 0:
                continue
            for j in range(NUMBER_OF_PARTICIPANTS):
                if self.attack_samples[i] in self.data_reader.get_train_set(j):
                    pred_member[i] = [batch_y[i], j]
                    break
                pred_member[i] = [batch_y[i], "No"]
        return pred_member

    def partial_gradient_ascent(self, ascent_factor=ASCENT_FACTOR, agr=False, ascent_fraction=0.5):
        """
        Perform gradient ascent on only a subset of the attack samples
        """
        cache = self.get_flatten_parameters()
        rand_perm = torch.randperm(len(self.attack_samples))
        ascent_count = round(len(self.attack_samples) * ascent_fraction)
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples[rand_perm[:ascent_count]])
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient = - ascent_factor * gradient
        to_load = cache + gradient
        self.load_parameters(to_load)
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples[rand_perm[ascent_count:]])
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if agr:
            self.aggregator.agr_loss_gradient_collect(gradient, indices)
        else:
            self.aggregator.collect(gradient, indices)

    def evaluate_attack_result(self, adaptive_prediction=True, adaptive_strategy=SCORE_BASED_STRATEGY, rounds=None):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        confidence = torch.max(out, 1).values
        accuracy = ((prediction == batch_y).sum() / len(self.attack_samples))
        low_confidence_samples = torch.sort(confidence).indices[:torch.round(len(self.attack_samples) * (1 - accuracy)).int()]
        classification_counter = 0
        if adaptive_prediction and self.last_round_shared_grad is not None and self.global_gradient.norm() != 0:
            monitor = self.examine_sample_gradients()
            if adaptive_strategy == NORM_BASED_STRATEGY:
                norm_diff = self.get_classification_norm_based(monitor)
        for i in range(len(self.attack_samples)):
            if adaptive_prediction and self.last_round_shared_grad is not None and self.global_gradient.norm() != 0:
                if adaptive_strategy == SCORE_BASED_STRATEGY:
                    membership = self.get_classification_score_based(monitor[i][0])
                elif adaptive_strategy == NORM_BASED_STRATEGY:
                    membership = i not in norm_diff[:torch.round(len(self.attack_samples) * (1 - accuracy))]
                if i in low_confidence_samples and prediction[i] == batch_y[i]:
                    classification_counter += 1
                    if membership:
                        attack_result.append(1)
                    else:
                        attack_result.append(0)
                else:
                    if prediction[i] == batch_y[i]:
                        attack_result.append(1)
                    else:
                        attack_result.append(0)
            else:
                if prediction[i] == batch_y[i]:
                    attack_result.append(1)
                else:
                    attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)
            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1
        if rounds is not None:
            self.pred_history_new[rounds] = attack_result
        print("overall {} classified attacks".format(classification_counter))
        self.member_prediction = attack_result
        return true_member, false_member, true_non_member, false_non_member

    def evaluate_label_attack_result(self, base_pred=TRAIN_EPOCH):
        """
        Evaluate the attack result, return the overall accuracy, and precise
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            # Defalut: use the pred before attack as base

            if prediction[i] == self.pred_history[0][base_pred][i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member

    def in_round(self, dataindex):
        result = False
        for j in self.aggregator.robust.appearence_list:
            if self.attack_samples[dataindex] in self.data_reader.get_train_set(j) and j!=5:
                result = True
                return result
            else:
                pass
        return result

    def evaluate_optimized_attack_result(self,adaptive_prediction=True, adaptive_strategy=SCORE_BASED_STRATEGY):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        """
        print(self.aggregator.robust.appearence_list)
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        attack_sample = []
        # for i in [0,1,2,3]:
        #     batch_x1,batch_y = self.data_reader
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)

        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        confidence = torch.max(out, 1).values
        accuracy = ((prediction == batch_y).sum() / len(self.attack_samples))
        low_confidence_samples = torch.sort(confidence).indices[:torch.round(len(self.attack_samples) * (1 - accuracy)).int()]
        classification_counter = 0
        if adaptive_prediction and self.last_round_shared_grad is not None and self.global_gradient.norm() != 0:
            monitor = self.examine_sample_gradients()
            if adaptive_strategy == NORM_BASED_STRATEGY:
                norm_diff = self.get_classification_norm_based(monitor)
        for i in range(len(self.attack_samples)):
            # if self.in_round(i):
            if prediction[i] == batch_y[i]:
                # print(prediction[i])
                attack_result.append(1)
                # print(attack_result)
            elif prediction[i] != batch_y[i]:
                attack_result.append(0)



                # print(attack_result)


            # print(ground_truth)

                # else:
                #     ground_truth.append(0)

            if self.attack_samples[i] in self.data_reader.train_set and self.in_round(i):
                ground_truth.append(1)
            elif self.attack_samples[i] in self.data_reader.train_set and self.in_round(i) == False:
                ground_truth.append(0)
            else:
                ground_truth.append(3)

            # print(attack_result)
            # print(ground_truth)
            # print(i)
            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 1):
                false_non_member += 1
            else:
                pass
        print("overall {} classified attacks".format(classification_counter))
        return true_member, false_member, true_non_member, false_non_member

    def get_classification_score_based(self, monitor_window):
        """
        Get the prediction outcome of a given monitor window using the membership score
        """
        # strong_attack = (self.last_round_shared_grad < 0).sum() > (len(monitor_window)*accuracy)
        member_score = torch.logical_and((self.last_round_shared_grad[monitor_window] < 0),
                                         (self.global_gradient[monitor_window] >= 0)).sum()
        non_member_score = torch.logical_and((self.last_round_shared_grad[monitor_window] < 0),
                                             (self.global_gradient[monitor_window] < 0)).sum()
        # print("member score = {}, non_member score = {}".format(member_score, non_member_score))
        return member_score >= non_member_score

    def get_classification_norm_based(self, monitor: dict):
        """
        Get the prediction outcome of a given monitor using norm of the difference
        """
        all_diff = torch.zeros(self.attack_samples.size())
        for key in monitor.keys():
            index, value = monitor[key]
            norm1 = (value - self.global_gradient[index]).norm()
            norm2 = (value - self.last_round_shared_grad[index]).norm()
            all_diff[key] = norm2 - norm1
        all_diff = torch.sort(all_diff).indices
        return all_diff
