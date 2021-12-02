import pandas as pd
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from constants import *


class DataReader:
    """
    The class to read data set from the given file
    """
    def __init__(self, data_set=DEFAULT_SET, label_column=LABEL_COL, batch_size=BATCH_SIZE, distribution=DEFAULT_DISTRIBUTION):
        """
        Load the data from the given data path
        :param path: the path of csv file to load data
        :param label_column: the column index of csv file to store the labels
        :param label_size: The number of overall classes in the given data set
        """
        # load the csv file
        if data_set == PURCHASE100:
            path = PURCHASE100_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)
        elif data_set == CIFAR_10:
            samples = np.vstack(
                [np.genfromtxt(CIFAR_10_PATH+"train{}.csv".format(x), delimiter=',') for x in range(4)]
            )
            self.data = torch.tensor(samples[:, :-1], dtype=torch.float).to(DEVICE)
            self.labels = torch.tensor(samples[:, -1], dtype=torch.int64).to(DEVICE)
        elif data_set == LOCATION30:
            path = LOCATION30_PATH
            data_frame = pd.read_csv(path, header=None)
            # extract the label
            self.labels = torch.tensor(data_frame[label_column].to_numpy(), dtype=torch.int64).to(DEVICE)
            self.labels -= 1
            data_frame.drop(label_column, inplace=True, axis=1)
            # extract the data
            self.data = torch.tensor(data_frame.to_numpy(), dtype=torch.float).to(DEVICE)
        elif data_set == TEXAS100:
            path = TEXAS100_PATH
            self.data = np.load(path)
            self.labels = self.data['labels']
            self.data = self.data['features']
            self.labels = np.argmax(self.labels, axis=1)
            self.labels = torch.tensor(self.labels, dtype=torch.int64).to(DEVICE)
            self.data = torch.tensor(self.data, dtype=torch.float).to(DEVICE)
        elif data_set == MNIST:
            # Normalize input
            MNIST_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.5, ],
                    std=[0.5, ])
            ])
            mnist = tv.datasets.MNIST(MNIST_PATH, transform=MNIST_transform, download=True)
            loader = DataLoader(mnist, batch_size=len(mnist))
            self.data = next(iter(loader))[0]
            self.data = torch.flatten(self.data, 1)
            self.labels = next(iter(loader))[1]
            self.data = self.data.float()
            self.labels = self.labels.long()
        elif data_set == GNOME:
            loaded = np.load(GNOME_PATH)
            self.data = torch.tensor(loaded['features'], dtype=torch.float).to(DEVICE)
            self.labels = torch.tensor(loaded['labels'], dtype=torch.int64).to(DEVICE)

        self.data = self.data.to(DEVICE)
        self.labels = self.labels.to(DEVICE)


        # initialize the training and testing batches indices
        self.train_set = None
        self.test_set = None
        overall_size = self.labels.size(0)
        if distribution is None:
            # divide data samples into batches, drop the last bit of data samples to make sure each batch is full sized
            overall_size -= overall_size % batch_size
            rand_perm = torch.randperm(self.labels.size(0)).to(DEVICE)
            rand_perm = rand_perm[:overall_size].to(DEVICE)
            self.batch_indices = rand_perm.reshape((-1, batch_size)).to(DEVICE)
            self.train_test_split()
        elif distribution == CLASS_BASED:
            self.train_test_split(batch_training=False)
            print("data split, train set length={}, test set length={}".format(len(self.train_set), len(self.test_set)))
            self.class_indices = {}
            self.class_training = {}
            self.class_testing = {}
            for i in range(torch.max(self.labels).item()+1):
                self.class_indices[i] = torch.nonzero((self.labels == i)).to(DEVICE)
                self.class_training[i] = torch.tensor(np.intersect1d(self.class_indices[i].cpu(), self.train_set.cpu())).to(DEVICE)
                self.class_testing[i] = torch.tensor(np.intersect1d(self.class_indices[i].cpu(), self.test_set.cpu())).to(DEVICE)
                print("Label {}, overall samples ={}, train_set={}, test_set={}".format(i, len(self.class_indices[i]), len(self.class_training[i]), len(self.class_testing[i])))

        # print(self.batch_indices.size())

        print("Data set "+DEFAULT_SET+
              " has been loaded, overall {} records, batch size = {}, testing batches: {}, training batches: {}"
              .format(overall_size, batch_size, self.test_set.size(0), self.train_set.size(0)))

    def train_test_split(self, ratio=TRAIN_TEST_RATIO, batch_training=BATCH_TRAINING):
        """
        Split the data set into training set and test set according to the given ratio
        :param ratio: tuple (float, float) the ratio of train set and test set
        :param batch_training: True to train by batch, False will not
        :return: None
        """
        if batch_training:
            train_count = round(self.batch_indices.size(0) * ratio[0] / sum(ratio))
            self.train_set = self.batch_indices[:train_count].to(DEVICE)
            self.test_set = self.batch_indices[train_count:].to(DEVICE)
        else:
            train_count = round(self.data.size(0) * ratio[0] / sum(ratio))
            rand_perm = torch.randperm(self.data.size(0)).to(DEVICE)
            self.train_set = rand_perm[:train_count].to(DEVICE)
            self.test_set = rand_perm[train_count:].to(DEVICE)

    def get_train_set(self, participant_index=0, distribution=DEFAULT_DISTRIBUTION, by_batch=BATCH_TRAINING, batch_size=BATCH_SIZE):
        """
        Get the indices for each training batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each training batch
        """
        if distribution is None:
            batches_per_participant = self.train_set.size(0) // NUMBER_OF_PARTICIPANTS
            lower_bound = participant_index * batches_per_participant
            upper_bound = (participant_index + 1) * batches_per_participant
            return self.train_set[lower_bound: upper_bound]
        if distribution == CLASS_BASED:
            class_count = torch.max(self.labels).item() + 1
            class_per_participant = class_count // NUMBER_OF_PARTICIPANTS
            my_set = []
            lower_bound = participant_index * class_per_participant
            upper_bound = (participant_index + 1) * class_per_participant
            for i in range(lower_bound, upper_bound):
                my_set.append(self.class_training[i])
            if participant_index == NUMBER_OF_PARTICIPANTS - 1:
                for i in range(upper_bound, class_count):
                    my_set.append(self.class_training[i])
            all_samples = torch.hstack(my_set)
            if by_batch:
                lenth = len(all_samples)
                lenth -= lenth % batch_size
                all_samples = all_samples[:lenth].reshape((-1, batch_size))
            # print("The size of training set for participant {} is {}".format(participant_index, all_samples.size()))
            return all_samples

    def get_test_set(self, participant_index=0, distribution=DEFAULT_DISTRIBUTION, by_batch=BATCH_TRAINING, batch_size=BATCH_SIZE):
        """
        Get the indices for each test batch
        :param participant_index: the index of a particular participant, must be less than the number of participants
        :return: tensor[number_of_batches_allocated, BATCH_SIZE] the indices for each test batch
        """
        if distribution is None:
            batches_per_participant = self.test_set.size(0) // NUMBER_OF_PARTICIPANTS
            lower_bound = participant_index * batches_per_participant
            upper_bound = (participant_index + 1) * batches_per_participant
            return self.test_set[lower_bound: upper_bound]
        elif distribution == CLASS_BASED:
            class_count = torch.max(self.labels).item() + 1
            class_per_participant = class_count // NUMBER_OF_PARTICIPANTS
            my_set = []
            lower_bound = participant_index * class_per_participant
            upper_bound = (participant_index + 1) * class_per_participant
            for i in range(lower_bound, upper_bound):
                my_set.append(self.class_testing[i])
            if participant_index == NUMBER_OF_PARTICIPANTS - 1:
                for i in range(upper_bound, class_count):
                    my_set.append(self.class_testing[i])
            all_samples = torch.hstack(my_set)
            if by_batch:
                lenth = len(all_samples)
                lenth -= lenth % batch_size
                all_samples = all_samples[:lenth].reshape((-1, batch_size))
            # print("The size of testing set for participant {} is {}".format(participant_index, all_samples.size()))
            return all_samples

    def get_batch(self, batch_indices):
        """
        Get the batch of data according to given batch indices
        :param batch_indices: tensor[BATCH_SIZE], the indices of a particular batch
        :return: tuple (tensor, tensor) the tensor representing the data and labels
        """
        return self.data[batch_indices], self.labels[batch_indices]

    def get_samples(self, sample_indices):
        pass

    def get_black_box_batch(self, member_rate=BLACK_BOX_MEMBER_RATE, attack_batch_size=NUMBER_OF_ATTACK_SAMPLES):
        """
        Generate batches for black box training
        :param member_rate The rate of member data samples
        :param attack_batch_size the number of data samples allocated to the black-box attacker
        """
        member_count = round(attack_batch_size * member_rate)
        non_member_count = attack_batch_size - member_count
        train_flatten = self.train_set.flatten().to(DEVICE)
        test_flatten = self.test_set.flatten().to(DEVICE)
        member_indices = train_flatten[torch.randperm(len(train_flatten))[:member_count]].to(DEVICE)
        non_member_indices = test_flatten[torch.randperm((len(test_flatten)))[:non_member_count]].to(DEVICE)
        result = torch.cat([member_indices, non_member_indices]).to(DEVICE)
        result = result[torch.randperm(len(result))].to(DEVICE)
        return result, member_indices, non_member_indices
