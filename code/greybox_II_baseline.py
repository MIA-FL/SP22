from organizer import *
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY, 'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_targetd_greybox'.format(TIME_STAMP,DEFAULT_SET,DEFAULT_AGR,TRAIN_EPOCH,MAX_EPOCH-TRAIN_EPOCH))
org = Organizer()
print("TorchVersion: {}  CUDAavailable: {}".format(torch.__version__,torch.cuda.is_available()))
org.set_random_seed()
org.federated_training_targeted_grey_box(logger)