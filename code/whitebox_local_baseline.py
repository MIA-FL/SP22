from organizer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_white_local'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_AGR,
                                                                               TRAIN_EPOCH,
                                                                               MAX_EPOCH - TRAIN_EPOCH))
org = Organizer()
org.set_random_seed()
org.federated_training_nontarget_white_box(logger, adaptive=True)