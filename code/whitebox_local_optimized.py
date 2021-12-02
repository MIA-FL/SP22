from organizer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_{}_TrainEpoch{}_PrepareEpoch{}_AttackEpoch{}_whitebox_local_optimized'.format(TIME_STAMP, DEFAULT_SET,
                                                                                               DEFAULT_AGR, str(DEFAULT_DISTRIBUTION),
                                                                                               TRAIN_EPOCH,
                                                                                               MAX_EPOCH - TRAIN_EPOCH, WHITE_BOX_GLOBAL_TARGETED_ROUND))
org = Organizer()
org.set_random_seed()
org.optimized_local_nontarget_white_box(logger)