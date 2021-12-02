from models import *
from constants import *
import pandas as pd
import numpy as np
import random

seed = GLOBAL_SEED
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                     'log_{}_{}_{}_TrainEpoch{}_nonTarget{}_targeted{}_global_targeted_round_robbin'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_DISTRIBUTION,
                                                                               TRAIN_EPOCH, MAX_EPOCH - TRAIN_EPOCH, WHITE_BOX_GLOBAL_TARGETED_ROUND))
logger.info("Initializing...")
data_reader = DataReader()
sample_model = TargetModel(data_reader)
aggregator = GlobalAttacker(data_reader, sample_model.get_flatten_parameters())
global_model = FederatedModel(data_reader, aggregator)
global_model.init_global_model()
aggregator.get_shadow_model(global_model)
participants = []
for i in range(NUMBER_OF_PARTICIPANTS):
    participants.append(FederatedModel(data_reader, aggregator))
    participants[i].init_participant(global_model, i)
logger.info("Data distribution is {}".format(DEFAULT_DISTRIBUTION))
logger.info("Dataset is {}".format(DEFAULT_SET))
logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
non_target_recorder = pd.DataFrame(columns=["Epoch", "Attack_acc", "Test_acc", "Member_pred", "Non-member_pred", "True_mem", "False_mem", "True_non", "Fasle_non"])
target_recorder = pd.DataFrame(columns=["Round", "Target_acc", "True_target", "True_member", "Pred_member", "pred_owner0", "pred_owner1", "pred_owner2", "pred_owner3", "pred_owner4"])
logger.info("Start non-targeted attacking...")

for j in range(MAX_EPOCH):
    if j == TRAIN_EPOCH:
        aggregator.perform_attack = True
        logger.info("Start attacking...")
    global_param = global_model.get_flatten_parameters()
    for i in range(NUMBER_OF_PARTICIPANTS):
        participants[i].collect_parameters(global_param)
        participants[i].share_gradient()
    global_model.apply_gradient()
    test_loss, test_acc = global_model.test_outcome()

    # Recording
    true_member, false_member, true_non_member, false_non_member = aggregator.evaluate_attack_result()
    attack_precision = true_member / (true_member + false_member) if (true_member + false_member) > 0 else 0
    attack_accuracy = (true_member + true_non_member) / (
            true_member + true_non_member + false_member + false_non_member)
    attack_recall = true_member / (true_member + false_non_member) if (true_member + false_non_member) > 0 else 0
    pred_acc_member = aggregator.evaluate_member_accuracy().item()
    pred_acc_non_member = aggregator.evaluate_non_member_accuracy().item()
    logger.info("Epoch {} - Attack_acc={:.4f}, Test acc={:.4f}, Precision={:.4f}, Recall={:.4f}, Member_pred={:.4f}, Non-member_pred={:.4f}"
                .format(j+1, attack_accuracy, test_acc, attack_precision, attack_recall, pred_acc_member, pred_acc_non_member))
    non_target_recorder.loc[len(non_target_recorder)] = (j+1, attack_accuracy, test_acc, pred_acc_member, pred_acc_non_member, true_member, false_member, true_non_member, false_non_member)

recorder_suffix = "_global_targeted_round_robbin"
non_target_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + "_" + str(DEFAULT_DISTRIBUTION) + recorder_suffix + "_preparation.csv")

logger.info("Initializing targeted attack...")
aggregator.init_round_robbin()
best_round, best_target_acc, best_true_target = 0, 0, 0
for j in range(WHITE_BOX_GLOBAL_TARGETED_ROUND):
    for i in range(NUMBER_OF_PARTICIPANTS):
        aggregator.round_robbin_isolation(participants[i])
        tm, fm, tn, fn = aggregator.round_robbin_evaluation(participants[i])
        attack_acc = (tm + tn) / (tm + fm + tn + fn)
        logger.info("Round {} participant {}: attack acc={:.4f}, true member={}, false member={}".format(j+1, i, attack_acc, tm, fm))
    tt, owners = aggregator.round_summary()
    target_acc = tt / aggregator.members.size(0)
    target_recorder.loc[len(target_recorder)] = (j+1, target_acc, tt, aggregator.members.size(0), aggregator.attack_samples.size(0), owners[0], owners[1], owners[2], owners[3], owners[4])
    logger.info("Round {}, target_acc {}, true_target {}, true_member {}, pred_member {}, owner0={}, owner1={}, owner2={}, owner3={}, owner4={}"
                .format(j+1, target_acc, tt, aggregator.members.size(0), aggregator.attack_samples.size(0), owners[0], owners[1], owners[2], owners[3], owners[4]))
    if tt > best_true_target:
        best_round = j + 1
        best_true_target = tt
        best_target_acc = target_acc
logger.info("Best round {}, best target acc = {}, best true_garget = {}".format(best_round, best_target_acc, best_true_target))
confidence_recorder = aggregator.get_round_robbing_summary()
target_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + "_" + str(DEFAULT_DISTRIBUTION) + recorder_suffix + "_attack.csv")
confidence_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + "_" + str(DEFAULT_DISTRIBUTION) + recorder_suffix + "_confidence.csv")
