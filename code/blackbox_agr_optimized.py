from models import *
from constants import *
import pandas as pd
import numpy as np
import copy, os, random
import matplotlib.pyplot as plt


class Organizer():
    def __init__(self, train_epoch=TRAIN_EPOCH):
        self.set_random_seed()
        self.reader = DataReader()
        self.target = TargetModel(self.reader)
        self.bar_recorder = 0
        self.last_acc = 0

    def exit_bar(self, acc, threshold, bar):
        if acc - self.last_acc <= threshold:
            self.bar_recorder += 1
        else:
            self.bar_recorder = 0
        self.last_acc = acc
        return self.bar_recorder > bar

    def set_random_seed(self, seed=GLOBAL_SEED):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def federated_training_nontarget_ada_black_box(self, logger, adaptive=False, record_process=True,
                                                   record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Dataset is {}".format(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()
        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", test_loss, test_acc, 0)
        logger.info("Global model initiated, loss={}, acc={}".format(test_loss, test_acc))
        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            test_loss, test_acc = participants[i].test_outcome()
            if DEFAULT_AGR == FANG:
                aggregator.agr_model_acquire(global_model)
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))
        # Initialize attacker
        attacker = BlackBoxMalicious(self.reader, aggregator)
        shuffle_factor = SHUFFLE_FACTOR
        attacker.optimized_evaluation_init()
        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            train_acc_collector = []

            for i in range(NUMBER_OF_PARTICIPANTS):
                if i == 0:
                    print(i)
                    attacker.collect_parameters(global_parameters)
                    if DEFAULT_AGR == "FANG" or "MULTI_KRUM":
                        # logger.info("Current AGR selection is {}".format(aggregator.robust.appearence_list))
                        true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                            )
                        attack_precision = true_member / (true_member + false_member)
                        attack_accuracy = (true_member + true_non_member) / (
                                true_member + true_non_member + false_member + false_non_member)
                        attack_recall = true_member / (true_member + false_non_member)
                    if DEFAULT_AGR == "KRUM" and aggregator.robust.appearence_list == [5]:
                        pass
                    else:
                        true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                            )
                        attack_precision = true_member / (true_member + false_member)
                        attack_accuracy = (true_member + true_non_member) / (
                                true_member + true_non_member + false_member + false_non_member)
                        attack_recall = true_member / (true_member + false_non_member)
                    # attacker.record_pred()
                    # random_attack = random.randint(0,9)

                    if j < TRAIN_EPOCH:
                        attacker.train()
                    else:
                        # print(random_attack)
                        # if random_attack in [2,3,4]:

                        if shuffle_factor < 0.95:
                            shuffle_factor = shuffle_factor + ADJUST_RATE
                            logger.info("Current shuffle factor is {}".format(shuffle_factor))
                            attacker.optimized_attack(shuffle_factor=shuffle_factor)
                        else:
                            shuffle_factor = shuffle_factor - ADJUST_RATE
                            logger.info("Current shuffle factor is {}".format(shuffle_factor))
                            attacker.optimized_attack(shuffle_factor=shuffle_factor)
                        # attacker.train()
                        #     attacker.optimized_attack(shuffle_factor=shuffle_factor)
                        # elif j % 50 == 0:
                        #     shuffle_factor = SHUFFLE_FACTOR
                        #     logger.info("Current shuffle factor is {}".format(shuffle_factor))
                        #     # attacker.train(attack = True)
                        #
                        # else:

                        # else:
                        #     logger.info("Slow down, Current shuffle factor is {}".format(shuffle_factor/2))
                        #     attacker.optimized_attack(shuffle_factor=shuffle_factor/2)

                # The participants collect the global parameters before training
                # print(i)
                participants[i].collect_parameters(global_parameters)

                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
                train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                train_acc_collector.append(train_acc)
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info(
                    "Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i,
                                                                                                             test_loss,
                                                                                                             test_acc,
                                                                                                             train_loss,
                                                                                                            train_acc))


            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision,
                                                                                 attack_recall))
            pred_acc_member = attacker.evaluate_member_accuracy()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy()
            logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                        .format(pred_acc_member, pred_acc_non_member,
                                BLACK_BOX_MEMBER_RATE * pred_acc_member + (1 - BLACK_BOX_MEMBER_RATE) * (
                                            1 - pred_acc_non_member)))
            attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                         attack_accuracy, attack_precision, attack_recall, \
                                                         pred_acc_member, pred_acc_non_member, \
                                                         true_member, false_member, true_non_member, false_non_member)
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            # Printing and recording
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info(
                "Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc,
                                                                                        train_acc))
        # Printing and recording
        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]

        logger.info(
            "Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n" \
                .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc, \
                        non_member_pred_acc, best_attack_acc_epoch))
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            recorder_suffix = "greybox_misleading"
            if adaptive:
                recorder_suffix = "whitebox_active"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) + \
                                   "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + recorder_suffix + "_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                         'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_blackbox_op'.format(TIME_STAMP, DEFAULT_SET,
                                                                                             DEFAULT_AGR,
                                                                                             TRAIN_EPOCH,
                                                                                             MAX_EPOCH - TRAIN_EPOCH))
    org = Organizer()
    org.set_random_seed()
    org.federated_training_nontarget_ada_black_box(logger)