from models import *
import pandas as pd
import numpy as np
import copy, os, random
import matplotlib.pyplot as plt
from constants import *

class Organizer():
    def __init__(self):
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

    def set_random_seed(self, seed = GLOBAL_SEED):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def simple_black_box(self, logger, acc_recorder, attack_recorder, train_epoch, global_model, \
        participants, attacker):
        
        saved_global_model = None
        saved_participants = None
        saved_attacker = None
        saved_acc_recorder = None
        saved_attack_recorder = None
        train_start = train_epoch-STRIDE if train_epoch != 0 else train_epoch
        

        for j in range(train_start,train_epoch+ATTACK_EPOCH):
            if j == (train_epoch):
                saved_global_model = copy.deepcopy(global_model)
                saved_participants = copy.deepcopy(participants)
                saved_attacker = copy.deepcopy(attacker)
                saved_acc_recorder = copy.deepcopy(acc_recorder)
                saved_attack_recorder = copy.deepcopy(attack_recorder)
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            train_acc_collector = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
                train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                train_acc_collector.append(train_acc)
                test_loss, test_acc = participants[i].test_outcome()
                logger.info("StartPoint {} Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".\
                    format(train_epoch, j + 1, i, test_loss, test_acc, train_loss, train_acc))
            attacker.collect_parameters(global_parameters)
            # attacker.record_pred()
            if j < (train_epoch) and train_epoch != 0:
                attacker.train()
            else:
                attacker.train(attack=True)
            true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
            attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
            all_sample = true_member+true_non_member+false_member+false_non_member
            attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
            attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
            logger.info("StartPoint {} Epoch {} Attack accuracy = {}, Precision = {}, Recall={}".\
                format(train_epoch, j + 1, attack_accuracy, attack_precision, attack_recall))
            pred_acc_member = attacker.evaluate_member_accuracy()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy()
            logger.info("StartPoint {} Epoch {} Prediction accuracy, member={}, non-member={}".\
                format(train_epoch, j + 1, pred_acc_member,pred_acc_non_member))
            attack_recorder.loc[len(attack_recorder)] = (j+1, \
                attack_accuracy, attack_precision, attack_recall, \
                pred_acc_member, pred_acc_non_member, \
                true_member, false_member, true_non_member, false_non_member)
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("StartPoint {} Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(train_epoch, j + 1, test_loss, test_acc, train_acc))
        
        acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + \
                "StartPoint_" + str(train_epoch)+ "_model.csv")
        attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + \
            "StartPoint_" + str(train_epoch) + "_attacker.csv")

        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        best_attack_acc_precision = attack_recorder["precision"][best_attack_index]
        best_attack_acc_recall = attack_recorder["recall"][best_attack_index]
        best_attack_acc_pred_acc_member = attack_recorder["pred_acc_member"][best_attack_index]
        best_attack_acc_pred_acc_non_member = attack_recorder["pred_acc_non_member"][best_attack_index]
        best_attack_acc_true_member = attack_recorder["true_member"][best_attack_index]
        best_attack_acc_false_member = attack_recorder["false_member"][best_attack_index]
        best_attack_acc_true_non_member = attack_recorder["true_non_member"][best_attack_index]
        best_attack_acc_false_non_member = attack_recorder["false_non_member"][best_attack_index]
        return saved_acc_recorder,saved_attack_recorder,saved_global_model,saved_participants,saved_attacker, \
            (int(train_epoch), int(best_attack_acc_epoch), int(best_attack_acc_epoch-train_epoch), best_attack_acc, target_model_train_acc, target_model_test_acc, \
            best_attack_acc_precision, best_attack_acc_recall, best_attack_acc_pred_acc_member, \
            best_attack_acc_pred_acc_non_member, int(best_attack_acc_true_member), \
            int(best_attack_acc_false_member), int(best_attack_acc_true_non_member),\
            int(best_attack_acc_false_non_member))

    def federated_training_black_box_recursive(self, logger, train_epoch, attack_epoch, attack_each_start_recorder):
        global_model = None
        participants = None
        attacker = None
        result = None
        acc_recorder = None
        attack_recorder = None
        
        if train_epoch != 0:
            acc_recorder, attack_recorder, global_model, participants, attacker \
                = self.federated_training_black_box_recursive(logger, \
                train_epoch - 10, attack_epoch, attack_each_start_recorder)
            aggregator = global_model.get_aggregator()
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants[i].update_aggregator(aggregator)
            attacker.update_aggregator(aggregator)
            acc_recorder, attack_recorder, global_model, participants, attacker, result = \
                self.simple_black_box(logger, acc_recorder, attack_recorder, train_epoch, \
                    global_model, participants, attacker)
            attack_each_start_recorder.loc[len(attack_each_start_recorder)] = result
        else:
            acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
            attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
            aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
            global_model = FederatedModel(self.reader, aggregator)
            global_model.init_global_model()
            test_loss, test_acc = global_model.test_outcome()
            logger.info("Global model initiated, loss={}, acc={}".format(test_loss, test_acc))
            participants = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants.append(FederatedModel(self.reader, aggregator))
                participants[i].init_participant(global_model, i)
                test_loss, test_acc = participants[i].test_outcome()
                logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))
            attacker = BlackBoxMalicious(self.reader, aggregator)
            acc_recorder, attack_recorder, global_model, participants, attacker, result = \
                self.simple_black_box(logger, acc_recorder, attack_recorder, train_epoch, \
                    global_model, participants, attacker)

            attack_each_start_recorder.loc[len(attack_each_start_recorder)] = result
        logger.info("\nStarting_point={}\nbest_attack_acc_in_global_epoch={}\nbest_attack_acc_in_attack_epoch={}\nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nprecision={}\nrecall={}\nmember_pred_acc={}\nnon-member_pred_acc={}\ntrue_member={}\nfalse_member={}\ntrue_non_member={}\nfalse_non_member={}\n"\
            .format(int(result[0]),int(result[1]),int(result[2]),result[3],result[4],result[5],result[6],\
            result[7],result[8],result[9],int(result[10]),int(result[11]),int(result[12]),int(result[13])))
        return acc_recorder, attack_recorder, global_model, participants, attacker

    def federated_training_black_box_starting_point(self, logger):
        attack_each_start_recorder = pd.DataFrame(columns=["start_epoch","best_in_global_epoch","best_in_attack_epoch", "acc", "target_model_train_acc", \
            "target_model_test_acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
            "true_member", "false_member", "true_non_member", "false_non_member"])
        self.federated_training_black_box_recursive(logger, MAX_TRAIN_EPOCH, ATTACK_EPOCH, attack_each_start_recorder)
        attack_each_start_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "_MaxTrainEpoch_" + str(MAX_TRAIN_EPOCH) + "_AttackEpoch_" + str(ATTACK_EPOCH) + "_overview_blackbox_start_point.csv")
        
        best_attack_index = attack_each_start_recorder["acc"].idxmax()
        best_start_point = attack_each_start_recorder["start_epoch"][best_attack_index]
        best_attack_acc = attack_each_start_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_each_start_recorder["best_in_global_epoch"][best_attack_index]
        target_model_train_acc = attack_each_start_recorder["target_model_train_acc"][best_attack_index]
        target_model_test_acc = attack_each_start_recorder["target_model_test_acc"][best_attack_index]
        member_pred_acc = attack_each_start_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_each_start_recorder["pred_acc_non_member"][best_attack_index]

        logger.info("Best result:\nstart_point={}\nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n"\
            .format(best_start_point, best_attack_acc, target_model_train_acc, target_model_test_acc, \
                member_pred_acc, non_member_pred_acc, best_attack_acc_epoch))

        return None

if __name__ == '__main__':
    logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY, 'log_{}_{}_MAX_TRAIN_EPOCH{}_AttackEpoch_{}_blackbox_start_point'.format(TIME_STAMP,DEFAULT_SET,str(MAX_TRAIN_EPOCH),str(ATTACK_EPOCH)))
    org = Organizer()
    org.set_random_seed()
    org.federated_training_black_box_starting_point(logger)