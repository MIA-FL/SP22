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

    def simple_whitebox_global(self, logger, acc_recorder, attack_recorder, train_epoch, global_model, \
        participants, attacker):
        owner_recorder = pd.DataFrame(columns=["StartPoint", "participant", "round", "target_acc", \
            "true_member", "true_non_member", "false_member", "false_non_member"])
        saved_global_model = None
        saved_participants = None
        #saved_attacker = None
        saved_acc_recorder = None
        saved_attack_recorder = None
        train_start = train_epoch-STRIDE if train_epoch != 0 else train_epoch

        aggregator = global_model.get_aggregator()
        shadow_model = FederatedModel(self.reader, aggregator)
        shadow_model.init_global_model()
        

        for j in range(train_start,train_epoch+ATTACK_EPOCH):
            if j == (train_epoch):
                saved_global_model = copy.deepcopy(global_model)
                saved_participants = copy.deepcopy(participants)
                #saved_attacker = copy.deepcopy(attacker)
                saved_acc_recorder = copy.deepcopy(acc_recorder)
                saved_attack_recorder = copy.deepcopy(attack_recorder)

            if (j < (train_epoch) and train_epoch != 0) == False:
                aggregator.perform_attack = True
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
                
            true_member, false_member, true_non_member, false_non_member = aggregator.evaluate_attack_result()
            attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
            all_sample = true_member+true_non_member+false_member+false_non_member
            attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
            attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
            logger.info("StartPoint {} Epoch {} Attack accuracy = {}, Precision = {}, Recall={}".\
                format(train_epoch, j + 1, attack_accuracy, attack_precision, attack_recall))
            pred_acc_member = aggregator.evaluate_member_accuracy()
            pred_acc_non_member = aggregator.evaluate_non_member_accuracy()
            logger.info("StartPoint {} Epoch {} Prediction accuracy, member={}, non-member={}".\
                format(train_epoch, j + 1, pred_acc_member,pred_acc_non_member))
            attack_recorder.loc[len(attack_recorder)] = (j+1, \
                attack_accuracy, attack_precision, attack_recall, \
                pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                true_member, false_member, true_non_member, false_non_member)
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("StartPoint {} Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(train_epoch, j + 1, test_loss, test_acc, train_acc))
        
        


        aggregator.evaluate_attack_result(record_prediction=True)
        aggregator.constraint_attack_according_prediction()
        rounds = {}
        for k in range(NUMBER_OF_PARTICIPANTS):
            aggregator.get_shadow_model(shadow_model)
            shadow_model.collect_parameters(participants[k].get_flatten_parameters())
            aggregator.get_isolated_victim(participants[k])
            member_count = aggregator.members.size(0)
            logger.info("Currently isolating participant {}, member rate={}, true member={}, total pred member={}"
                        .format(k, aggregator.current_member_rate, member_count, aggregator.attack_samples.size(0)))
            for j in range(WHITE_BOX_GLOBAL_TARGETED_ROUND):
                if j not in rounds:
                    rounds[j] = []
                
                
                participants[k].share_gradient()
                participants[k].load_parameters(aggregator.isolate_share())
                true_member, false_member, true_non_member, false_non_member = aggregator.evaluate_isolated_attack()
                ownership_accuracy = 0
                if true_member + false_member >0:
                    ownership_accuracy = true_member / (true_member + false_member)
                membership_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                # attacker.record_pred()
                rounds[j].append(ownership_accuracy)
                logger.info("Isolation round {}, Membership accuracy = {:.4f}, Ownership accuracy = {:.4f}".format(j, membership_accuracy, ownership_accuracy))
                pred_acc_member = 0
                if true_member>0:
                    pred_acc_member = aggregator.evaluate_member_accuracy().item()
                pred_acc_non_member = aggregator.evaluate_non_member_accuracy().item()
                logger.info("Prediction accuracy, member={:.4f}, non-member={:.4f}, expected_accuracy={:.4f}"
                            .format(pred_acc_member, pred_acc_non_member,
                                    aggregator.current_member_rate * pred_acc_member + (1 - aggregator.current_member_rate) * (
                                            1 - pred_acc_non_member)))
                owner_recorder.loc[len(owner_recorder)] = (train_epoch, k, j, ownership_accuracy, true_member, true_non_member, false_member, false_non_member)

        best_round = None
        best_owner_acc = 0
        for round in rounds.keys():
            mean = sum(rounds[round])/len(rounds[round])
            if mean > best_owner_acc:
                best_owner_acc = mean
                best_round = round
            

        acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + \
                "StartPoint_" + str(train_epoch)+ "_model.csv")
        attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + \
            "StartPoint_" + str(train_epoch) + "_attacker.csv")
        owner_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + \
            "StartPoint_" + str(train_epoch) + "_owner.csv")




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
        return saved_acc_recorder,saved_attack_recorder,saved_global_model,saved_participants,attacker, \
            (int(train_epoch), int(best_attack_acc_epoch), int(best_attack_acc_epoch-train_epoch), best_attack_acc, target_model_train_acc, target_model_test_acc, \
            best_attack_acc_precision, best_attack_acc_recall, best_attack_acc_pred_acc_member, \
            best_attack_acc_pred_acc_non_member, int(best_attack_acc_true_member), \
            int(best_attack_acc_false_member), int(best_attack_acc_true_non_member),\
            int(best_attack_acc_false_non_member), best_round, best_owner_acc)

    def federated_training_greykbox_recursive(self, logger, train_epoch, attack_epoch, attack_each_start_recorder):
        global_model = None
        participants = None
        attacker = None
        result = None
        acc_recorder = None
        attack_recorder = None
        
        if train_epoch != 0:
            acc_recorder, attack_recorder, global_model, participants, attacker \
                = self.federated_training_greykbox_recursive(logger, \
                train_epoch - 10, attack_epoch, attack_each_start_recorder)
            aggregator = global_model.get_aggregator()
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants[i].update_aggregator(aggregator)
            acc_recorder, attack_recorder, global_model, participants, attacker, result = \
                self.simple_whitebox_global(logger, acc_recorder, attack_recorder, train_epoch, \
                    global_model, participants, attacker)
            attack_each_start_recorder.loc[len(attack_each_start_recorder)] = result
        else:
            acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
            attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])

            aggregator = GlobalAttacker(self.reader, self.target.get_flatten_parameters(),DEFAULT_AGR)
            logger.info("AGR is {}".format(DEFAULT_AGR))
            logger.info(str(DEFAULT_SET))
            logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
            global_model = FederatedModel(self.reader, aggregator)
            global_model.init_global_model()
            
            test_loss, test_acc = global_model.test_outcome()
            aggregator.get_shadow_model(global_model)
            logger.info("Global model initiated, loss={}, acc={}".format(test_loss, test_acc))
            participants = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants.append(FederatedModel(self.reader, aggregator))
                participants[i].init_participant(global_model, i)
                test_loss, test_acc = participants[i].test_outcome()
                logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))
            
            acc_recorder, attack_recorder, global_model, participants, attacker, result = \
                self.simple_whitebox_global(logger, acc_recorder, attack_recorder, train_epoch, \
                    global_model, participants, attacker)

            attack_each_start_recorder.loc[len(attack_each_start_recorder)] = result
        logger.info("\nStarting_point={}\nbest_attack_acc_in_global_epoch={}\nbest_attack_acc_in_attack_epoch={}\nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nprecision={}\nrecall={}\nmember_pred_acc={}\nnon-member_pred_acc={}\ntrue_member={}\nfalse_member={}\ntrue_non_member={}\nfalse_non_member={}\nbest_round={}\nbest_owner_acc={}\n"\
            .format(int(result[0]),int(result[1]),int(result[2]),result[3],result[4],result[5],result[6],\
            result[7],result[8],result[9],int(result[10]),int(result[11]),int(result[12]),int(result[13]),int(result[14]),result[15]))
        return acc_recorder, attack_recorder, global_model, participants, attacker

    def federated_training_simple_whitebox_global_starting_point(self, logger):
        attack_each_start_recorder = pd.DataFrame(columns=["start_epoch","best_in_global_epoch","best_in_attack_epoch", "acc", "target_model_train_acc", \
            "target_model_test_acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
            "true_member", "false_member", "true_non_member", "false_non_member", "best_round", "best_owner_acc"])
        self.federated_training_greykbox_recursive(logger, MAX_TRAIN_EPOCH, ATTACK_EPOCH, attack_each_start_recorder)
        attack_each_start_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "_MaxTrainEpoch_" + str(MAX_TRAIN_EPOCH) + "_AttackEpoch_" + str(ATTACK_EPOCH) + "_overview_whitebox_global_target_point.csv")
        
        best_attack_index = attack_each_start_recorder["best_owner_acc"].idxmax()
        best_start_point = attack_each_start_recorder["start_epoch"][best_attack_index]
        best_attack_acc = attack_each_start_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_each_start_recorder["best_in_global_epoch"][best_attack_index]
        target_model_train_acc = attack_each_start_recorder["target_model_train_acc"][best_attack_index]
        target_model_test_acc = attack_each_start_recorder["target_model_test_acc"][best_attack_index]
        member_pred_acc = attack_each_start_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_each_start_recorder["pred_acc_non_member"][best_attack_index]
        best_round = attack_each_start_recorder["best_round"][best_attack_index]
        best_owner_acc = attack_each_start_recorder["best_owner_acc"][best_attack_index]

        logger.info("Best result:\nstart_point={}\nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\nbest_round={}\nbest_owner_acc={}\n"\
            .format(best_start_point, best_attack_acc, target_model_train_acc, target_model_test_acc, \
                member_pred_acc, non_member_pred_acc, best_attack_acc_epoch, best_round, best_owner_acc))

        return None

if __name__ == '__main__':
    logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY, 'log_{}_{}_MAX_TRAIN_EPOCH{}_AttackEpoch_{}_whitebox_global_target_start_point'.format(TIME_STAMP,DEFAULT_SET,str(MAX_TRAIN_EPOCH),str(ATTACK_EPOCH)))
    org = Organizer()
    org.set_random_seed()
    org.federated_training_simple_whitebox_global_starting_point(logger)