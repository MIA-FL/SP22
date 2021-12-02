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

    def set_random_seed(self, seed = GLOBAL_SEED):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def plot_attack_performance(self, attack_recorder):
        epoch = attack_recorder["epoch"]
        acc = attack_recorder["acc"]
        precise = attack_recorder["precise"]

        plt.figure(figsize=(20,10))

        plt.subplot(1,2,1)
        plt.plot(epoch, acc)
        #plt.vlines(train_epoch, 0, max(non_attacked_non_member_loss)+0.2, colors = "r", linestyles = "dashed")
        plt.title('Attack Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')

        plt.subplot(1,2,2)
        plt.plot(epoch, precise)
        #plt.vlines(train_epoch, 0, max(non_attacked_member_acc)+0.2, colors = "r", linestyles = "dashed")
        plt.title('Attack Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        
        plt.show()

    def centralized_training(self):
        self.target.init_parameters()
        max_acc = 0
        best_model = self.target.model.state_dict()
        for i in range(MAX_EPOCH):
            print("Starting epoch {}...".format(i))
            loss, acc = self.target.normal_epoch()
            print("Training loss = {}, training accuracy = {}".format(loss, acc))
            loss, acc = self.target.test_outcome()
            print("Test loss = {}, test accuracy = {}".format(loss, acc))
            if acc > max_acc:
                max_acc = acc
                best_model = copy.deepcopy(self.target.model.state_dict())
        torch.save(best_model, EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Centralized_best_model")

    def federated_training_basic(self, record_model=False, record_process=True):
        """
        A code sample to start a federated learning setting using this package
        """
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "loss", "accuracy"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), robust_mechanism=DEFAULT_AGR)
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        loss, acc = global_model.test_outcome()

        # provide the global model as baseline model to Fang defense
        if DEFAULT_AGR == FANG:
            aggregator.agr_model_acquire(global_model)

        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", loss, acc)
        print("Global model initiated, loss={}, acc={}".format(loss, acc))
        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, loss, acc)
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))
        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
                # Printing and recording
                loss, acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, loss, acc)
                print("Epoch {} Participant {}, loss={}, acc={}".format(j + 1, i, loss, acc))
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            # Printing and recording
            loss, acc = global_model.test_outcome()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", loss, acc)
            print("Epoch {} Global model, loss={}, acc={}".format(j + 1, loss, acc))
        # Printing and recording
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Federated_Train.csv")

    def federated_training_black_box(self,logger, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info(str(DEFAULT_SET))
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
        for j in range(MAX_EPOCH):
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
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # attacker attack
            attacker.collect_parameters(global_parameters)
            # attacker.record_pred()
            # if j < TRAIN_EPOCH:
            attacker.train()
            # else:
            #     attacker.train(attack=True)
            true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
            attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
            all_sample = true_member+true_non_member+false_member+false_non_member
            attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
            attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
            #attack_precise, attack_acc, member_acc = attacker.evaluate_attack_result()
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
            pred_acc_member = attacker.evaluate_member_accuracy()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy()
            logger.info("Prediction accuracy, member={}, non-member={}".format(pred_acc_member,pred_acc_non_member))
            attack_recorder.loc[len(attack_recorder)] = (j+1, \
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
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
        # Printing and recording
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR)+ "blackbox_passive_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) +"_AGR="+ str(DEFAULT_AGR)+ "blackbox_passive_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

    def simple_black_box(self, train_epoch, total_epoch, global_model, participants, attacker):
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        saved_global_model = None
        saved_participants = None
        saved_attacker = None

        for j in range(total_epoch):
            if j == 10 :
                attacker.train()
            else:
                saved_global_model = copy.deepcopy(global_model)
                saved_participants = copy.deepcopy(participants)
                saved_attacker = copy.deepcopy(attacker)
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
            attacker.collect_parameters(global_parameters)
            # attacker.record_pred()
            if j < 10 and total_epoch > 10:
                attacker.train()
            else:
                attacker.train(attack=True)
            true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
            attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
            all_sample = true_member+true_non_member+false_member+false_non_member
            attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
            attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0

            pred_acc_member = attacker.evaluate_member_accuracy()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy()

            attack_recorder.loc[len(attack_recorder)] = (j+1+train_epoch, \
                attack_accuracy, attack_precision, attack_recall, \
                pred_acc_member, pred_acc_non_member, \
                true_member, false_member, true_non_member, false_non_member)
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index] - train_epoch
        best_attack_acc_precision = attack_recorder["precision"][best_attack_index]
        best_attack_acc_recall = attack_recorder["recall"][best_attack_index]
        best_attack_acc_pred_acc_member = attack_recorder["pred_acc_member"][best_attack_index]
        best_attack_acc_pred_acc_non_member = attack_recorder["pred_acc_non_member"][best_attack_index]
        best_attack_acc_true_member = attack_recorder["true_member"][best_attack_index]
        best_attack_acc_false_member = attack_recorder["false_member"][best_attack_index]
        best_attack_acc_true_non_member = attack_recorder["true_non_member"][best_attack_index]
        best_attack_acc_false_non_member = attack_recorder["false_non_member"][best_attack_index]
        return saved_global_model,saved_participants,saved_attacker, (train_epoch, best_attack_acc_epoch, \
            best_attack_acc, best_attack_acc_precision, \
            best_attack_acc_recall, best_attack_acc_pred_acc_member, best_attack_acc_pred_acc_non_member, \
                best_attack_acc_true_member, best_attack_acc_false_member, best_attack_acc_true_non_member, \
                    best_attack_acc_false_non_member)

    def federated_training_black_box_recursive(self, logger, train_epoch, attack_epoch, attack_each_start_recorder):
        global_model = None
        participants = None
        attacker = None
        result = None
        total_epoch = train_epoch + attack_epoch
        if train_epoch != 0:
            global_model, participants, attacker = self.federated_training_black_box_recursive(logger, \
                train_epoch - 10, attack_epoch, attack_each_start_recorder)
            aggregator = global_model.get_aggregator()
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants[i].update_aggregator(aggregator)
            attacker.update_aggregator(aggregator)
            saved_global_model,saved_participants,saved_attacker, result = \
                self.simple_black_box(train_epoch, total_epoch, global_model, participants, attacker)
            attack_each_start_recorder.loc[len(attack_each_start_recorder)] = result
        else:
            aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
            global_model = FederatedModel(self.reader, aggregator)
            global_model.init_global_model()
            participants = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                participants.append(FederatedModel(self.reader, aggregator))
                participants[i].init_participant(global_model, i)
            attacker = BlackBoxMalicious(self.reader, aggregator)
            saved_global_model,saved_participants,saved_attacker, result = \
                self.simple_black_box(train_epoch, total_epoch, global_model, participants, attacker)

            attack_each_start_recorder.loc[len(attack_each_start_recorder)] = result
        #print(result)
        logger.info("\nStarting_point={}\nbest_attack_acc_epoch={}\nattack_acc={}\nprecision={}\nrecall={}\nmember_pred_acc={}\nnon-member_pred_acc={}\ntrue_member={}\nfalse_member={}\ntrue_non_member={}\nfalse_non_member={}\n"\
            .format(int(result[0]),int(result[1]),result[2],result[3],result[4],result[5],result[6],\
                int(result[7]),int(result[8]),int(result[9]),int(result[10])))
        return saved_global_model,saved_participants,saved_attacker

    def federated_training_black_box_starting_point(self, logger, attack_epoch = ATTACK_EPOCH):
        attack_each_start_recorder = pd.DataFrame(columns=["start_epoch","epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        self.federated_training_black_box_recursive(logger, MAX_TRAIN_EPOCH, attack_epoch, attack_each_start_recorder)
        return None

    def federated_training_nontargeted_grey_box(self, logger,record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(), robust_mechanism=DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info("Dataset is {}".format(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()
        if DEFAULT_AGR == FANG:
            aggregator.agr_model_acquire(global_model)
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
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            print("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))
        # Initialize attacker
        attacker = GreyBoxMalicious(self.reader, aggregator)
        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            train_acc_collector = []
            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
                # Printing and recording
                train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                train_acc_collector.append(train_acc)
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # Printing and recording
            # attacker attack
            attacker.collect_parameters(global_parameters)
            # attacker.record_pred()
            attacker.record_pred_before_attack()
            if j < TRAIN_EPOCH:
                attacker.gradient_ascent()
            else:
                attacker.gradient_ascent(attack=True)
                
                history={}
                
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
                attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
                all_sample = true_member+true_non_member+false_member+false_non_member
                attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
                attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
                
                #attack_precise, attack_acc, member_acc = attacker.evaluate_attack_result()
                logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
                history[i] = (attack_accuracy, attack_precision, attack_recall)
                    #print("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall), end=" ")
                pred_acc_member = attacker.evaluate_member_accuracy()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                #print("Best base epoch:{}, acc:{}".format(best_base_pred,best_acc))
                logger.info("Prediction accuracy, member={}, non-member={}".format(pred_acc_member,pred_acc_non_member))
                attack_recorder.loc[len(attack_recorder)] = (j+1, \
                    attack_accuracy, attack_precision, attack_recall, \
                    pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                    true_member, false_member, true_non_member, false_non_member)
                    
            attacker.record_pred_after_attack() # save prediction for comparing changes after attack
            
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
        # Printing and recording

        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]

        logger.info("Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n"\
            .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc,\
                non_member_pred_acc, best_attack_acc_epoch))

        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) +\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "nontargeted_greybox_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "nontargeted_greybox_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

    def federated_training_nontarget_misleadning_grey_box(self,logger, adaptive=False, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info(str(DEFAULT_SET))
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
        attacker = WhiteBoxMalicious(self.reader, aggregator)
        for j in range(MAX_EPOCH):
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
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # attacker attack
            attacker.collect_parameters(global_parameters)
            attacker.record_pred_before_attack()
            
            # attacker.record_pred()
            if j >= TRAIN_EPOCH:
                attacker.train(attack=True)
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(adaptive_prediction=False)
                attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
                all_sample = true_member+true_non_member+false_member+false_non_member
                attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
                attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
                logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
                pred_acc_member = attacker.evaluate_member_accuracy()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                            .format(pred_acc_member,pred_acc_non_member, BLACK_BOX_MEMBER_RATE*pred_acc_member + (1-BLACK_BOX_MEMBER_RATE)* (1-pred_acc_non_member)))
                attack_recorder.loc[len(attack_recorder)] = (j+1, \
                    attack_accuracy, attack_precision, attack_recall, \
                    pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                    true_member, false_member, true_non_member, false_non_member)
            else:
                attacker.train()
            attacker.record_pred_after_attack()
            
            # Global model collects the aggregated gradient

            global_model.apply_gradient()
            # Printing and recording
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
        # Printing and recording

        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]

        logger.info("Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n"\
            .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc,\
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
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix +"_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix + "_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

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
        attacker = WhiteBoxMalicious(self.reader, aggregator)
        attacker.optimized_evaluation_init()
        ascent_factor = ASCENT_FACTOR
        for j in range(MAX_EPOCH):
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
            # attacker attack
            attacker.collect_parameters(global_parameters)
            if DEFAULT_AGR in ["FANG", "MULTI_KRUM"]:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_optimized_attack_result(
                    adaptive_prediction=adaptive)
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            if DEFAULT_AGR == "KRUM" and aggregator.robust.appearence_list == [5]:
                pass
            else:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                    adaptive_prediction=adaptive)
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            # attacker.record_pred()

            if j < TRAIN_EPOCH:
                attacker.train()
            else:
                if ascent_factor < 1:
                    ascent_factor += 0.002
                    attacker.partial_gradient_ascent(ascent_factor=ascent_factor)
                    print(ascent_factor)
                else:
                    attacker.partial_gradient_ascent(ascent_factor=ascent_factor)
                    print(ascent_factor)
                # attacker.train(attack = True)
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision,
                                                                                 attack_recall))
            pred_acc_member = attacker.evaluate_member_accuracy().cpu()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy().cpu()
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

    def federated_training_targeted_grey_box_once_absent(self, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "loss", "accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters())
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        loss, acc = global_model.test_outcome()
        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", loss, acc)
        print("Global model initiated, loss={}, acc={}".format(loss, acc))
        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, loss, acc)
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))
        # Initialize attacker
        attacker = GreyBoxMalicious(self.reader, aggregator)
        current_absent = None
        for j in range(MAX_EPOCH+1):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            current_absent = j%NUMBER_OF_PARTICIPANTS
            for i in range(NUMBER_OF_PARTICIPANTS):
                if i == current_absent:
                    continue
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient()
                # Printing and recording
                loss, acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, loss, acc)
                print("Epoch {} Participant {}, loss={}, acc={}".format(j + 1, i, loss, acc))
            # Printing and recording
            loss, acc = global_model.test_outcome()
            # attacker attack
            attacker.collect_parameters(global_parameters)
            # attacker.record_pred()
            attacker.record_pred_before_attack()
            if j < TRAIN_EPOCH:
                attacker.gradient_ascent()
            else:
                attacker.gradient_ascent(attack=True)
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
                attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
                all_sample = true_member+true_non_member+false_member+false_non_member
                attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
                attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
                print("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall), end=" ")
                
                pred_acc_member = attacker.evaluate_member_accuracy()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                print("Prediction accuracy, member={}, non-member={}".format(pred_acc_member,pred_acc_non_member))
                attack_recorder.loc[len(attack_recorder)] = (j+1, \
                    attack_accuracy, attack_precision, attack_recall, \
                    pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                    true_member, false_member, true_non_member, false_non_member)
                    
            attacker.record_pred_after_attack() # save prediction for comparing changes after attack
            
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", loss, acc)
            print("Epoch {} Global model, loss={}, acc={}".format(j + 1, loss, acc))
        # Printing and recording
        attacker.target_member_once_absent()
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "targeted_greybox_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "targeted_greybox_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)


    def federated_training_targeted_grey_box_once_per_round(self, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "loss", "accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),)
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        loss, acc = global_model.test_outcome()
        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", loss, acc)
        print("Global model initiated, loss={}, acc={}".format(loss, acc))
        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, loss, acc)
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))
        # Initialize attacker
        attacker = GreyBoxMalicious(self.reader, aggregator)
        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            current_attendee = j%NUMBER_OF_PARTICIPANTS
            if j < TRAIN_EPOCH:
                for i in range(NUMBER_OF_PARTICIPANTS):
                    # The participants collect the global parameters before training
                    participants[i].collect_parameters(global_parameters)
                    # The participants calculate local gradients and share to the aggregator
                    participants[i].share_gradient()
                    # Printing and recording
                    loss, acc = participants[i].test_outcome()
                    if record_process:
                        acc_recorder.loc[len(acc_recorder)] = (j + 1, i, loss, acc)
                    print("Epoch {} Participant {}, loss={}, acc={}".format(j + 1, i, loss, acc))
            else:
                # The participants collect the global parameters before training
                participants[current_attendee].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[current_attendee].share_gradient()
                # Printing and recording
                loss, acc = participants[current_attendee].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, current_attendee, loss, acc)
                print("Epoch {} Participant {}, loss={}, acc={}".format(j + 1, current_attendee, loss, acc))
            # Printing and recording
            loss, acc = global_model.test_outcome()
            # attacker attack
            attacker.collect_parameters(global_parameters)
            # attacker.record_pred()
            attacker.record_pred_before_attack()
            if j < TRAIN_EPOCH:
                attacker.gradient_ascent()
            else:
                attacker.gradient_ascent(attack=True)
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
                attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
                all_sample = true_member+true_non_member+false_member+false_non_member
                attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
                attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
                print("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall), end=" ")
                
                pred_acc_member = attacker.evaluate_member_accuracy()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                print("Prediction accuracy, member={}, non-member={}".format(pred_acc_member,pred_acc_non_member))
                attack_recorder.loc[len(attack_recorder)] = (j+1, \
                    attack_accuracy, attack_precision, attack_recall, \
                    pred_acc_member, pred_acc_non_member, \
                    true_member, false_member, true_non_member, false_non_member)
                    
            attacker.record_pred_after_attack() # save prediction for comparing changes after attack
            
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", loss, acc)
            print("Epoch {} Global model, loss={}, acc={}".format(j + 1, loss, acc))
        # Printing and recording
        attacker.target_member_once_per_round()
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "targeted_greybox_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "targeted_greybox_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

    def federated_training_targeted_grey_box(self, logger, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
                                                "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        target_recorder = pd.DataFrame(columns=["round", "true_target", "true_member", "total_pred_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),)
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
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))
        # Initialize attacker
        attacker = GreyBoxMalicious(self.reader, aggregator)
        attacker.prune_data()
        current_attendee = 0
        for j in range(MAX_EPOCH+2):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            train_acc_collector = []
            attacker.collect_parameters(global_parameters)
            attacker.record_pred_before_attack()
            if j < TRAIN_EPOCH:
                attacker.gradient_ascent()
                attacker.record_pred_after_attack()
                for i in range(NUMBER_OF_PARTICIPANTS):
                    # The participants collect the global parameters before training
                    participants[i].collect_parameters(global_parameters)
                    # The participants calculate local gradients and share to the aggregator
                    participants[i].share_gradient()
                    train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                    train_acc_collector.append(train_acc)
                    # Printing and recording
                    test_loss, test_acc = participants[i].test_outcome()
                    if record_process:
                        acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                    logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            else:
                if j%2 == 0:
                    attacker.gradient_ascent(attack=True)
                    attacker.record_pred_after_attack()
                    history={}
                
                    true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
                    attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
                    all_sample = true_member+true_non_member+false_member+false_non_member
                    attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
                    attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
                    
                    #attack_precise, attack_acc, member_acc = attacker.evaluate_attack_result()
                    logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
                    history[i] = (attack_accuracy, attack_precision, attack_recall)
                        #print("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall), end=" ")
                    pred_acc_member = attacker.evaluate_member_accuracy()
                    pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                    #print("Best base epoch:{}, acc:{}".format(best_base_pred,best_acc))
                    logger.info("Prediction accuracy, member={}, non-member={}".format(pred_acc_member,pred_acc_non_member))
                    attack_recorder.loc[len(attack_recorder)] = (j+1, \
                        attack_accuracy, attack_precision, attack_recall, \
                        pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                        true_member, false_member, true_non_member, false_non_member)
                else:
                    
                    # The participants collect the global parameters before training
                    participants[current_attendee].collect_parameters(global_parameters)
                    # The participants calculate local gradients and share to the aggregator
                    participants[current_attendee].share_gradient()
                    train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                    train_acc_collector.append(train_acc)
                    # Printing and recording
                    test_loss, test_acc = participants[i].test_outcome()
                    current_attendee = (current_attendee+1)%NUMBER_OF_PARTICIPANTS
                    if record_process:
                        acc_recorder.loc[len(acc_recorder)] = (j + 1, current_attendee, test_loss, test_acc, train_acc)
                    logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            global_model.apply_gradient()
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
        # Target
        pred_member = attacker.get_pred_member()
        target_counter = {}
        total_attack_epoch = MAX_EPOCH - TRAIN_EPOCH
        epochs_in_round = (NUMBER_OF_PARTICIPANTS * 2)
        total_round = total_attack_epoch // epochs_in_round
        for rounds in range(total_round):
            true_target, true_member, total_pred_member = attacker.target_member(pred_member, target_counter, epochs_in_round, rounds)
            target_recorder.loc[len(target_recorder)] = (rounds+1, true_target, true_member, total_pred_member)
            logger.info("UsedRounds = {} TruePredict={} TrueMember={} TotalPredMemebr={}"\
            .format(rounds+1, true_target, true_member, total_pred_member))

        
        
        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]

        logger.info("Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n"\
            .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc,\
                non_member_pred_acc, best_attack_acc_epoch))

        max_true_target = target_recorder["true_target"].max()
        best_round_index = target_recorder[target_recorder["true_target"]==max_true_target].index[0]
        best_round = target_recorder["round"][best_round_index]
        true_member = target_recorder["true_member"][best_round_index]
        total_pred_member = target_recorder["total_pred_member"][best_round_index]
        target_acc = max_true_target/true_member if true_member != 0 else 0

        logger.info("Best target: \niid={}\nbest_round={}\ntarget_acc={}\ntrue_target={}\ntrue_member={}\ntotal_pred_member={}"\
            .format(str(DEFAULT_DISTRIBUTION==None), best_round, target_acc, max_true_target, true_member, total_pred_member))

        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR) +\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "targeted_greybox_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "targeted_greybox_attacker.csv")
            target_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + \
                    "TotalRounds"+ str(total_round) + "iid" + str(DEFAULT_DISTRIBUTION==None) +"target_greybox_performance.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)



    def agr_proposed_federated_black_box(self, record_process=True, record_model=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "loss", "accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", "acc", "precise", "member_acc", "pred_acc_member", "pred_acc_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters())
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        # Acquire model for AGR computation
        aggregator.agr_model_acquire(global_model)
        loss, acc = global_model.test_outcome()
        # Recording and printing
        if record_process:
            acc_recorder.loc[len(acc_recorder)] = (0, "g", loss, acc)
        print("Global model initiated, loss={}, acc={}".format(loss, acc))
        # Initialize participants
        participants = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            participants.append(FederatedModel(self.reader, aggregator))
            participants[i].init_participant(global_model, i)
            loss, acc = participants[i].test_outcome()
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, loss, acc)
            print("Participant {} initiated, loss={}, acc={}".format(i, loss, acc))
        # Initialize attacker
        attacker = BlackBoxMalicious(self.reader, aggregator)
        for j in range(MAX_EPOCH):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()

            for i in range(NUMBER_OF_PARTICIPANTS):
                # The participants collect the global parameters before training
                participants[i].collect_parameters(global_parameters)
                # The participants calculate local gradients and share to the aggregator
                participants[i].share_gradient(agr=True)
                # Printing and recording
                loss, acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, loss, acc)
                print("Epoch {} Participant {}, loss={:.4e}, acc={:.4f}".format(j + 1, i, loss, acc))
            # Printing and recording
            loss, acc = aggregator.orig_loss, aggregator.orig_acc
            # attacker attack
            if self.exit_bar(acc, 0.005, 5):
                attacker.collect_parameters(global_parameters)
                attack_precise, attack_acc, member_acc = attacker.evaluate_attack_result()
                print("Attack accuracy = {:.4f}, precise = {:.4f}, member acc={:.4f}".format(attack_acc, attack_precise, member_acc), end=" ")
                pred_acc_member = attacker.evaluate_member_accuracy()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                print("Prediction accuracy, member={:.4f}, non-member={:.4f}".format(pred_acc_member,pred_acc_non_member))
                attack_recorder.loc[len(attack_recorder)] = (j+1, attack_acc, attack_precise, member_acc, pred_acc_member, pred_acc_non_member)
                attacker.partial_gradient_ascent(agr=True)
            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", loss, acc)
            print("Epoch {} Global model, loss={}, acc={}".format(j + 1, loss, acc))
        # Printing and recording
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Federated_Models.csv")
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "Federated_Train.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + "BlackBoxAttack.csv")

    def federated_training_nontarget_white_box(self, logger, adaptive=False, record_process=True, record_model=False, adaptive_epoch=200):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info(str(DEFAULT_SET))
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
        attacker = WhiteBoxMalicious(self.reader, aggregator)
        for j in range(MAX_EPOCH):
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
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # attacker attack
            attacker.collect_parameters(global_parameters)
            attacker.record_pred_before_attack()
            if j < adaptive_epoch and j > TRAIN_EPOCH:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(adaptive_prediction=adaptive)
            else:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                    adaptive_prediction=False)
            attack_precision = true_member / (true_member + false_member)
            attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
            attack_recall = true_member / (true_member + false_non_member)
            # attacker.record_pred()
            if j >= TRAIN_EPOCH:
                attacker.train(attack=True)
            else:
                attacker.train()
            attacker.record_pred_after_attack()
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
            pred_acc_member = attacker.evaluate_member_accuracy()
            pred_acc_non_member = attacker.evaluate_non_member_accuracy()
            logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                        .format(pred_acc_member,pred_acc_non_member, BLACK_BOX_MEMBER_RATE*pred_acc_member + (1-BLACK_BOX_MEMBER_RATE)* (1-pred_acc_non_member)))
            attack_recorder.loc[len(attack_recorder)] = (j+1, \
                attack_accuracy, attack_precision, attack_recall, \
                pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                true_member, false_member, true_non_member, false_non_member)
            # Global model collects the aggregated gradient

            global_model.apply_gradient()
            # Printing and recording
            if DEFAULT_AGR in [MULTI_KRUM, FANG]:
                logger.info("AGR selected participants = {}".format(RobustMechanism.appearence_list))
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
        # Printing and recording
        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]

        logger.info("Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n"\
            .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc,\
                non_member_pred_acc, best_attack_acc_epoch))
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            recorder_suffix = "greybox_misleading"
            if adaptive:
                recorder_suffix = "whitebox_local"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix +"_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix + "_attacker.csv")


    def federated_training_nontarget_ada_grey_box(self,logger, adaptive=False, record_process=True, record_model=False, plot=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
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
        attacker = WhiteBoxMalicious(self.reader, aggregator)
        attacker.optimized_evaluation_init()
        ascent_factor = ASCENT_FACTOR
        for j in range(MAX_EPOCH):
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
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # attacker attack
            attacker.collect_parameters(global_parameters)
            if DEFAULT_AGR in [FANG, MULTI_KRUM]:
                # if DEFAULT_AGR == KRUM and aggregator.robust.appearence_list == [5]:
                #     pass
            # else:

                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_optimized_attack_result(adaptive_prediction=adaptive)
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                            true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            else:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                    adaptive_prediction=adaptive)
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            # attacker.record_pred()

            if j < TRAIN_EPOCH:
                attacker.train()
            else:
                if  ascent_factor < 1 and j % 3 == 0:
                    ascent_factor+=ADJUST_RATE
                    attacker.train(attack=True, ascent_factor=ascent_factor, ascent_fraction=0.5)
                    logger.info("Current ascent factor is {}".format(ascent_factor))
                else:
                    attacker.train(attack=True, ascent_factor=ascent_factor, ascent_fraction=0.5)
                    logger.info("Current ascent factor is {}".format(ascent_factor))
                # attacker.train(attack = True)
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
            # if DEFAULT_AGR == KRUM and aggregator.robust.appearence_list == [5]:
            #     pass
            # el

            if DEFAULT_AGR in [FANG, MULTI_KRUM]:
                logger.info("Current AGR selection is {}".format(aggregator.robust.appearence_list))
                pred_acc_member = attacker.optimized_evaluate_member_accuracy().cpu()
                pred_acc_non_member = attacker.optimized_evaluate_non_member_accuracy().cpu()
                logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                            .format(pred_acc_member,pred_acc_non_member, BLACK_BOX_MEMBER_RATE*pred_acc_member + (1-BLACK_BOX_MEMBER_RATE)* (1-pred_acc_non_member)))
                attack_recorder.loc[len(attack_recorder)] = (j+1, \
                    attack_accuracy, attack_precision, attack_recall, \
                    pred_acc_member, pred_acc_non_member, \
                    true_member, false_member, true_non_member, false_non_member)
            else:
                pred_acc_member = attacker.evaluate_member_accuracy().cpu()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy().cpu()
                logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                            .format(pred_acc_member, pred_acc_non_member,
                                    BLACK_BOX_MEMBER_RATE * pred_acc_member + (1 - BLACK_BOX_MEMBER_RATE) * (
                                                1 - pred_acc_non_member)))
                attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                             attack_accuracy, attack_precision, attack_recall, \
                                                             pred_acc_member, pred_acc_non_member, \
                                                             true_member, false_member, true_non_member,
                                                             false_non_member)

            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            # Printing and recording
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
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
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+ \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + recorder_suffix +"_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + recorder_suffix + "_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

    def federated_training_targeted_white_box(self,logger, adaptive=False, record_process=True, record_model=False, plot=False, adaptive_epoch=200):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        target_recorder = pd.DataFrame(columns=["round", "true_target", "true_member", "total_pred_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info(str(DEFAULT_SET))
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
        attacker = WhiteBoxMalicious(self.reader, aggregator)
        attacker.prune_data()
        current_attendee = 0
        for j in range(MAX_EPOCH+2):
            # The global model's parameter is shared to each participant before every communication round
            global_parameters = global_model.get_flatten_parameters()
            train_acc_collector = []
            attacker.collect_parameters(global_parameters)
            attacker.record_pred_before_attack()
            
            if j < TRAIN_EPOCH:
                attacker.train()
                attacker.record_pred_after_attack()
                for i in range(NUMBER_OF_PARTICIPANTS):
                    # The participants collect the global parameters before training
                    participants[i].collect_parameters(global_parameters)
                    # The participants calculate local gradients and share to the aggregator
                    participants[i].share_gradient()
                    # Printing and recording
                    train_loss, train_acc = participants[i].train_loss, participants[i].train_acc
                    train_acc_collector.append(train_acc)
                    # Printing and recording
                    test_loss, test_acc = participants[i].test_outcome()
                    if record_process:
                        acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                    logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            else:
                if j%2 == 0:
                    attacker.train(attack=True)
                    attacker.record_pred_after_attack()
                    history={}
                    if j < adaptive_epoch and j > TRAIN_EPOCH:
                        true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(adaptive_prediction=True, rounds=(j-TRAIN_EPOCH)//(2 * NUMBER_OF_PARTICIPANTS))
                    else:
                        true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                            adaptive_prediction=False, rounds=(j-TRAIN_EPOCH)//(2 * NUMBER_OF_PARTICIPANTS))
                
                    # true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result()
                    attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
                    all_sample = true_member+true_non_member+false_member+false_non_member
                    attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
                    attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
                    
                    #attack_precise, attack_acc, member_acc = attacker.evaluate_attack_result()
                    logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
                    history[i] = (attack_accuracy, attack_precision, attack_recall)
                        #print("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall), end=" ")
                    pred_acc_member = attacker.evaluate_member_accuracy()
                    pred_acc_non_member = attacker.evaluate_non_member_accuracy()
                    #print("Best base epoch:{}, acc:{}".format(best_base_pred,best_acc))
                    logger.info("Prediction accuracy, member={}, non-member={}".format(pred_acc_member,pred_acc_non_member))
                    attack_recorder.loc[len(attack_recorder)] = (j+1, \
                        attack_accuracy, attack_precision, attack_recall, \
                        pred_acc_member.cpu(), pred_acc_non_member.cpu(), \
                        true_member, false_member, true_non_member, false_non_member)
                else:
                    
                    # The participants collect the global parameters before training
                    participants[current_attendee].collect_parameters(global_parameters)
                    # The participants calculate local gradients and share to the aggregator
                    participants[current_attendee].share_gradient()
                    train_loss, train_acc = participants[current_attendee].train_loss, participants[current_attendee].train_acc
                    train_acc_collector.append(train_acc)
                    # Printing and recording
                    test_loss, test_acc = participants[current_attendee].test_outcome()
                    if record_process:
                        acc_recorder.loc[len(acc_recorder)] = (j + 1, current_attendee, test_loss, test_acc, train_acc)
                    logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, current_attendee, test_loss, test_acc, train_loss, train_acc))
                    current_attendee = (current_attendee+1)%NUMBER_OF_PARTICIPANTS
            global_model.apply_gradient()
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))

        # Target
        target_counter = {}
        total_attack_epoch = MAX_EPOCH - TRAIN_EPOCH
        epochs_in_round = (NUMBER_OF_PARTICIPANTS * 2)
        total_round = total_attack_epoch // epochs_in_round
        for rounds in range(total_round):
            pred_member = attacker.get_pred_member(rounds)
            target_recorder.loc[len(target_recorder)] = attacker.target_member(pred_member, target_counter, epochs_in_round, rounds)

        best_attack_index = attack_recorder["acc"].idxmax()
        best_attack_acc = attack_recorder["acc"][best_attack_index]
        best_attack_acc_epoch = attack_recorder["epoch"][best_attack_index]
        target_model_index = acc_recorder[acc_recorder["epoch"] == best_attack_acc_epoch].index
        target_model_train_acc = acc_recorder["train_accuracy"][target_model_index].values[0]
        target_model_test_acc = acc_recorder["test_accuracy"][target_model_index].values[0]
        member_pred_acc = attack_recorder["pred_acc_member"][best_attack_index]
        non_member_pred_acc = attack_recorder["pred_acc_non_member"][best_attack_index]

        logger.info("Best result: \nattack_acc={}\ntarget_model_train_acc={}\ntarget_model_test_acc={}\nmember_pred_acc={}\nnon-member_pred_acc={}\nbest_attack_acc_epoch={}\n"\
            .format(best_attack_acc, target_model_train_acc, target_model_test_acc, member_pred_acc,\
                non_member_pred_acc, best_attack_acc_epoch))

        max_true_target = target_recorder["true_target"].max()
        best_round_index = target_recorder[target_recorder["true_target"]==max_true_target].index[0]
        best_round = target_recorder["round"][best_round_index]
        true_member = target_recorder["true_member"][best_round_index]
        total_pred_member = target_recorder["total_pred_member"][best_round_index]
        target_acc = max_true_target/true_member if true_member != 0 else 0

        logger.info("Best target: \niid={}\nbest_round={}\ntarget_acc={}\ntrue_target={}\ntrue_member={}\ntotal_pred_member={}"\
            .format(str(DEFAULT_DISTRIBUTION==None), best_round, target_acc, max_true_target, true_member, total_pred_member))

        # Printing and recording
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            recorder_suffix = "greybox_misleading"
            if adaptive:
                recorder_suffix = "whitebox_active"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix +"_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix + "_attacker.csv")
            target_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + \
                    "TotalRounds"+ str(total_round) + "iid" + str(DEFAULT_DISTRIBUTION==None) +"target_local_whitebox_performance.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

    def federated_training_nontarget_whitebox_global(self,logger, record_process=True, record_model=False, plot=False, defend=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = GlobalAttacker(self.reader, self.target.get_flatten_parameters(),DEFAULT_AGR)
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info(str(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()
        aggregator.get_shadow_model(global_model)
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
            if defend:
                participants[i].activate_defend()
            if DEFAULT_AGR == FANG:
                aggregator.agr_model_acquire(global_model)
            # Recording and printing
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (0, i, test_loss, test_acc, 0)
            logger.info("Participant {} initiated, loss={}, acc={}".format(i, test_loss, test_acc))
        for j in range(MAX_EPOCH):
            if j > TRAIN_EPOCH:
                aggregator.perform_attack = True
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
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # attacker attack
            true_member, false_member, true_non_member, false_non_member = aggregator.evaluate_attack_result()
            attack_precision = true_member / (true_member + false_member)
            attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
            attack_recall = true_member / (true_member + false_non_member)
            # attacker.record_pred()
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
            pred_acc_member = aggregator.evaluate_member_accuracy()
            pred_acc_non_member = aggregator.evaluate_non_member_accuracy()
            logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                        .format(pred_acc_member,pred_acc_non_member, BLACK_BOX_MEMBER_RATE*pred_acc_member + (1-BLACK_BOX_MEMBER_RATE)* (1-pred_acc_non_member)))
            attack_recorder.loc[len(attack_recorder)] = (j+1, \
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
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
            if j % 5 == 0 and defend:
                logger.info("Current drop out for participants {}, {}, {}, {}, {}"
                            .format(participants[0].drop_out, participants[1].drop_out, participants[2].drop_out, participants[3].drop_out, participants[4].drop_out))
        # Printing and recording
        if record_model:
            param_recorder["global"] = global_model.get_flatten_parameters().detach().numpy()
            for i in range(NUMBER_OF_PARTICIPANTS):
                param_recorder["participant{}".format(i)] = participants[i].get_flatten_parameters().detach().numpy()
            param_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + "Federated_Models.csv")
        if record_process:
            recorder_suffix = "whitebox_global_nontargeted"
            if defend:
                recorder_suffix = "whitebox_global_defense"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix +"_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + "_AGR="+ str(DEFAULT_AGR) + recorder_suffix + "_attacker.csv")
        if plot:
            self.plot_attack_performance(attack_recorder)

    def targeted_global_attack(self, logger, record_process=True):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", "isolated",
                                                "member_acc", "owner_acc(precision)", "recall", "pred_acc_member", "pred_acc_non_member",
                                                "true_member", "false_member", "true_non_member", "false_non_member"])
        # Initialize aggregator with given parameter size
        aggregator = GlobalAttacker(self.reader, self.target.get_flatten_parameters())
        logger.info("AGR is {}".format(DEFAULT_AGR))
        logger.info(str(DEFAULT_SET))
        logger.info("Member ratio is {}".format(BLACK_BOX_MEMBER_RATE))
        # Initialize global model
        global_model = FederatedModel(self.reader, aggregator)
        global_model.init_global_model()
        shadow_model = FederatedModel(self.reader, aggregator)
        shadow_model.init_global_model()
        test_loss, test_acc = global_model.test_outcome()
        aggregator.get_shadow_model(global_model)
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
        for j in range(MAX_EPOCH):
            if j > TRAIN_EPOCH:
                aggregator.perform_attack = True
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
            # attacker attack
            true_member, false_member, true_non_member, false_non_member = aggregator.evaluate_attack_result()
            attack_precision = true_member/(true_member+false_member) if (true_member+false_member) != 0 else 0
            all_sample = true_member+true_non_member+false_member+false_non_member
            attack_accuracy = (true_member+true_non_member)/(all_sample) if (all_sample) != 0 else 0
            attack_recall = true_member/(true_member+false_non_member) if (true_member+false_non_member) != 0 else 0
            # attacker.record_pred()
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision,
                                                                                 attack_recall))
            pred_acc_member = aggregator.evaluate_member_accuracy().item()
            pred_acc_non_member = aggregator.evaluate_non_member_accuracy().item()
            logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                        .format(pred_acc_member, pred_acc_non_member,
                                BLACK_BOX_MEMBER_RATE * pred_acc_member + (1 - BLACK_BOX_MEMBER_RATE) * (
                                            1 - pred_acc_non_member)))
            attack_recorder.loc[len(attack_recorder)] = (j + 1, "NA",
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
        recorder_suffix = "whitebox_global_targeted_preparation"
        if record_process:
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + "_AGR=" + str(DEFAULT_AGR) + recorder_suffix + "_model.csv")
        aggregator.evaluate_attack_result(record_prediction=True)
        aggregator.constraint_attack_according_prediction()
        for k in range(NUMBER_OF_PARTICIPANTS):
            aggregator.get_shadow_model(shadow_model)
            shadow_model.collect_parameters(participants[k].get_flatten_parameters())
            aggregator.get_isolated_victim(participants[k])
            member_count = aggregator.members.size(0)
            logger.info("Currently isolating participant {}, member rate={}, true member={}, total pred member={}"
                        .format(k, aggregator.current_member_rate, member_count, aggregator.attack_samples.size(0)))
            for j in range(WHITE_BOX_GLOBAL_TARGETED_ROUND):
                participants[k].share_gradient()
                participants[k].load_parameters(aggregator.isolate_share())
                true_member, false_member, true_non_member, false_non_member = aggregator.evaluate_isolated_attack()
                ownership_accuracy = 0
                if true_member + false_member >0:
                    ownership_accuracy = true_member / (true_member + false_member)
                membership_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                # attacker.record_pred()
                logger.info("Isolation round {}, Membership accuracy = {:.4f}, Ownership accuracy = {:.4f}".format(j, membership_accuracy, ownership_accuracy))
                pred_acc_member = 0
                if true_member>0:
                    pred_acc_member = aggregator.evaluate_member_accuracy().item()
                pred_acc_non_member = aggregator.evaluate_non_member_accuracy().item()
                logger.info("Prediction accuracy, member={:.4f}, non-member={:.4f}, expected_accuracy={:.4f}"
                            .format(pred_acc_member, pred_acc_non_member,
                                    aggregator.current_member_rate * pred_acc_member + (1 - aggregator.current_member_rate) * (
                                            1 - pred_acc_non_member)))
                attack_recorder.loc[len(attack_recorder)] = (j + 1, k,
                                                             attack_accuracy, attack_precision, attack_recall, \
                                                             pred_acc_member, pred_acc_non_member, \
                                                             true_member, false_member, true_non_member,
                                                             false_non_member)
        if record_process:
            confidence_recorder = aggregator.print_pred_dictionary()
            confidence_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                                   "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + "_" + str(DEFAULT_DISTRIBUTION) + recorder_suffix + "_confidence.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + \
                                   "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(
                MAX_EPOCH - TRAIN_EPOCH) + "_" + str(DEFAULT_DISTRIBUTION) + recorder_suffix + "_attacker.csv")

    def optimized_local_nontarget_white_box(self, logger, adaptive=False, record_process=True, record_model=False):
        # Initialize data frame for recording purpose
        acc_recorder = pd.DataFrame(columns=["epoch", "participant", "test_loss", "test_accuracy", "train_accuracy"])
        attack_recorder = pd.DataFrame(columns=["epoch", \
            "acc", "precision", "recall", "pred_acc_member", "pred_acc_non_member", \
                "true_member", "false_member", "true_non_member", "false_non_member"])
        param_recorder = pd.DataFrame()
        # Initialize aggregator with given parameter size
        aggregator = Aggregator(self.target.get_flatten_parameters(),DEFAULT_AGR)
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
        attacker = WhiteBoxMalicious(self.reader, aggregator)
        attacker.optimized_evaluation_init()
        ascent_factor = ASCENT_FACTOR
        for j in range(MAX_EPOCH):
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
                # Printing and recording
                test_loss, test_acc = participants[i].test_outcome()
                if record_process:
                    acc_recorder.loc[len(acc_recorder)] = (j + 1, i, test_loss, test_acc, train_acc)
                logger.info("Epoch {} Participant {}, test_loss={}, test_acc={}, train_loss={}, train_acc={}".format(j + 1, i, test_loss, test_acc, train_loss, train_acc))
            # attacker attack
            attacker.collect_parameters(global_parameters)
            if DEFAULT_AGR in [FANG, MULTI_KRUM]:
                # if DEFAULT_AGR == KRUM and aggregator.robust.appearence_list == [5]:
                #     pass
            # else:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_optimized_attack_result(adaptive_prediction=adaptive)
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                            true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            else:
                true_member, false_member, true_non_member, false_non_member = attacker.evaluate_attack_result(
                    adaptive_prediction=adaptive)
                attack_precision = true_member / (true_member + false_member)
                attack_accuracy = (true_member + true_non_member) / (
                        true_member + true_non_member + false_member + false_non_member)
                attack_recall = true_member / (true_member + false_non_member)
            # attacker.record_pred()

            if j < TRAIN_EPOCH:
                attacker.train()
            else:
                attacker.train(attack=True, ascent_fraction=1, white_box_optimize=True)
                # attacker.train(attack = True)
            logger.info("Attack accuracy = {}, Precision = {}, Recall={}".format(attack_accuracy, attack_precision, attack_recall))
            # if DEFAULT_AGR == KRUM and aggregator.robust.appearence_list == [5]:
            #     pass
            # el
            if DEFAULT_AGR in [FANG, MULTI_KRUM]:
                pred_acc_member = attacker.optimized_evaluate_member_accuracy().cpu()
                pred_acc_non_member = attacker.optimized_evaluate_non_member_accuracy().cpu()
                logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                            .format(pred_acc_member,pred_acc_non_member, BLACK_BOX_MEMBER_RATE*pred_acc_member + (1-BLACK_BOX_MEMBER_RATE)* (1-pred_acc_non_member)))
                attack_recorder.loc[len(attack_recorder)] = (j+1, \
                    attack_accuracy, attack_precision, attack_recall, \
                    pred_acc_member, pred_acc_non_member, \
                    true_member, false_member, true_non_member, false_non_member)
            else:
                pred_acc_member = attacker.evaluate_member_accuracy().cpu()
                pred_acc_non_member = attacker.evaluate_non_member_accuracy().cpu()
                logger.info("Prediction accuracy, member={}, non-member={}, expected_accuracy={}"
                            .format(pred_acc_member, pred_acc_non_member,
                                    BLACK_BOX_MEMBER_RATE * pred_acc_member + (1 - BLACK_BOX_MEMBER_RATE) * (
                                                1 - pred_acc_non_member)))
                attack_recorder.loc[len(attack_recorder)] = (j + 1, \
                                                             attack_accuracy, attack_precision, attack_recall, \
                                                             pred_acc_member, pred_acc_non_member, \
                                                             true_member, false_member, true_non_member,
                                                             false_non_member)

            # Global model collects the aggregated gradient
            global_model.apply_gradient()
            if DEFAULT_AGR in [MULTI_KRUM, FANG]:
                logger.info("AGR selected participants = {}".format(RobustMechanism.appearence_list))
            # Printing and recording
            test_loss, test_acc = global_model.test_outcome()
            train_acc = torch.mean(torch.tensor(train_acc_collector)).item()
            if record_process:
                acc_recorder.loc[len(acc_recorder)] = (j + 1, "g", test_loss, test_acc, train_acc)
            logger.info("Epoch {} Global model, test_loss={}, test_acc={}, train_acc={}".format(j + 1, test_loss, test_acc, train_acc))
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
            recorder_suffix = "whitebox_optimized"
            acc_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+ \
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + recorder_suffix +"_model.csv")
            attack_recorder.to_csv(EXPERIMENTAL_DATA_DIRECTORY + TIME_STAMP + DEFAULT_SET + str(DEFAULT_AGR)+\
                "TrainEpoch" + str(TRAIN_EPOCH) + "AttackEpoch" + str(MAX_EPOCH-TRAIN_EPOCH) + recorder_suffix + "_attacker.csv")

if __name__ == '__main__':
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    logger = make_logger("project", EXPERIMENTAL_DATA_DIRECTORY,
                         'log_{}_{}_{}_TrainEpoch{}_AttackEpoch{}_whitebox_global_targeted'.format(TIME_STAMP, DEFAULT_SET, DEFAULT_AGR,
                                                                                   TRAIN_EPOCH,
                                                                                   MAX_EPOCH - TRAIN_EPOCH))
    org = Organizer()
    org.set_random_seed()
    org.targeted_global_attack(logger)
