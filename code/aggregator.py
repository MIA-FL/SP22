from typing import List, Any

import torch
import numpy as np
import pandas as pd
from constants import *
from data_reader import DataReader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """

    def __init__(self, sample_gradients: torch.Tensor, robust_mechanism=None):
        """
        Initiate the aggregator according to the tensor size of a given sample
        """
        self.sample_gradients = sample_gradients.to(DEVICE)
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(DEVICE)
        self.robust = RobustMechanism(robust_mechanism)


        # AGR related parameters
        self.agr_model = None
        self.agr_model_calculated = False
        self.orig_loss = 0
        self.orig_acc = 0

    def reset(self):
        """
        Reset the aggregator to 0
        """
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size())
        self.agr_model_calculated = False

    def collect(self, gradient: torch.Tensor, source, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        """
        if sample_count is None:
            self.collected_gradients.append(gradient)
            if indices is not None:
                self.counter_by_indices[indices] += 1
            self.counter += 1
        else:
            self.collected_gradients.append(gradient * sample_count)
            if indices is not None:
                self.counter_by_indices[indices] += sample_count
            self.counter += sample_count

    def get_outcome(self, reset=False, by_indices=False):
        """
        Get the aggregated gradients and reset the aggregator if needed, apply robust aggregator mechanism if needed
        """
        if by_indices:
            result = sum(self.collected_gradients) / self.counter_by_indices
        else:
            result = self.robust.getter(self.collected_gradients, malicious_user=NUMBER_OF_ADVERSARY)
        if reset:
            self.reset()
        return result

    def agr_model_acquire(self, model):
        """
        Make use of the given model for AGR verification
        """
        self.agr_model = model
        self.robust.agr_model_acquire(model)

    def agr_loss_gradient_collect(self, gradient: torch.Tensor, indices=None, sample_count=None):
        """
        Collect gradients with AGR activated, the proposed AGR will evaluate the gradients by loss value and accuracy
        """
        if not self.agr_model_calculated:
            self.orig_loss, self.orig_acc = self.agr_model.test_outcome()
            self.agr_model_calculated = True
            # print("Benchmark for current round aggregation: loss={}, acc={}".format(self.orig_loss, self.orig_acc))
        test_loss, test_acc = self.agr_model.test_gradients(gradient)
        print("Implication: loss diff={:+.3e}, acc diff={:+.4f}".format(test_loss-self.orig_loss, test_acc-self.orig_acc), end=" ")
        if self.examine_bar(test_loss, test_acc):
            self.collect(gradient, indices, sample_count)

    def agr_set_orig(self, loss, acc):
        """
        Set the original loss and acc as benchmark to avoid duplicated calculation
        """
        self.orig_loss = loss
        self.orig_acc = acc
        self.agr_model_calculated = True

    def examine_bar(self, test_loss, test_acc, bar=0.05):
        """
        Examine the given loss and acc, see if it meets the criteria to be accepted by the aggregator
        """
        result = True
        diff_loss = test_loss - self.orig_loss
        diff_acc = test_acc - self.orig_acc
        if diff_loss > self.orig_loss * bar or diff_acc < min(- self.orig_acc * bar, - bar * bar):
            result = False
            print("Gradient Rejected!")
        return result


class RobustMechanism:
    """
    The robust aggregator applied in the aggregator
    """
    appearence_list = [0,1,2,3,4]
    def __init__(self, robust_mechanism):
        self.type = robust_mechanism
        if robust_mechanism is None:
            self.function = self.naive_average
        elif robust_mechanism == TRMEAN:
            self.function = self.trmean
        elif robust_mechanism in (KRUM, MULTI_KRUM):
            self.function = self.multi_krum
        elif robust_mechanism == BULYAN:
            self.function = self.bulyan
        elif robust_mechanism == MEDIAN:
            self.function = self.median
        elif robust_mechanism == FANG:
            self.function = self.Fang_defense
        self.agr_model = None


    def agr_model_acquire(self, model):
        """
        Acquire the model used for LRR and ERR verification in Fang Defense
        The model must have the same parameters as the global model
        """
        self.agr_model = model

    def naive_average(self, input_gradients, malicious_user:int):
        """
        The naive aggregator
        """
        return torch.mean(input_gradients, 0)

    def trmean(self, input_gradients, malicious_user: int):
        """
        The trimmed mean
        """
        sorted_updates = torch.sort(input_gradients, 0)[0]
        if malicious_user*2 < len(input_gradients):
            return torch.mean(sorted_updates[malicious_user: -malicious_user], 0)
        else:
            return torch.mean(sorted_updates, 0)

    def median(self, input_gradients, malicious_user: int):
        """
        The median AGR
        """
        return torch.median(input_gradients, 0).values

    def multi_krum(self, all_updates, n_attackers):
        """
        The multi-krum method copied from Mengyap's update
        """
        multi_k =  (self.type == MULTI_KRUM)
        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            # torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
                (candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break

        # aggregate = torch.mean(candidates, dim=0)
        print("Selected candicates = {}".format(np.array(candidate_indices)))
        # if not multi_k:
        #     RobustMechanism.appearence_list = candidate_indices
        RobustMechanism.appearence_list = candidate_indices
        return torch.mean(candidates, dim=0)

    def bulyan(self, all_updates, n_attackers):
        """
        The code for robust AGR Bulyan
        """
        nusers = all_updates.shape[0]
        bulyan_cluster = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(bulyan_cluster) < (nusers - 2 * n_attackers):
            # torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
            # print(distances)

            distances = torch.sort(distances, dim=1)[0]

            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
            if not len(indices):
                break
            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat(
                (bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

        # print('dim of bulyan cluster ', bulyan_cluster.shape)

        n, d = bulyan_cluster.shape
        param_med = torch.median(bulyan_cluster, dim=0)[0]
        sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
        sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]
        print("Selected participants = {}".format(np.array(candidate_indices)))
        return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)

    def Fang_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The LRR mechanism proposed in Fang defense
        """
        # Get the baseline loss and accuracy without removing any of the inputs
        all_avg = torch.mean(input_gradients, 0)
        base_loss, base_acc = self.agr_model.test_gradients(all_avg)
        loss_impact = []
        err_impact = []
        # Get all the loss value and accuracy without ith input
        for i in range(len(input_gradients)):
            avg_without_i = (sum(input_gradients[:i]) + sum(input_gradients[i+1:])) / (input_gradients.size(0) - 1)
            ith_loss, ith_acc = self.agr_model.test_gradients(avg_without_i)
            loss_impact.append(torch.tensor(base_loss - ith_loss))
            err_impact.append(torch.tensor(ith_acc - base_acc))
        loss_impact = torch.hstack(loss_impact)
        err_impact = torch.hstack(err_impact)
        loss_rank = torch.argsort(loss_impact, dim=-1)
        acc_rank = torch.argsort(err_impact, dim=-1)
        result = []
        # print(loss_rank, acc_rank)
        for i in range(len(input_gradients)):
            if i in loss_rank[:-malicious_user] and i in acc_rank[:-malicious_user]:
                result.append(i)
        print("Selected inputs are from participants number {}".format(result))
        RobustMechanism.appearence_list = result
        return torch.mean(input_gradients[result], dim=0)

    def PCA_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The proposed PCA based defense
        """
        all_mean = torch.mean(input_gradients, dim=0)
        input_gradients -= all_mean
        #todo

    def getter(self, x, malicious_user=NUMBER_OF_ADVERSARY):
        """
        The getter method applying the robust AGR
        """
        x = torch.vstack(x)
        return self.function(x, malicious_user)


class GlobalAttacker(Aggregator):
    """
    The attacker is the aggregator
    """
    def __init__(self, data_reader: DataReader, sample_gradients: torch.Tensor, robust_mechanism=None):
        super(GlobalAttacker, self).__init__(sample_gradients, robust_mechanism)
        self.robust = RobustMechanism(None)
        self.data_reader = data_reader
        self.shadow_model = None
        self.all_non_members = None
        self.all_members = None
        self.members = None
        self.non_members = None
        self.all_samples = self.get_attack_sample(data_reader)
        self.attack_samples = self.all_samples
        self.descending_samples = None
        self.shuffled_labels = {}
        self.perform_attack = False
        self.shuffle_labels()
        self.isolated_victim = None
        self.isolated_gradient = []
        self.current_member_rate = 0
        self.isolate_round = 0
        self.recorded_prediction = None
        self.sample_confidence = {}
        self.round_robbin_votes = {}

    def get_attack_sample(self, data_reader: DataReader,
                          attack_samples=NUMBER_OF_ATTACK_SAMPLES, member_rate=BLACK_BOX_MEMBER_RATE):
        """
        Randomly select a sample from the data set
        :return: shuffled data of attacker samples
        """
        member_count = round(attack_samples * member_rate)
        non_member_count = attack_samples - member_count
        self.all_members = data_reader.train_set.flatten()[
            torch.randperm(len(data_reader.train_set.flatten()))[:member_count]]
        self.all_non_members = data_reader.test_set.flatten()[
            torch.randperm(len(data_reader.test_set.flatten()))[:non_member_count]]
        self.members = self.all_members
        self.non_members = self.all_non_members
        return torch.cat([self.members, self.non_members])[torch.randperm(attack_samples)]

    def get_shadow_model(self, shadow_model):
        """
        Get the shadow model for isolating a participant
        """
        self.shadow_model = shadow_model

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

    def gradient_ascent(self, ascent_factor=ASCENT_FACTOR, batch_size=BATCH_SIZE,
                        mislead=True):
        """
        Apply gradient ascent to collected gradients
        """
        cache = self.shadow_model.get_flatten_parameters()
        ascending_samples = self.attack_samples
        # Perform gradient ascent for ascending samples
        i = 0
        while i * batch_size < len(ascending_samples):
            batch_index = ascending_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index)
            out = self.shadow_model.model(x)
            loss = self.shadow_model.loss_function(out, y)
            self.shadow_model.optimizer.zero_grad()
            loss.backward()
            self.shadow_model.optimizer.step()
            i += 1
        asc_gradient = self.shadow_model.get_flatten_parameters() - cache
        self.shadow_model.load_parameters(cache)

        # mislead labels
        mislead_gradients = []
        for k in range(len(self.shuffled_labels)):
            i = 0
            while i * batch_size < len(ascending_samples):
                batch_index = ascending_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index)
                y = self.shuffled_labels[k][batch_index]
                out = self.shadow_model.model(x)
                loss = self.shadow_model.loss_function(out, y)
                self.shadow_model.optimizer.zero_grad()
                loss.backward()
                self.shadow_model.optimizer.step()
                i += 1
            mislead_gradients.append(self.shadow_model.get_flatten_parameters() - cache)
            self.shadow_model.load_parameters(cache)

        # select the best misleading gradient
        selected_k = 0
        largest_gradient_diff = 0
        for k in range(len(mislead_gradients)):
            diff = (mislead_gradients[k] - asc_gradient).norm()
            if diff > largest_gradient_diff:
                largest_gradient_diff = diff
                selected_k = k
        return - asc_gradient * ascent_factor + mislead_gradients[selected_k]

    def get_outcome(self, reset=False, by_indices=False, normalize=False):
        """
        Get the aggregated gradients and reset the aggregator if needed, apply robust aggregator mechanism if needed
        """
        if self.perform_attack:
            self.collected_gradients.append(self.gradient_ascent())
        outcome = super(GlobalAttacker, self).get_outcome(reset, by_indices)
        return outcome

    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.members)
        with torch.no_grad():
            out = self.shadow_model.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members)
        with torch.no_grad():
            out = self.shadow_model.model(batch_x)
        prediction = torch.max(out, 1).indices
        accurate = (prediction == batch_y).sum()
        return accurate / len(batch_y)

    def evaluate_attack_result(self, record_prediction=False):
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
        out = self.shadow_model.model(batch_x)
        prediction = torch.max(out, 1).indices
        pred_members = []
        for i in range(len(self.attack_samples)):
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
                if record_prediction:
                    pred_members.append(self.attack_samples[i])
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.members:
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
        if record_prediction:
            self.recorded_prediction = torch.hstack(pred_members)
        return true_member, false_member, true_non_member, false_non_member

    def collect(self, gradient: torch.Tensor, source, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        """
        if self.isolated_victim is None or source != self.isolated_victim.participant_index:
            if sample_count is None:
                self.collected_gradients.append(gradient)
                if indices is not None:
                    self.counter_by_indices[indices] += 1
                self.counter += 1
            else:
                self.collected_gradients.append(gradient * sample_count)
                if indices is not None:
                    self.counter_by_indices[indices] += sample_count
                self.counter += sample_count
        else:
            self.isolated_gradient.append(gradient)

    def isolate_share(self):
        """
        Share aggregated gradients to the isolated participant
        """
        cache = self.shadow_model.get_flatten_parameters()
        if len(self.isolated_gradient) > 0:
            ascent_grad = self.gradient_ascent()
            norm = self.isolated_gradient[0].norm()
            ascent_grad = ascent_grad * norm / ascent_grad.norm()
            self.isolated_gradient.append(ascent_grad)
            gradient = self.robust.getter(self.isolated_gradient)
            self.isolated_gradient = []
            cache = self.shadow_model.get_flatten_parameters() + gradient
            self.shadow_model.load_parameters(cache)
        return cache

    def get_isolated_victim(self, victim):
        """
        Isolate one of the participants as the victim
        """
        self.isolated_victim = victim
        victim_members = victim.train_set.flatten()
        self.members = torch.tensor(np.intersect1d(victim_members.cpu().detach().numpy(), self.attack_samples.cpu().detach().numpy())).to(DEVICE)
        combined = torch.cat((self.attack_samples, self.members)).to(DEVICE)
        uniques, counts = combined.unique(return_counts=True)
        self.non_members = uniques[counts == 1]
        self.current_member_rate = self.members.size(0) / self.attack_samples.size(0)

    def constraint_attack_according_prediction(self):
        """
        Limit the attack samples to the prediction outcome
        """
        self.attack_samples = self.recorded_prediction
        for sample in self.attack_samples:
            with torch.no_grad():
                self.sample_confidence[sample.item()] = torch.zeros(NUMBER_OF_PARTICIPANTS).to(DEVICE)

    def evaluate_isolated_attack(self):
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
        out = self.shadow_model.model(batch_x)
        prediction = torch.max(out, 1).indices
        confidence = torch.max(out, 1).values
        for i in range(len(self.attack_samples)):
            sample = self.attack_samples[i].item()
            if prediction[i] == batch_y[i]:
                attack_result.append(1)
                self.sample_confidence[sample][self.isolated_victim.participant_index] = confidence[i]
            else:
                attack_result.append(0)
            if sample in self.members:
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

    def print_pred_dictionary(self):
        """
        Print out the prediction outcome for all labels
        """
        correct = 0
        overall = 0
        recorder = pd.DataFrame(columns=["sample","predicted_owner","true_owner","correct","confident0","confident1","confident2","confident3","confident4"])
        for sample in self.sample_confidence.keys():
            predicted_owner = -1
            max_confidence = -9999999999
            for i in range(NUMBER_OF_PARTICIPANTS):
                current_confidence = self.sample_confidence[sample][i]
                if current_confidence != 0 and current_confidence>max_confidence:
                    predicted_owner = i
                    max_confidence = current_confidence
            real_owner = self.get_sample_owner(sample)
            confidence = self.sample_confidence[sample].detach().numpy()
            recorder.loc[len(recorder)] = (sample, predicted_owner, real_owner, predicted_owner==real_owner,
                                           confidence[0], confidence[1], confidence[2], confidence[3], confidence[4])
            if predicted_owner == real_owner:
                correct+=1
            overall+=1
        print("Overall {} samples, correct prediction {}, accuracy={:.4f}".format(overall, correct, correct/overall))
        return recorder

    def get_sample_owner(self, sample):
        """
        Return the owner of this sample
        """
        for i in range(NUMBER_OF_PARTICIPANTS):
            if sample in self.data_reader.get_train_set(i):
                return i
        return -1

    def init_round_robbin(self):
        """
        Initialize the round robbin, update the attack samples to the predicted members and create evaluation schema
        """
        self.evaluate_attack_result(record_prediction=True)
        self.constraint_attack_according_prediction()
        for sample in self.sample_confidence.keys():
            self.round_robbin_votes[sample] = torch.zeros(NUMBER_OF_PARTICIPANTS, dtype=torch.int)
        self.members = torch.tensor(
            np.intersect1d(self.members.cpu().detach().numpy(), self.attack_samples.cpu().detach().numpy())).to(
            DEVICE)
        combined = torch.cat((self.attack_samples, self.members)).to(DEVICE)
        uniques, counts = combined.unique(return_counts=True)
        self.non_members = uniques[counts == 1]
        self.current_member_rate = self.members.size(0) / self.attack_samples.size(0)

    def round_robbin_isolation(self, victim, shadow=None):
        """
        Isolate victim one by one
        """
        self.isolated_victim = victim
        if shadow is None:
            self.shadow_model = victim
        else:
            self.shadow_model = shadow
        victim.share_gradient()
        if len(self.isolated_gradient) > 0:
            ascent_grad = self.gradient_ascent()
            norm = self.isolated_gradient[0].norm()
            ascent_grad = ascent_grad * norm / ascent_grad.norm()
            self.isolated_gradient.append(ascent_grad)
            gradient = self.robust.getter(self.isolated_gradient)
            self.isolated_gradient = []
            cache = self.shadow_model.get_flatten_parameters() + gradient
            self.shadow_model.load_parameters(cache)
            if shadow is not None:
                victim.load_parameters(cache)
            # cache2 = shadow.get_flatten_parameters()
            # cache3 = victim.get_flatten_parameters()
            # print("There are {} same parameters, overall {}".format(torch.sum(cache2 == cache3), cache2.size(0)))

    def round_robbin_evaluation(self, victim, shadow=None):
        """
        Evaluate the attack outcome and update confidence table
        """
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples)
        part_index = victim.participant_index
        with torch.no_grad():
            if shadow is None:
                out = victim.model(batch_x)
            else:
                # out_orig = victim.model(batch_x)
                out = shadow.model(batch_x)
                # same = torch.sum(out.flatten() == out_orig.flatten())
                # print("There are overall {} outputs, {} same ".format(out.flatten().size(0), same))
            soft = torch.nn.Softmax(1)(out)
        pred = torch.max(soft, 1).indices
        conf = torch.max(soft, 1).values
        tm, fm, tn, fn = 0, 0, 0, 0
        for i in range(self.attack_samples.size(0)):
            sample = self.attack_samples[i].item()
            pred_true = pred[i] == batch_y[i]
            ground_true = sample in victim.train_set
            # Evaluate current participant prediction
            if pred_true and ground_true:
                tm += 1
            elif pred_true and not ground_true:
                fm += 1
            elif not pred_true and not  ground_true:
                tn += 1
            else:
                fn += 1
            # Record current participant confidence
            if pred_true:
                self.sample_confidence[sample][part_index] = conf[i]
            else:
                self.sample_confidence[sample][part_index] = -1
        return tm, fm, tn, fn

    def round_summary(self):
        """
        Summarize the attack outcome for the current round
        """
        true_target = 0
        owner_counts = [0] * 5
        for sample in self.sample_confidence.keys():
            temp_owner = -1
            max_confidence = -1
            for i in range(NUMBER_OF_PARTICIPANTS):
                current_confidence = self.sample_confidence[sample][i]
                if current_confidence > max_confidence:
                    temp_owner = i
                    max_confidence = current_confidence
            if temp_owner != -1:
                self.round_robbin_votes[sample][temp_owner] = self.round_robbin_votes[sample][temp_owner] + 1
            true_owner = self.get_sample_owner(sample)
            predicted_owner = -1
            val, index = torch.max(self.round_robbin_votes[sample], 0)
            if val > 0:
                predicted_owner = index
                if predicted_owner == true_owner:
                    true_target += 1
                    owner_counts[predicted_owner] = owner_counts[predicted_owner] + 1
        return true_target, owner_counts

    def get_round_robbing_summary(self):
        """
        Get the data frame describing the round robbing summary
        """
        df = pd.DataFrame(columns=["sample","predicted_owner","true_owner","correct","confident0","confident1","confident2","confident3","confident4","vote0","vote1", "vote2", "vote3", "vote4"])
        for sample in self.sample_confidence.keys():
            true_owner = self.get_sample_owner(sample)
            predicted_owner = -1
            val, index = torch.max(self.round_robbin_votes[sample], 0)
            if val > 0:
                predicted_owner = index.item()
            to_append = [sample, predicted_owner, true_owner, predicted_owner == true_owner]
            for i in range(NUMBER_OF_PARTICIPANTS):
                to_append.append(self.sample_confidence[sample][i].item())
            for i in range(NUMBER_OF_PARTICIPANTS):
                to_append.append(self.round_robbin_votes[sample][i].item())
            df.loc[len(df)] = to_append
        return df