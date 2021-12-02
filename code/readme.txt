## Code directory for MIA-FL

# Backbone files
These files provides the core functionality in the experiment setting
aggregator.py // The aggregator collects and calculate the gradients from participants
constants.py // All hyper-parameters
data_reader.py // The module loading data from data set files and distribute them to participants
models.py // The participants, global model, and local attackers
organizer.py // The module setting up different experiments

# Experiment runnable
These files are the runnable experiment files calling above backbone files
blackbox_agr_op.py
blackbox_agr_optimized.py
blackbox_baseline.py
blackbox_optimized.py
blackbox_starting_baseline.py
grey1_baseline_texas_trmean.py
greybox_I_baseline_misleading.py
greybox_I_baseline.py
greybox_II_baseline.py
greybox1_starting_baseline.py
optimized_greybox1.py
whitebox_global_non_target_starting_baseline.py
whitebox_global_non_targeted_baseline.py
whitebox_global_target_starting_baseline.py
whitebox_global_targeted_round_robbin_shadow_ver.py
whitebox_global_targeted_round_robbin_starting_point_baseline.py
whitebox_global_targeted_round_robbin.py
whitebox_local_baseline.py
whitebox_local_optimized.py
whitebox_local_targeted_baseline.py

# Set up
The dataset_purchase.tgz need to be extracted before running