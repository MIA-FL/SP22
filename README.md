### This repository is to provide code and data 

* * *

Supplementry data \- https://mia-fl.github.io/MIA-FL/

* * *

#### Code Handbook

* * *


##### File structures

Backbone files

These files provides the core functionality in the experiment setting

1.  aggregator.py \- The aggregator collects and calculate the gradients from participants
2.  constants.py \- All hyper-parameters
3.  data\_reader.py \- The module loading data from data set files and distribute them to participants
4.  models.py \- The participants, global model, and local attackers
5.  organizer.py \- The module setting up different experiments

* * *

Experiment runnable

These files are the runnable experiment files calling above backbone files

1.  blackbox\_agr\_op.py
2.  blackbox\_agr\_optimized.py
3.  blackbox\_baseline.py
4.  blackbox\_optimized.py
5.  blackbox\_starting\_baseline.py
6.  grey1\_baseline\_texas\_trmean.py
7.  greybox\_I\_baseline\_misleading.py
8.  greybox\_I\_baseline.py
9.  greybox\_II\_baseline.py
10.  greybox1\_starting\_baseline.py
11.  optimized\_greybox1.py
12.  whitebox\_global\_non\_target\_starting\_baseline.py
13.  whitebox\_global\_non\_targeted\_baseline.py
14.  whitebox\_global\_target\_starting\_baseline.py
15.  whitebox\_global\_targeted\_round\_robbin\_shadow\_ver.py
16.  whitebox\_global\_targeted\_round\_robbin\_starting\_point\_baseline.py
17.  whitebox\_global\_targeted\_round\_robbin.py
18.  whitebox\_local\_baseline.py
19.  whitebox\_local\_optimized.py
20.  whitebox\_local\_targeted\_baseline.py

* * *

Setup

The dataset\_purchase.tgz need to be extracted as 'dataset\_purchase' before running

* * *

##### Constant List

Key constants

1.  DEFAULT\_SET
    *   The chosen dataset to run experiments
    *   Available values
        1.  PURCHASE100
        2.  CIFAR\_10
        3.  LOCATION30
        4.  TEXAS100
        5.  GNOME
2.  DEFAULT\_AGR
    *   The chosen robust aggregation mechanisms to defend malicious updates
    *   Available values
        1.  TRMEAN = "Trimmed Mean"
        2.  KRUM = "Krum"
        3.  MULTI\_KRUM = "Multi-Krum"
        4.  MEDIAN = "Median"
        5.  FANG = "Fang"
        6.  None (Please use the python reserved None value)
3.  NUMBER\_OF\_PARTICIPANTS: The number of participants involved in federated learning process
4.  NUMBER\_OF\_ADVERSARY: The number of adversary involved in attack experiments, the adversary is in a DISJOINT set with participants
5.  MAX\_EPOCH: The overall rounds of federated training for the current experiment
6.  TRAIN\_EPOCH: The overall rounds of deferated training without malicious updates. In other words, the number of training rounds that the adversary waits the model to converge
7.  EXPERIMENTAL\_DATA\_DIRECTORY: The directory to store experiment data, it must be a existing directory
8.  GLOBAL\_SEED: The random seed used
