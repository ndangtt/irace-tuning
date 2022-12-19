import pandas as pd
import sys
import os
import importlib

from irace import irace, Parameters, Param, Categorical, Symbol
from irace.compatibility.config_space import convert_from_config_space
import numpy as np

from surrogate import convert_params_to_vec
import json
from ConfigSpace.read_and_write import pcs
from pyrfr import regression
from multiprocessing import cpu_count
from utils import suppress_stdout
import rpy2.robjects as ro

tpoc = importlib.import_module("tuning-PoC")
#from tpoc import get_instances_training, get_instances_validation, get_instances_meta_validation, predict_surrogate, surrogate_target_runner

# Load the list of instance features
with open('./target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json') as f:
    instances_features = json.load(f)
instances = list(instances_features.keys())

# Load the random forest model
model = regression.binary_rss_forest()
model.load_from_binary_file('./target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin')

# Load parameters as ConfigSpace.ConfigurationSpace
with open('./target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs') as f:
    cs = pcs.read(f)
parameters = convert_from_config_space(cs)

instances_for_tuning = tpoc.get_instances_training(instances)
instances_for_evaluation = tpoc.get_instances_meta_validation(instances)

threads = cpu_count()

def evaluate_irace_conf(conf, seed):
    scenario = dict(
        instances = instances_for_tuning,
        maxExperiments = 3000,
        debugLevel = 0,
        parallel = threads,
        digits = 15,
        boundMax = 1000,
        logFile = '', 
        seed = seed
    )

    scenario.update(conf)

    tuner = irace(scenario, parameters, tpoc.surrogate_target_runner)

    with suppress_stdout():
        best_configs: pd.DataFrame = tuner.run()

    # Get a single configuration as the best config.
    best_config = best_configs.to_dict(orient='records')[0]

    def validate(config):
        '''
        Get the performance of a configuration on the validation set
        '''
        sum = 0
        for instance in instances_for_evaluation:
            sum += tpoc.predict_surrogate(config, instance)
        return sum / len(instances_for_evaluation)
    
    return validate(best_config)

if __name__ == "__main__":
    tconfs = pd.read_csv("confs.csv")
    tconfs["performance"] = ""
    nRuns = 50
    for i in range(len(tconfs.index))[:2]:
        conf = tconfs.drop([".ID.",".PARENT.","performance"], axis=1).iloc[i,:].to_dict()
        costs = []
        print(f"\nEvaluating {conf}")
        for seed in range(nRuns):
            print(f"\t{seed}/{nRuns}",end="\r")
            c  = evaluate_irace_conf(conf, seed)
            costs.append(c)
        tconfs.at[i, "performance"] = costs
    tconfs.to_csv("confs.csv",index=False)
