#!/usr/bin/env python
# coding: utf-8


from irace import irace, Parameters, Param, Categorical, Symbol
from irace.compatibility.config_space import convert_from_config_space
import numpy as np
from surrogate import convert_params_to_vec
import json
from ConfigSpace.read_and_write import pcs
from pyrfr import regression
from multiprocessing import cpu_count
from utils import suppress_stdout
import pandas as pd
import rpy2.robjects as ro

# Helper functions
def get_instances_training(instances):
    '''Get the training instances for the target irace'''
    return instances[:len(instances) // 3]

def get_instances_validation(instances):
    '''Get the validation instances for validating the performance of target irace'''
    return instances[len(instances) // 3: len(instances) // 3 * 2]

def get_instances_meta_training(instances):
    '''Get the set of instance for the irace with the best configuration to train on. Currently it equals to the training instances'''
    return instances[:len(instances) // 3]

def get_instances_meta_validation(instances):
    '''Get the validation instances for the meta irace'''
    return instances[len(instances) // 3 * 2:]

def filter_nan(config):
    return dict([
        (k, v) for k, v in config.items() if not pd.isna(v)
    ])

# Load parameters as ConfigSpace.ConfigurationSpace
with open('./target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs') as f:
    cs = pcs.read(f)

# Load the list of instance features
with open('./target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json') as f:
    instances_features = json.load(f)
instances = list(instances_features.keys())

# Load the random forest model
model = regression.binary_rss_forest()
model.load_from_binary_file('./target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin')


def predict_surrogate(configuration, instance):
    '''
    Predict the performance of a giving configuration on an instance using the surrogate model.

    Args:
        configuration: A dictionary of configuration
        instance: A string of the instance name
    
    Returns:
        The performance as a floating number by the surrogate model
    '''
    instance_feature = instances_features[instance]['__ndarray__']
    # Call the magic function to convert configurations to a vector
    encoded_configurations = convert_params_to_vec(filter_nan(configuration), cs)
    x = np.hstack([encoded_configurations, instance_feature])
    y = model.predict(x)
    return 10**y

threads = cpu_count()
training_instances = get_instances_training(instances)
parameters = convert_from_config_space(cs)
validation_instances = get_instances_validation(instances)

def surrogate_target_runner(experiment, scenario):
    instance = experiment['instance']
    bound = experiment['bound']
    configuration = dict(experiment['configuration'])
    cost = predict_surrogate(configuration, instance)
    return dict(cost=cost, time=min(bound + 1, cost))

def target_irace(experiment, scenario):
    '''
    The target runner for the irace being tuned
    '''  
    scenario = dict(
        instances = training_instances,
        maxExperiments = 3000,
        debugLevel = 0,
        parallel = threads,
        digits = 15,
        boundMax = 1000,
        logFile = '', 
        seed = experiment['seed']
    )
    
    scenario.update(dict(experiment['configuration']))

    tuner = irace(scenario, parameters, surrogate_target_runner)

    with suppress_stdout():
        best_configs: pd.DataFrame = tuner.run()

    # Get a single configuration as the best config.
    best_config = best_configs.to_dict(orient='records')[0]

    def validate(config):
        '''
        Get the performance of a configuration on the validation set
        '''
        sum = 0
        for instance in validation_instances:
            sum += predict_surrogate(config, instance)
        return sum / len(validation_instances)
    
    return dict(cost=validate(best_config))


params = Parameters()
params.capping = Param(Categorical(('0', '1')), condition=Symbol('elitist') == '1')
params.cappingType = Param(Categorical(('median', 'mean', 'worst', 'best')), condition=Symbol('capping') == '1')
params.boundType = Param(Categorical(('candidate', 'instance')), condition=Symbol('capping') == '1')
params.testType = Param(Categorical(('f-test', 't-test')))
params.elitist = Param(Categorical(('0', '1')))

scenario = dict(
    instances = ['no-instance'], # FIXME: workaround for auto-optimization/iraceplot#32
    maxExperiments = 2000,
    debugLevel = 0,
    parallel = 2,
    digits = 15,
    seed = 123,
    logFile = "log.Rdata"
    )

defaults = pd.DataFrame(data=dict(
    capping = [0],
    cappingType = ['median'],
    boundType = ['candidate'], 
    elitist = [1]
))

if __name__=="__main__":

    tuner = irace(scenario, params, target_irace)
    defaults = "capping cappingType boundType elitist testType\n0 NA NA 1 'f-test'"
    tuner.set_initial_from_str(defaults)
    tuner.scenario["initConfigurations"]
    conf = scenario["initConfigurations"]
    conf["cappingType"] = [ro.NA_Integer]
    conf["boundType"] = [ro.NA_Integer]
    conf = conf.astype([('elitist', 'O'), ('testType', 'O'), ('capping', 'O'), ('boundType', int), ('cappingType', int)])
    tuner.scenario["initConfigurations"] = conf
    best_config = tuner.run()

    print(best_config)


