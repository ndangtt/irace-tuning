#!/usr/bin/env python
# coding: utf-8


from irace import irace, Parameters, Param, Categorical, Symbol
from irace.compatibility.config_space import convert_from_config_space
from irace.expressions import List
import numpy as np
from surrogate import convert_params_to_vec
import json
from ConfigSpace.read_and_write import pcs
from pyrfr import regression
from rpy2.robjects.packages import importr
from rpy2.robjects import StrVector
from utils import suppress_stdout, suppress_stderr
from multiprocessing import cpu_count
import pandas as pd

def get_training_experiment(instances):
    return instances[:len(instances) // 3]

def get_training_validation(instances):
    return instances[len(instances) // 3: len(instances) // 3 * 2]

def get_training_meta_validation(instances):
    return instances[len(instances) // 3:]

with open('./target_algorithms/surrogate/cplex_regions200/config_space.cplex_regions200.par10.random.pcs') as f:
    cs = pcs.read(f)

# Load the list of instance features
with open('./target_algorithms/surrogate/cplex_regions200/inst_feat_dict.cplex_regions200.par10.random.json') as f:
    instances_features = json.load(f)


instances = list(instances_features.keys())

# Load the random forest model
model = regression.binary_rss_forest()
model.load_from_binary_file('./target_algorithms/surrogate/cplex_regions200/pyrfr_model.cplex_regions200.par10.random.bin')

def filter_nan(config):
    return dict([
        (k, v) for k, v in config.items() if not pd.isna(v)
    ])

def predict_surrogate(configuration, instance):
    instance_feature = instances_features[instance]['__ndarray__']
    # Call the magic function to convert configurations to a vector
    encoded_configurations = convert_params_to_vec(filter_nan(configuration), cs)
    x = np.hstack([encoded_configurations, instance_feature])
    return model.predict(x)

def target_irace(experiment, scenario):
    print('started target_irace')
    import json
    def surrogate_target_runner(experiment, scenario):
        with open('abc.json', 'w') as f:
            json.dump(experiment, f)
        with open('db.json', 'w') as f:
            json.dump(list(scenario.keys()), f)
        instance = experiment['instance']
        configuration = dict(experiment['configuration'])
        return dict(cost=predict_surrogate(configuration, instance))
  

    scenario = dict(
        instances = get_training_experiment(instances),
        maxExperiments = 1008,
        debugLevel = 0,
        parallel = cpu_count(),
        digits = 15,
        boundMax = 1000,
        logFile = '', 
        seed = experiment['seed']
    )
    
    scenario.update(dict(experiment['configuration']))

    if scenario['capping'] == '1':
        scenario['elitist'] = '1'

    tuner = irace(scenario, convert_from_config_space(cs), surrogate_target_runner)

    with suppress_stdout():
        best_configs: pd.DataFrame = tuner.run()
    
    best_config = best_configs.to_dict(orient='records')[0]

    # TODO: Consider how we can move this to iracepy


    def validate(config):
        validation_instances = get_training_validation(instances)
        sum = 0
        for instance in validation_instances:
            sum += predict_surrogate(config, instance)
        return sum / len(validation_instances)
    
    return dict(cost=validate(best_config))


params = Parameters()
params.capping = Param(Categorical(('0', '1')))
params.cappingType = Param(Categorical(('median', 'mean', 'worst', 'best')), condition=Symbol('capping') == '1')
params.boundType = Param(Categorical(('candidate', 'instance')), condition=Symbol('capping') == '1')
params.testType = Param(Categorical(('f-test', 't-test')))
params.elitist = Param(Categorical(('0', '1')))

scenario = dict(
    instances = np.arange(1),
    maxExperiments = 180,
    debugLevel = 0,
    parallel = 2,
    digits = 15,
    seed = 123
    )

defaults = pd.DataFrame(data=dict(
    capping = [0],
    cappingType = ['median'],
    boundType = ['candidate'], 
    elitist = [1]
))


tuner = irace(scenario, params, target_irace)
best_config = tuner.run()

print(best_config)