import logging
import sys
import os
import naslib as nl
from argparse import ArgumentParser
from naslib.defaults.predictor_evaluator import PredictorEvaluator

import ConfigSpace as CS
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
import numpy as np
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario

from naslib.predictors import BayesianLinearRegression, BOHAMIANN, BonasPredictor, \
    DNGOPredictor, EarlyStopping, Ensemble, GCNPredictor, GPPredictor, \
    LCEPredictor, LCEMPredictor, LGBoost, MLPPredictor, NGBoost, OmniNGBPredictor, \
    OmniSemiNASPredictor, RandomForestPredictor, SVR_Estimator, SemiNASPredictor, \
    SoLosspredictor, SparseGPPredictor, VarSparseGPPredictor, XGBoost, ZeroCostV1, \
    ZeroCostV2, GPWLPredictor, OmniPredictor

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, \
    DartsSearchSpace, NasBenchNLPSearchSpace

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import get_project_root

params_xgb = {
    'objective': 'reg:squarederror',
    'eval_metric': "rmse",
    'booster': 'gbtree',
    'max_depth': None,
    'min_child_weight': None,
    'colsample_bytree': None,
    'learning_rate': None,
    'colsample_bylevel': None,
}
def predictor_run(configuration, seed, instance, budget, **kwargs):
    params_xgb["max_depth"] = configuration["max_depth"]
    params_xgb["min_child_weight"] = configuration["min_child_weight"]
    params_xgb["colsample_bytree"] = configuration["colsample_bytree"]
    params_xgb["learning_rate"] = configuration["learning_rate"]
    params_xgb["colsample_bylevel"] = configuration["colsample_bylevel"]
    print(params_xgb, flush=True)

    # evaluate the predictor
    results = predictor_evaluator.evaluate(params=params_xgb)
    print(results, flush=True)
    return 1 - results[-1]["kendalltau"]


config = utils.get_config_from_file_path(config_file='predictor_config_test.yaml', config_type='predictor')
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_predictors = {
        'bananas': Ensemble(predictor_type='bananas', num_ensemble=3, hpo_wrapper=True),
        'bayes_lin_reg': BayesianLinearRegression(encoding_type='adjacency_one_hot'),
        'bohamiann': BOHAMIANN(encoding_type='adjacency_one_hot'),
        'bonas': BonasPredictor(encoding_type='bonas', hpo_wrapper=True),
        'dngo': DNGOPredictor(encoding_type='adjacency_one_hot'),
        'fisher': ZeroCostV2(config, batch_size=64, method_type='fisher'),
        'gcn': GCNPredictor(encoding_type='gcn', hpo_wrapper=True),
        'gp': GPPredictor(encoding_type='adjacency_one_hot'),
        'gpwl': GPWLPredictor(ss_type=config.search_space, kernel_type='wloa', optimize_gp_hyper=True, h='auto'),
        'grad_norm': ZeroCostV2(config, batch_size=64, method_type='grad_norm'),
        'grasp': ZeroCostV2(config, batch_size=64, method_type='grasp'),
        'jacov': ZeroCostV1(config, batch_size=64, method_type='jacov'),
        'lce': LCEPredictor(metric=Metric.VAL_ACCURACY),
        'lce_m': LCEMPredictor(metric=Metric.VAL_ACCURACY),
        'lcsvr': SVR_Estimator(metric=Metric.VAL_ACCURACY, all_curve=False,
                               require_hyper=False),
        'lgb': LGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=False),
        'mlp': MLPPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=True),
        'nao': SemiNASPredictor(encoding_type='seminas', semi=False, hpo_wrapper=False),
        'ngb': NGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=False),
        'rf': RandomForestPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=False),
        'seminas': SemiNASPredictor(encoding_type='seminas', semi=True, hpo_wrapper=False),
        'snip': ZeroCostV2(config, batch_size=64, method_type='snip'),
        'sotl': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTL'),
        'sotle': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLE'),
        'sotlema': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLEMA'),
        'sparse_gp': SparseGPPredictor(encoding_type='adjacency_one_hot',
                                       optimize_gp_hyper=True, num_steps=100),
        'synflow': ZeroCostV2(config, batch_size=64, method_type='synflow'),
        'valacc': EarlyStopping(metric=Metric.VAL_ACCURACY),
        'valloss': EarlyStopping(metric=Metric.VAL_LOSS),
        'var_sparse_gp': VarSparseGPPredictor(encoding_type='adjacency_one_hot',
                                              optimize_gp_hyper=True, num_steps=200),
        'xgb': XGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=False),
        # path encoding experiments:
        'bayes_lin_reg_path': BayesianLinearRegression(encoding_type='path'),
        'bohamiann_path': BOHAMIANN(encoding_type='path'),
        'dngo_path': DNGOPredictor(encoding_type='path'),
        'gp_path': GPPredictor(encoding_type='path'),
        'lgb_path': LGBoost(encoding_type='path', hpo_wrapper=False),
        'ngb_path': NGBoost(encoding_type='path', hpo_wrapper=False),
        # omni:
        'omni_ngb': OmniNGBPredictor(encoding_type='adjacency_one_hot', config=config,
                                     zero_cost=['jacov'], lce=['sotle']),
        'omni_seminas': OmniSemiNASPredictor(encoding_type='seminas', config=config,
                                             semi=True, hpo_wrapper=False,
                                             zero_cost=['jacov'], lce=['sotle'],
                                             jacov_onehot=True),
        # omni ablation studies:
        'omni_ngb_no_lce': OmniNGBPredictor(encoding_type='adjacency_one_hot',
                                            config=config, zero_cost=['jacov'], lce=[]),
        'omni_seminas_no_lce': OmniSemiNASPredictor(encoding_type='seminas', config=config,
                                                    semi=True, hpo_wrapper=False,
                                                    zero_cost=['jacov'], lce=[],
                                                    jacov_onehot=True),
        'omni_ngb_no_zerocost': OmniNGBPredictor(encoding_type='adjacency_one_hot',
                                                 config=config, zero_cost=[], lce=['sotle']),
        'omni_ngb_no_encoding': OmniNGBPredictor(encoding_type=None, config=config,
                                                 zero_cost=['jacov'], lce=['sotle']),
        'omni_xgb': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                  zero_cost=['jacov'], lce=['sotle']),
        'omni_blr': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                  zero_cost=['jacov'], lce=['sotle'], model_type='BLR'),
        'omni_bananas': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                      zero_cost=['jacov'], lce=['sotle'], model_type='BANANAS'),
        'omni_bohamiann': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                        zero_cost=['jacov'], lce=['sotle'], model_type='BOHAMIANN'),
        'omni_bonas': OmniPredictor(encoding_type='bonas', config=config,
                                    zero_cost=['jacov'], lce=['sotle'], model_type='BONAS'),
        'omni_dngo': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                   zero_cost=['jacov'], lce=['sotle'], model_type='DNGO'),
        'omni_lgb': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                  zero_cost=['jacov'], lce=['sotle'], model_type='LGB'),
        'omni_gcn': OmniPredictor(encoding_type='gcn', config=config,
                                  zero_cost=['jacov'], lce=['sotle'], model_type='GCN'),
        'omni_rf': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                 zero_cost=['jacov'], lce=['sotle'], model_type='RF'),
        'omni_gp': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                 zero_cost=['jacov'], lce=['sotle'], model_type='GP'),
        'omni_sparsegp': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                       zero_cost=['jacov'], lce=['sotle'], model_type='SparseGP'),
        'omni_varsparsegp': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                                          zero_cost=['jacov'], lce=['sotle'], model_type='VarSparseGP'),
}

supported_search_spaces = {
        'nasbench101': NasBench101SearchSpace(),
        'nasbench201': NasBench201SearchSpace(),
        'darts': DartsSearchSpace(),
        'nlp': NasBenchNLPSearchSpace()
}

load_labeled = (True if config.search_space in ['darts', 'nlp'] else False)
dataset_api = get_dataset_api(config.search_space, config.dataset)

# initialize the search space and predictor
utils.set_seed(config.seed)
predictor = supported_predictors[config.predictor]
predictor.set_params(params=params_xgb)
search_space = supported_search_spaces[config.search_space]

    # initialize the PredictorEvaluator class
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, load_labeled=load_labeled,
                                           dataset_api=dataset_api)
predictor_evaluator.evaluate_precompute()


# Build Configuration Space which defines all parameters and their ranges.
# To illustrate different parameter types,
# we use continuous, integer and categorical parameters.
cs = ConfigurationSpace()

# We can add multiple hyperparameters at once:
max_depth = UniformIntegerHyperparameter("max_depth", 1, 15, default_value=6)
min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
colsample_bytree = UniformFloatHyperparameter('colsample_bytree', 0.0, 1.0, default_value=1.0)
colsample_bylevel = UniformFloatHyperparameter('colsample_bylevel', 0.0, 1.0, default_value=1.0)
learning_rate = UniformFloatHyperparameter('learning_rate', 0.0001, 1.0, default_value=0.1, log=True)
cs.add_hyperparameters([max_depth, min_child_weight, colsample_bytree, colsample_bylevel, learning_rate])

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                     "wallclock-limit": 82800,  # max duration to run the optimization (in seconds)
                     "cs": cs,  # configuration space
                     "deterministic": "true",
                     "limit_resources": False,  # Uses pynisher to limit memory and runtime
                     # Alternatively, you can also disable this.
                     # Then you should handle runtime and memory yourself in the TA
                     # "cutoff": 30,  # runtime limit for target algorithm
                     # "memory_limit": 3072,  # adapt this to reasonable value for your hardware
                     })

# max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
# max_iters = 2
# intensifier parameters
# intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_iters, 'eta': 3}
# To optimize, we pass the function to the SMAC-object
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=predictor_run)
# intensifier_kwargs=intensifier_kwargs)  # all arguments related to intensifier can be passed like this

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
                                      instance='1', seed=0)[1]
print("Value for default configuration: %.4f" % def_value)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                      seed=0)[1]
print("Optimized Value: %.4f" % inc_value)
# if __name__=='__main__':
#     config = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'booster': 'gbtree', 'max_depth': 6, 'min_child_weight': 1, 'colsample_bytree': 1.0, 'learning_rate': 0.1, 'colsample_bylevel': 1.0}
#     result = predictor_run(config=config)
#     print(result, flush=True)

