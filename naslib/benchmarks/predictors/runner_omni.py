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
from smac.initial_design.latin_hypercube_design import LHDesign
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
params_blr = {
    'alpha': None,
    'beta': None,
    'basis_func': None
}
params_bohamiann = {
    'num_steps': None,
    'num_burn_in_steps': None,
    'keep_every': None,
    'lr': None,
    'verbose': True
}
params_dngo = {
    'batch_size': None,
    'num_epochs': None,
    'learning_rate': None,
    'n_units_1': None,
    'n_units_2': None,
    'n_units_3': None,
    'alpha': None,
    'beta': None
}
params_gcn = {
    'gcn_hidden': None,
    'batch_size': None,
    'lr': None,
    'wd': None
}
params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'min_data_in_leaf': 5,
    'num_leaves': None,
    'learning_rate': None,
    'feature_fraction': None,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
params_bananas = {
    'num_layers': None,
    'layer_width': None,
    'batch_size': 32,
    'lr': None,
    'regularization': None
}
params_bonas = {
    'gcn_hidden': None,
    'batch_size': None,
    'lr': None
}
params_rf = {
    'n_estimators': None,
    'max_features': None,
    'min_samples_leaf': None,
    'min_samples_split': None,
    'bootstrap': False,
    'verbose': 0
}
params_gp = {
    'kernel_type': None,
    'lengthscale': None
}
params_s_gp = {
    'kernel_type': None,
    'lengthscale': None
}
params_vs_gp = {
    'kernel_type': None,
    'lengthscale': None
}
params_ngb = {
    'param:n_estimators': None,
    'param:learning_rate': None,
    'base:max_depth': None,
    'base:max_features': None,
    'base:min_samples_leaf': None,
    'base:min_samples_split': None,
}

def predictor_run(config, seed, instance, budget, **kwargs):
    zc = config['zc']
    predictor_evaluator.predictor.zero_cost = [zc]
    lc = config['lc']
    predictor_evaluator.predictor.lce = [lc]
    model_type = config['model_based_predictor']
    predictor_evaluator.predictor.model_type = model_type

    if config['model_based_predictor'] == 'XGB':
        params_xgb["max_depth"] = config["max_depth"]
        params_xgb["min_child_weight"] = config["min_child_weight"]
        params_xgb["colsample_bytree"] = config["colsample_bytree"]
        params_xgb["learning_rate"] = config["learning_rate"]
        params_xgb["colsample_bylevel"] = config["colsample_bylevel"]
        print(params_xgb, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_xgb)
        print(results, flush=True)
    if config['model_based_predictor'] == 'BLR':
        params_blr["alpha"] = config["alpha"]
        params_blr["beta"] = config["beta"]
        params_blr["basis_func"] = config["basis_func"]
        print(params_blr, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_blr)
        print(results, flush=True)
    if config['model_based_predictor'] == 'BANANAS':
        params_bananas["num_layers"] = config["num_layers"]
        params_bananas["layer_width"] = config["layer_width"]
        params_bananas["lr"] = config["lr_bananas"]
        params_bananas["regularization"] = config["regularization"]
        params_bananas['batch_size'] = config['batch_size_bananas']
        print(params_bananas, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_bananas)
        print(results, flush=True)
    if config['model_based_predictor'] == 'BOHAMIANN':
        params_bohamiann["num_steps"] = config["num_steps"]
        params_bohamiann["num_burn_in_steps"] = config["num_burn_in_steps"]
        params_bohamiann["keep_every"] = config["keep_every"]
        params_bohamiann["lr"] = config["lr_bohamiann"]
        print(params_bohamiann, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_bohamiann)
        print(results, flush=True)
    if config['model_based_predictor'] in ['BONAS', 'NAO', 'SEMINAS']:
        params_bonas["gcn_hidden"] = config["gcn_hidden"]
        params_bonas["batch_size"] = config["batch_size_bonas"]
        params_bonas["lr"] = config["lr_bonas"]
        print(params_bonas, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_bonas)
        print(results, flush=True)
    if config['model_based_predictor'] == 'DNGO':
        params_dngo["batch_size"] = config["batch_size_dngo"]
        params_dngo["num_epochs"] = config["num_epochs"]
        params_dngo["learning_rate"] = config["lr_dngo"]
        params_dngo["n_units_1"] = config["n_units_1"]
        params_dngo["n_units_2"] = config["n_units_2"]
        params_dngo["n_units_3"] = config["n_units_3"]
        params_dngo["alpha"] = config["alpha_dngo"]
        params_dngo["beta"] = config["beta_dngo"]
        print(params_dngo, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_dngo)
        print(results, flush=True)
    if config['model_based_predictor'] == 'LGB':
        params_lgb["num_leaves"] = config["num_leaves"]
        params_lgb["learning_rate"] = config["lr_lgb"]
        params_lgb["feature_fraction"] = config["feature_fraction"]

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_lgb)
        print(results, flush=True)
    if config['model_based_predictor'] == 'GCN':
        params_gcn["gcn_hidden"] = config["gcn_hidden_gcn"]
        params_gcn["batch_size"] = config["batch_size_gcn"]
        params_gcn["lr"] = config["lr_gcn"]
        params_gcn["wd"] = config["wd"]
        print(params_gcn, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_gcn)
        print(results, flush=True)
    if config['model_based_predictor'] == 'RF':
        params_rf["n_estimators"] = config["n_estimators"]
        params_rf["max_features"] = config["max_features"]
        params_rf["min_samples_leaf"] = config["min_samples_leaf"]
        params_rf["min_samples_split"] = config["min_samples_split"]

        print(params_rf, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_rf)
        print(results, flush=True)
    if config['model_based_predictor'] == 'GP':
        params_gp["kernel_type"] = config["kernel_gp"]
        params_gp["lengthscale"] = config["lengthscale_gp"]
        print(params_gp, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_gp)
        print(results, flush=True)
    if config['model_based_predictor'] == 'SparseGP':
        params_s_gp["kernel_type"] = config["kernel_s_gp"]
        params_s_gp["lengthscale"] = config["lengthscale_s_gp"]
        print(params_s_gp, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_s_gp)
        print(results, flush=True)
    if config['model_based_predictor'] == 'VarSparseGP':
        params_vs_gp["kernel_type"] = config["kernel_vs_gp"]
        params_vs_gp["lengthscale"] = config["lengthscale_vs_gp"]
        print(params_vs_gp, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_vs_gp)
        print(results, flush=True)
    if config['model_based_predictor'] == 'NGB':
        params_ngb['param:n_estimators'] = config['param:n_estimators']
        params_ngb['param:learning_rate'] = config['param:learning_rate']
        params_ngb['base:max_depth'] = config['base:max_depth']
        params_ngb['base:max_features'] = config['base:max_features']
        params_ngb['base:min_samples_leaf'] = config['base:min_samples_leaf']
        params_ngb['base:min_samples_split'] = config['base:min_samples_split']
        print(params_ngb, flush=True)

        # evaluate the predictor
        results = predictor_evaluator.evaluate(params=params_ngb)

    return 1 - results[-1]["kendalltau"]


config = utils.get_config_from_file_path(config_file='predictor_config.yaml', config_type='predictor')
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
# predictor.set_hyperparams(params=params_xgb)
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
xgb_params = []
blr_params = []
bananas_params = []
bonas_params = []
bohamiann_params = []
dngo_params = []
lgb_params = []
gcn_params = []
rf_params = []
gp_params = []
gp_s_params = []
gp_vs_params = []
ngb_params = []
########################################################################################################################
########################################################################################################################
# Omni Combination
model_based_predictor = CategoricalHyperparameter('model_based_predictor',
                                                  ['XGB', 'BLR', 'BANANAS', 'BOHAMIANN', 'BONAS',
                                                   'DNGO', 'LGB', 'GCN', 'RF', 'GP', 'SparseGP', 'VarSparseGP',
                                                   'NGB'],
                                                  default_value='XGB')
lc = CategoricalHyperparameter('lc', ['sotle', 'valacc'], default_value='sotle')
zc = CategoricalHyperparameter('zc', ['jacov', 'snip', 'synflow', 'grasp', 'fisher', 'grad_norm'],
                               default_value='jacov')
cs.add_hyperparameters([model_based_predictor, lc, zc])
########################################################################################################################
# XGB
max_depth = UniformIntegerHyperparameter("max_depth", 1, 15, default_value=6)
min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
colsample_bytree = UniformFloatHyperparameter('colsample_bytree', 0.0, 1.0, default_value=1.0)
colsample_bylevel = UniformFloatHyperparameter('colsample_bylevel', 0.0, 1.0, default_value=1.0)
learning_rate = UniformFloatHyperparameter('learning_rate', 0.0001, 1.0, default_value=0.1, log=True)
cs.add_hyperparameters([max_depth, min_child_weight, colsample_bytree, colsample_bylevel, learning_rate])
xgb_params.extend([max_depth, min_child_weight, colsample_bytree, colsample_bylevel, learning_rate])
########################################################################################################################
########################################################################################################################
# BLR
alpha = UniformFloatHyperparameter('alpha', 1e-5, 1e5, default_value=1.0)
beta = UniformFloatHyperparameter('beta', 1e-5, 1e5, default_value=100)
basis_func = CategoricalHyperparameter('basis_func', ['linear_basis_func', 'quadratic_basis_func'],
                                       default_value='linear_basis_func')
cs.add_hyperparameters([alpha, beta, basis_func])
blr_params.extend([alpha, beta, basis_func])
########################################################################################################################
########################################################################################################################
# BANANAS
num_layers = UniformIntegerHyperparameter('num_layers', 5, 25, default_value=20)
layer_width = UniformIntegerHyperparameter('layer_width', 5, 25, default_value=20)
lr_bananas = UniformFloatHyperparameter('lr_bananas', 0.00001, 0.1, log=True)
regularization = UniformFloatHyperparameter('regularization', 0, 1, default_value=0.2)
batch_size_bananas = UniformIntegerHyperparameter('batch_size_bananas', 32, 256, default_value=128, log=True)
cs.add_hyperparameters([num_layers, layer_width, lr_bananas, regularization, batch_size_bananas])
bananas_params.extend([num_layers, layer_width, lr_bananas, regularization, batch_size_bananas])
########################################################################################################################
########################################################################################################################
# BOHAMIANN
num_steps = UniformIntegerHyperparameter('num_steps', 60, 500, default_value=100)
keep_every = UniformIntegerHyperparameter('keep_every', 1, 10, default_value=5)
lr_bohamiann = UniformFloatHyperparameter('lr_bohamiann', 0.00001, 0.1, log=True)
num_burn_in_steps = UniformIntegerHyperparameter('num_burn_in_steps', 5, 50, default_value=10)
cs.add_hyperparameters([num_steps, keep_every, lr_bohamiann, num_burn_in_steps])
bohamiann_params.extend([num_steps, keep_every, lr_bohamiann, num_burn_in_steps])
########################################################################################################################
########################################################################################################################
# BONAS, NAO, SEMINAS
gcn_hidden = UniformIntegerHyperparameter('gcn_hidden', 16, 128, default_value=64, log=True)
batch_size_bonas = UniformIntegerHyperparameter('batch_size_bonas', 32, 256, default_value=128, log=True)
lr_bonas = UniformFloatHyperparameter('lr_bonas', 0.00001, 0.1, log=True)
cs.add_hyperparameters([gcn_hidden, batch_size_bonas, lr_bonas])
bonas_params.extend([gcn_hidden, batch_size_bonas, lr_bonas])
########################################################################################################################
########################################################################################################################
# DNGO
batch_size_dngo = UniformIntegerHyperparameter('batch_size_dngo', 32, 256, default_value=128, log=True)
num_epochs = UniformIntegerHyperparameter('num_epochs', 100, 1000, default_value=500)
lr_dngo = UniformFloatHyperparameter('lr_dngo', 0.00001, 0.1, default_value=0.01, log=True)
n_units_1 = UniformIntegerHyperparameter('n_units_1', 10, 100, default_value=50)
n_units_2 = UniformIntegerHyperparameter('n_units_2', 10, 100, default_value=50)
n_units_3 = UniformIntegerHyperparameter('n_units_3', 10, 100, default_value=50)
alpha_dngo = UniformFloatHyperparameter('alpha_dngo', 1e-5, 1e5, default_value=1.0)
beta_dngo = UniformFloatHyperparameter('beta_dngo', 1e-5, 1e5, default_value=100)
cs.add_hyperparameters([batch_size_dngo, num_epochs, lr_dngo, n_units_1, n_units_2, n_units_3, alpha_dngo, beta_dngo])
dngo_params.extend([batch_size_dngo, num_epochs, lr_dngo, n_units_1, n_units_2, n_units_3, alpha_dngo, beta_dngo])
########################################################################################################################
########################################################################################################################
# LGB
num_leaves = UniformIntegerHyperparameter('num_leaves', 10, 100, default_value=81)
lr_lgb = UniformFloatHyperparameter('lr_lgb', 0.00001, 0.5, default_value=0.5, log=True)
feature_fraction = UniformFloatHyperparameter('feature_fraction', 0.1, 1, default_value=0.9)
cs.add_hyperparameters([num_leaves, lr_lgb, feature_fraction])
lgb_params.extend([num_leaves, lr_lgb, feature_fraction])
########################################################################################################################
########################################################################################################################
# GCN
gcn_hidden_gcn = UniformIntegerHyperparameter('gcn_hidden_gcn', 64, 200, default_value=64, log=True)
batch_size_gcn = UniformIntegerHyperparameter('batch_size_gcn', 5, 32, default_value=7, log=True)
lr_gcn = UniformFloatHyperparameter('lr_gcn', 0.00001, 0.1, default_value=0.0001, log=True)
wd = UniformFloatHyperparameter('wd', 0.00001, 0.1, default_value=3e-4, log=True)
cs.add_hyperparameters([gcn_hidden_gcn, batch_size_gcn, lr_gcn, wd])
gcn_params.extend([gcn_hidden_gcn, batch_size_gcn, lr_gcn, wd])
########################################################################################################################
########################################################################################################################
# RF
n_estimators = UniformIntegerHyperparameter('n_estimators', 16, 128, default_value=116, log=True)
max_features = UniformFloatHyperparameter('max_features', 0.1, 0.9, default_value=0.17055852159745608, log=True)
min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, default_value=2)
min_samples_split = UniformIntegerHyperparameter('min_samples_split', 2, 20, default_value=2)
cs.add_hyperparameters([n_estimators, max_features, min_samples_leaf, min_samples_split])
rf_params.extend([n_estimators, max_features, min_samples_leaf, min_samples_split])
########################################################################################################################
########################################################################################################################
# GP
kernel_gp = CategoricalHyperparameter('kernel_gp', ['RBF', 'Matern32', 'Matern52'], default_value='RBF')
lengthscale_gp = UniformFloatHyperparameter('lengthscale_gp', 1e-5, 1e5, default_value=10)
cs.add_hyperparameters([kernel_gp, lengthscale_gp])
gp_params.extend([kernel_gp, lengthscale_gp])
########################################################################################################################
########################################################################################################################
# Sparse GP
kernel_s_gp = CategoricalHyperparameter('kernel_s_gp', ['RBF', 'Matern32', 'Matern52'], default_value='RBF')
lengthscale_s_gp = UniformFloatHyperparameter('lengthscale_s_gp', 1e-5, 1e5, default_value=10)
cs.add_hyperparameters([kernel_s_gp, lengthscale_s_gp])
gp_s_params.extend([kernel_s_gp, lengthscale_s_gp])
########################################################################################################################
########################################################################################################################
# Var Sparse GP
kernel_vs_gp = CategoricalHyperparameter('kernel_vs_gp', ['RBF', 'Matern32', 'Matern52'], default_value='RBF')
lengthscale_vs_gp = UniformFloatHyperparameter('lengthscale_vs_gp', 1e-5, 1e5, default_value=10)
cs.add_hyperparameters([kernel_vs_gp, lengthscale_vs_gp])
gp_vs_params.extend([kernel_vs_gp, lengthscale_vs_gp])
########################################################################################################################
########################################################################################################################
# NGB
n_estimators_ngb = UniformIntegerHyperparameter("param:n_estimators", 128, 512, default_value=128, log=True)
learning_rate_ngb = UniformFloatHyperparameter("param:learning_rate", 0.001, 0.1, default_value=0.001, log=True)
max_depth_ngb = UniformIntegerHyperparameter('base:max_depth', 1, 25, default_value=1.0)
max_features_ngb = UniformFloatHyperparameter('base:max_features', 0.1, 1.0, default_value=0.1)
min_samples_leaf_ngb = UniformIntegerHyperparameter('base:min_samples_leaf', 2, 20, default_value=2)
min_samples_split_ngb = UniformIntegerHyperparameter('base:min_samples_split', 2, 20, default_value=2)
cs.add_hyperparameters([n_estimators_ngb, learning_rate_ngb, max_depth_ngb, max_features_ngb, min_samples_leaf_ngb,
                        min_samples_split_ngb])
ngb_params.extend([n_estimators_ngb, learning_rate_ngb, max_depth_ngb, max_features_ngb, min_samples_leaf_ngb,
                   min_samples_split_ngb])
########################################################################################################################
########################################################################################################################
# Conditions
for param in xgb_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'XGB'))
for param in blr_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'BLR'))
for param in bananas_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'BANANAS'))
for param in bohamiann_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'BOHAMIANN'))
for param in bonas_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'BONAS'))
    # cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'NAO'))
    # cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'SEMINAS'))
for param in dngo_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'DNGO'))
for param in lgb_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'LGB'))
for param in gcn_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'GCN'))
for param in rf_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'RF'))
for param in gp_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'GP'))
for param in gp_s_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'SparseGP'))
for param in gp_vs_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'VarSparseGP'))
for param in gp_vs_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'VarSparseGP'))
for param in ngb_params:
    cs.add_condition(CS.EqualsCondition(param, model_based_predictor, 'NGB'))
########################################################################################################################


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
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(1),
                tae_runner=predictor_run, initial_design=LHDesign)
# intensifier_kwargs=intensifier_kwargs)  # all arguments related to intensifier can be passed like this

# Example call of the function with default values
# It returns: Status, Cost, Runtime, Additional Infos
def_value = smac.get_tae_runner().run(config=cs.get_default_configuration(),
                                      instance='1', seed=1)[1]
print("Value for default configuration: %.4f" % def_value)

# Start optimization
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',
                                      seed=1)[1]
print("Optimized Value: %.4f" % inc_value)
# if __name__=='__main__':
#     config = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'booster': 'gbtree', 'max_depth': 6, 'min_child_weight': 1, 'colsample_bytree': 1.0, 'learning_rate': 0.1, 'colsample_bylevel': 1.0}
#     result = predictor_run(config=config)
#     print(result, flush=True)
