import logging
import sys
import os
import naslib as nl

from naslib.defaults.predictor_evaluator import PredictorEvaluator

from naslib.predictors import BayesianLinearRegression, BOHAMIANN, BonasPredictor, \
DNGOPredictor, EarlyStopping, Ensemble, GCNPredictor, GPPredictor, \
LCEPredictor, LCEMPredictor, LGBoost, MLPPredictor, NGBoost, OmniNGBPredictor, \
OmniSemiNASPredictor, RandomForestPredictor, SVR_Estimator, SemiNASPredictor, \
SoLosspredictor, SparseGPPredictor, VarSparseGPPredictor, XGBoost, ZeroCostV1, \
ZeroCostV2, GPWLPredictor, OmniPredictor, GATPredictor

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, \
DartsSearchSpace, NasBenchNLPSearchSpace

from naslib.utils import utils, setup_logger, get_dataset_api
from naslib.utils.utils import get_project_root


config = utils.get_config_from_args(config_type='predictor')
utils.set_seed(config.seed)
logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
utils.log_args(config)

supported_predictors = {
    'bananas': Ensemble(predictor_type='bananas', num_ensemble=3, hpo_wrapper=True),
    'bayes_lin_reg': BayesianLinearRegression(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'bohamiann': BOHAMIANN(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'bonas': BonasPredictor(encoding_type='bonas', hpo_wrapper=True),
    'dngo': DNGOPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'fisher': ZeroCostV2(config, batch_size=64, method_type='fisher'),
    'gcn': GCNPredictor(encoding_type='gcn', hpo_wrapper=True),
    'gp': GPPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'gpwl': GPWLPredictor(ss_type=config.search_space, kernel_type='wloa', optimize_gp_hyper=True, h='auto'),
    'grad_norm': ZeroCostV2(config, batch_size=64, method_type='grad_norm'),
    'grasp': ZeroCostV2(config, batch_size=64, method_type='grasp'),
    'jacov': ZeroCostV1(config, batch_size=64, method_type='jacov'),
    'lce': LCEPredictor(metric=Metric.VAL_ACCURACY),
    'lce_m': LCEMPredictor(metric=Metric.VAL_ACCURACY),
    'lcsvr': SVR_Estimator(metric=Metric.VAL_ACCURACY, all_curve=False,
                           require_hyper=False),
    'lgb': LGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'mlp': MLPPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'nao': SemiNASPredictor(encoding_type='seminas', semi=False, hpo_wrapper=True),
    'ngb': NGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'rf': RandomForestPredictor(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'seminas': SemiNASPredictor(encoding_type='seminas', semi=True, hpo_wrapper=True),
    'snip': ZeroCostV2(config, batch_size=64, method_type='snip'),
    'sotl': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTL'),
    'sotle': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLE'),
    'sotlema': SoLosspredictor(metric=Metric.TRAIN_LOSS, sum_option='SoTLEMA'),
    'sparse_gp': SparseGPPredictor(encoding_type='adjacency_one_hot',
                                   optimize_gp_hyper=True, num_steps=100, hpo_wrapper=True),
    'synflow': ZeroCostV2(config, batch_size=64, method_type='synflow'),
    'valacc': EarlyStopping(metric=Metric.VAL_ACCURACY),
    'valloss': EarlyStopping(metric=Metric.VAL_LOSS),
    'var_sparse_gp': VarSparseGPPredictor(encoding_type='adjacency_one_hot',
                                          optimize_gp_hyper=True, num_steps=200, hpo_wrapper=True),
    'xgb': XGBoost(encoding_type='adjacency_one_hot', hpo_wrapper=True),
    'gat': GATPredictor(encoding_type='gat', hpo_wrapper=True),
    # path encoding experiments:
    'bayes_lin_reg_path': BayesianLinearRegression(encoding_type='path'),
    'bohamiann_path': BOHAMIANN(encoding_type='path'),
    'dngo_path': DNGOPredictor(encoding_type='path'),
    'gp_path': GPPredictor(encoding_type='path'),
    'lgb_path': LGBoost(encoding_type='path', hpo_wrapper=True),
    'ngb_path': NGBoost(encoding_type='path', hpo_wrapper=True),
    # omni:
    'omni_ngb': OmniNGBPredictor(encoding_type='adjacency_one_hot', config=config,
                                 zero_cost=['jacov'], lce=['sotle'], hpo_wrapper=False),
    'omni_seminas': OmniSemiNASPredictor(encoding_type='seminas', config=config,
                                         semi=True, hpo_wrapper=True,
                                         zero_cost=['jacov'], lce=['sotle'],
                                         jacov_onehot=True),
    # omni ablation studies:
    'omni_ngb_no_lce': OmniNGBPredictor(encoding_type='adjacency_one_hot',
                                        config=config, zero_cost=['jacov'], lce=[]),
    'omni_seminas_no_lce': OmniSemiNASPredictor(encoding_type='seminas', config=config,
                                                semi=True, hpo_wrapper=True,
                                                zero_cost=['jacov'], lce=[],
                                                jacov_onehot=True),
    'omni_ngb_no_zerocost': OmniNGBPredictor(encoding_type='adjacency_one_hot',
                                             config=config, zero_cost=[], lce=['sotle']),
    'omni_ngb_no_encoding': OmniNGBPredictor(encoding_type=None, config=config,
                                             zero_cost=['jacov'], lce=['sotle']),
    'omni_xgb': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                              zero_cost=['jacov'], lce=['sotle'], hpo_wrapper=True),
    'omni_lgb': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                              zero_cost=['grad_norm'], lce=['sotle'], hpo_wrapper=True),
    'omni_ngb2': OmniPredictor(encoding_type='adjacency_one_hot', config=config,
                              zero_cost=['grad_norm'], lce=['sotle'], model_type='NGB', hpo_wrapper=True),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace(),
    'nlp': NasBenchNLPSearchSpace()
}

"""
If the API did not evaluate *all* architectures in the search space, 
set load_labeled=True
"""
load_labeled = (True if config.search_space in ['darts', 'nlp'] else False)
dataset_api = get_dataset_api(config.search_space, config.dataset)

# initialize the search space and predictor
predictor = supported_predictors[config.predictor]
params_ngb = {
    "base:max_depth": 10,
    "base:max_features": 0.7394770722643155,
    "base:min_samples_leaf": 1,
    "base:min_samples_split": 2,
    "param:learning_rate": 0.006724595976491001,
    "param:n_estimators": 236
        }
# predictor.set_hyperparams(params=params_ngb)
search_space = supported_search_spaces[config.search_space]

# initialize the PredictorEvaluator class
predictor_evaluator = PredictorEvaluator(predictor, config=config)
predictor_evaluator.adapt_search_space(search_space, load_labeled=load_labeled,
                                       dataset_api=dataset_api)

# evaluate the predictor
predictor_evaluator.evaluate_precompute()
predictor_evaluator.evaluate()
