import time
import numpy as np
import copy
import logging
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore

import ConfigSpace as CS
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from functools import partial
from naslib.predictors.predictor import Predictor
from naslib.predictors.lcsvr import loguniform
from naslib.predictors.zerocost_v1 import ZeroCostV1
from naslib.predictors.utils.encodings import encode
from naslib.utils import utils
from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)

def parse_params(params, identifier):
    to_return = {}
    for k, v in params.items():
        if k.startswith(identifier):
            to_return[k.replace(identifier, "")] = v

    return to_return


class OmniNGBPredictor(Predictor):

    def __init__(self, zero_cost, lce, encoding_type, ss_type=None, config=None, 
                 n_hypers=35, run_pre_compute=True, min_train_size=0, max_zerocost=np.inf, hpo_wrapper=False):
        
        self.zero_cost = zero_cost
        self.encoding_type = encoding_type
        self.config = config
        self.n_hypers = n_hypers
        self.config = config
        self.lce = lce
        self.ss_type = ss_type
        self.run_pre_compute = run_pre_compute
        self.min_train_size = min_train_size
        self.max_zerocost = max_zerocost
        self.hpo_wrapper = hpo_wrapper

    def pre_compute(self, xtrain, xtest, unlabeled_data=None):
        """
        All of this computation could go into fit() and query(), but we do it
        here to save time, so that we don't have to re-compute Jacobian covariances
        for all train_sizes when running experiment_types that vary train size or fidelity.        
        """
        self.xtrain_zc_info = {}
        self.xtest_zc_info = {}

        if len(self.zero_cost) > 0:
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

            for method_name in self.zero_cost:
                zc_method = ZeroCostV1(self.config, batch_size=64, method_type=method_name)
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                xtrain_zc_scores = zc_method.query(xtrain)
                xtest_zc_scores = zc_method.query(xtest)
                
                train_mean = np.mean(np.array(xtrain_zc_scores)) 
                train_std = np.std((np.array(xtrain_zc_scores)))
                
                normalized_train = (np.array(xtrain_zc_scores) - train_mean)/train_std
                normalized_test = (np.array(xtest_zc_scores) - train_mean)/train_std
                
                self.xtrain_zc_info[f'{method_name}_scores'] = normalized_train
                self.xtest_zc_info[f'{method_name}_scores'] = normalized_test

    def get_random_params(self):
        params = {
            'param:n_estimators': int(loguniform(128, 512)),
            'param:learning_rate': loguniform(.001, .1),
            'param:minibatch_frac': np.random.uniform(.1, 1),
            'base:max_depth': np.random.choice(24) + 1,
            'base:max_features': np.random.uniform(.1, 1),
            'base:min_samples_leaf': np.random.choice(18) + 2,
            'base:min_samples_split': np.random.choice(18) + 2,
        }
        return params
    def get_configspace(self):
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("param:n_estimators", 128, 512, default_value=128, log=True)
        learning_rate = UniformFloatHyperparameter("param:learning_rate", 0.001, 0.1, default_value=0.001, log=True)
        minibatch_frac = UniformFloatHyperparameter('param:minibatch_frac', 0.1, 1.0, default_value=1.0)
        max_depth = UniformIntegerHyperparameter('base:max_depth', 1, 25, default_value=1.0)
        max_features = UniformFloatHyperparameter('base:max_features', 0.1, 1.0, default_value=0.1)
        min_samples_leaf = UniformIntegerHyperparameter('base:min_samples_leaf', 2, 20, default_value=2)
        min_samples_split = UniformIntegerHyperparameter('base:min_samples_split', 2, 20, default_value=2)
        cs.add_hyperparameters([n_estimators, learning_rate, minibatch_frac, max_depth, max_features, min_samples_leaf,
                                min_samples_split])

        return cs

    def run_hpo(self, xtrain, ytrain, smac=True):
        if not smac:
            min_score = 100000
            best_params = None
            for i in range(self.n_hypers):
                params = self.get_random_params()
                for key in ['base:min_samples_leaf', 'base:min_samples_split']:
                    params[key] = max(2, min(params[key], int(len(xtrain)/3)-1))
            
                score = self.cross_validate(xtrain, ytrain, params)
                if score < min_score:
                    min_score = score
                    best_params = params
                    logger.info('{} new best {}, {}'.format(i, score, params))
        else:
            cs = self.get_configspace()
            # SMAC scenario object
            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                                 "runcount-limit": self.n_hypers,
                                 "cs": cs,  # configuration space
                                 "deterministic": "true",
                                 "limit_resources": False,
                                 })
            smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                            tae_runner=partial(self.cross_validate, xtrain, ytrain))
            # Start optimization
            try:
                best_params = smac.optimize()
            finally:
                best_params = smac.solver.incumbent
                print("Best Params: %s" % best_params)
        return best_params.get_dictionary()
        
    def cross_validate(self, xtrain, ytrain, params):
        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **parse_params(params.get_dictionary(), 'base:'))
        model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                             verbose=False, **parse_params(params.get_dictionary(), 'param:'))
        scores = cross_val_score(model, xtrain, ytrain, cv=3)
        return np.mean(scores)

    def prepare_features(self, xdata, info, train=True):
        # prepare training data features
        full_xdata = [[] for _ in range(len(xdata))]
        if len(self.zero_cost) > 0 and self.train_size <= self.max_zerocost: 
            if self.run_pre_compute:
                for key in self.xtrain_zc_info:
                    if train:
                        full_xdata = [[*x, self.xtrain_zc_info[key][i]] for i, x in enumerate(full_xdata)]
                    else:
                        full_xdata = [[*x, self.xtest_zc_info[key][i]] for i, x in enumerate(full_xdata)]
            else:
                # if the zero_cost scores were not precomputed, they are in info
                full_xdata = [[*x, info[i]] for i, x in enumerate(full_xdata)]

        if 'sotle' in self.lce and len(info[0]['TRAIN_LOSS_lc']) >= 3:
            train_losses = np.array([lcs['TRAIN_LOSS_lc'][-1] for lcs in info])
            mean = np.mean(train_losses)
            std = np.std(train_losses)
            normalized = (train_losses - mean)/std
            full_xdata = [[*x, normalized[i]] for i, x in enumerate(full_xdata)]
            
        elif 'sotle' in self.lce and len(info[0]['TRAIN_LOSS_lc']) < 3:
            logger.info('Not enough fidelities to use train loss')

        if 'valacc' in self.lce and len(info[0]['VAL_ACCURACY_lc']) >= 3:
            val_accs = [lcs['VAL_ACCURACY_lc'][-1] for lcs in info]
            mean = np.mean(val_accs)
            std = np.std(val_accs)
            normalized = (val_accs - mean)/std
            full_xdata = [[*x, normalized[i]] for i, x in enumerate(full_xdata)]

        if self.encoding_type is not None:
            xdata_encoded = np.array([encode(arch, encoding_type=self.encoding_type,
                                             ss_type=self.ss_type) for arch in xdata])            
            full_xdata = [[*x, *xdata_encoded[i]] for i, x in enumerate(full_xdata)]

        return np.array(full_xdata)
        
    def fit(self, xtrain, ytrain, train_info, learn_hyper=True):

        # if we are below the min train size, use the zero_cost and lce info
        if len(xtrain) < self.min_train_size:
            self.trained = False
            return None
        self.trained = True
        self.train_size = len(xtrain)

        # prepare training data labels
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        ytrain = (np.array(ytrain)-self.mean)/self.std
        xtrain = self.prepare_features(xtrain, train_info, train=True)
        params = self.hyperparams

        # todo: this code is repeated in cross_validate
        base_learner = DecisionTreeRegressor(criterion='friedman_mse',
                                             random_state=None,
                                             splitter='best',
                                             **parse_params(params, 'base:'))
        self.model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore,
                                  verbose=True, **parse_params(params, 'param:'))
        self.model.fit(xtrain, ytrain)

    def query(self, xtest, info):
        if self.trained:
            test_data = self.prepare_features(xtest, info, train=False)
            return np.squeeze(self.model.predict(test_data)) * self.std + self.mean
        else:
            logger.info('below the train size, so returning info')
            return info
    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        if len(self.lce) > 0:
            # add the metrics needed for the lce predictors
            required_metric_dict = {'sotle':Metric.TRAIN_LOSS, 'valacc':Metric.VAL_ACCURACY}
            self.metric = [required_metric_dict[key] for key in self.lce]

            reqs = {'requires_partial_lc':True, 
                    'metric':self.metric, 
                    'requires_hyperparameters':False, 
                    'hyperparams':{}, 
                    'unlabeled':False, 
                    'unlabeled_factor':0
                   }
        else:
            reqs = super().get_data_reqs()

        return reqs
