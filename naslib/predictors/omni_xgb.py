import time
import numpy as np
import copy
import logging
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from ngboost.distns import Normal
from ngboost.scores import LogScore
from naslib.predictors.trees.ngb import loguniform
from naslib.predictors.trees import BaseTree
from naslib.predictors.predictor import Predictor
from naslib.predictors.lcsvr import loguniform
from naslib.predictors.zerocost_v1 import ZeroCostV1
from naslib.predictors.zerocost_v2 import ZeroCostV2
from naslib.predictors.trees.xgb import XGBoost
from naslib.predictors import BayesianLinearRegression as BLR
from naslib.predictors import Ensemble, BOHAMIANN, BonasPredictor, DNGOPredictor, LGBoost, GCNPredictor,\
    RandomForestPredictor, GPPredictor, SparseGPPredictor, VarSparseGPPredictor
from naslib.predictors.utils.encodings import encode
from naslib.utils import utils
from naslib.search_spaces.core.query_metrics import Metric
from naslib.predictors.omni_seminas import convert_arch_to_seq

logger = logging.getLogger(__name__)


class OmniPredictor(Predictor):

    def __init__(self, zero_cost, lce, encoding_type, ss_type=None, config=None,
                 n_hypers=35, run_pre_compute=True, min_train_size=0, max_zerocost=np.inf, model_type='XGB',
                 hpo_wrapper=False):
        self.zero_cost_all = ['jacov', 'snip', 'synflow', 'grasp', 'fisher', 'grad_norm']
        self.zero_cost = zero_cost
        self.encoding_type = encoding_type
        self.config = config
        self.n_hypers = n_hypers
        self.config = config
        self.lce = lce
        self.lce_all = ['sotle', 'valacc']
        self.ss_type = ss_type
        self.run_pre_compute = run_pre_compute
        self.min_train_size = min_train_size
        self.max_zerocost = max_zerocost
        self.model_type = model_type
        self.hyperparams = None
        self.hpo_wrapper = hpo_wrapper

    def set_hyperparams(self, params=None):
        self.hyperparams = params
    

    def pre_compute(self, xtrain, xtest, unlabeled_data=None):
        """
        All of this computation could go into fit() and query(), but we do it
        here to save time, so that we don't have to re-compute Jacobian covariances
        for all train_sizes when running experiment_types that vary train size or fidelity.
        """
        self.xtrain_zc_info = {}
        self.xtest_zc_info = {}

        if len(self.zero_cost_all) > 0:
            self.train_loader, _, _, _, _ = utils.get_train_val_loaders(self.config, mode='train')

            for method_name in self.zero_cost_all:
                if method_name == 'jacov':
                    zc_method = ZeroCostV1(self.config, batch_size=64, method_type=method_name)
                else:
                    zc_method = ZeroCostV2(self.config, batch_size=64, method_type=method_name)
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                xtrain_zc_scores = zc_method.query(xtrain)
                xtest_zc_scores = zc_method.query(xtest)

                train_mean = np.mean(np.array(xtrain_zc_scores))
                train_std = np.std((np.array(xtrain_zc_scores)))

                normalized_train = (np.array(xtrain_zc_scores) - train_mean) / train_std
                normalized_test = (np.array(xtest_zc_scores) - train_mean) / train_std

                self.xtrain_zc_info[f'{method_name}_scores'] = normalized_train
                self.xtest_zc_info[f'{method_name}_scores'] = normalized_test


    def get_random_params(self):
        params_xgb = {
            'objective': 'reg:squarederror',
            'eval_metric': "rmse",
            # 'early_stopping_rounds': 100,
            'booster': 'gbtree',
            'max_depth': int(np.random.choice(range(1, 15))),
            'min_child_weight': int(np.random.choice(range(1, 10))),
            'colsample_bytree': np.random.uniform(.0, 1.0),
            'learning_rate': loguniform(.001, .5),
            # 'alpha': 0.24167936088332426,
            # 'lambda': 31.393252465064943,
            'colsample_bylevel': np.random.uniform(.0, 1.0),
        }
        params_lgb = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'min_data_in_leaf': 5,
            'num_leaves': int(np.random.choice(90) + 10),
            'learning_rate': loguniform(.001, .1),
            'feature_fraction': np.random.uniform(.1, 1),
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        params_gcn = {
            'gcn_hidden': int(loguniform(64, 200)),
            'batch_size': int(loguniform(5, 32)),
            'lr': loguniform(.00001, .1),
            'wd': loguniform(.00001, .1)
        }
        params_bananas = {
                'num_layers': int(np.random.choice(range(5,25))),
                'layer_width': int(np.random.choice(range(5,25))),
                'batch_size': 32,
                'lr': np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
                'regularization': 0.2
        }
        params_bonas = {
            'gcn_hidden': int(loguniform(16, 128)),
            'batch_size': int(loguniform(32, 256)),
            'lr': loguniform(.00001, .1)
        }
        params_rf = {
            'n_estimators': int(loguniform(16, 128)),
            'max_features': loguniform(.1, .9),
            'min_samples_leaf': int(np.random.choice(19) + 1),
            'min_samples_split': int(np.random.choice(18) + 2),
            'bootstrap': False,
            # 'verbose': -1
        }
        if self.model_type == 'XGB':
            return params_xgb
        elif self.model_type in ['BLR', 'BOHAMIANN', 'DNGO', 'GP', 'SparseGP', 'VarSparseGP']:
            params = None
        elif self.model_type == 'BANANAS':
            return params_bananas
        elif self.model_type == 'BONAS':
            return params_bonas
        elif self.model_type == 'LGB':
            return params_lgb
        elif self.model_type == 'GCN':
            return params_gcn
        elif self.model_type == 'RF':
            return params_rf
        return params


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
            normalized = (train_losses - mean) / std
            full_xdata = [[*x, normalized[i]] for i, x in enumerate(full_xdata)]

        elif 'sotle' in self.lce and len(info[0]['TRAIN_LOSS_lc']) < 3:
            logger.info('Not enough fidelities to use train loss')

        if 'valacc' in self.lce and len(info[0]['VAL_ACCURACY_lc']) >= 3:
            val_accs = [lcs['VAL_ACCURACY_lc'][-1] for lcs in info]
            mean = np.mean(val_accs)
            std = np.std(val_accs)
            normalized = (val_accs - mean) / std
            full_xdata = [[*x, normalized[i]] for i, x in enumerate(full_xdata)]

        if self.encoding_type is not None:
            xdata_encoded = [encode(arch, encoding_type=self.encoding_type,
                                             ss_type=self.ss_type) for arch in xdata]
            if self.encoding_type == 'bonas' or self.encoding_type == 'gcn':
                for i, x_data in enumerate(xdata_encoded):
                    for ops in x_data['operations']:
                        np.append(ops, full_xdata[i][0])
                        np.append(ops, full_xdata[i][1])
                return np.array(xdata_encoded)
            elif self.encoding_type == 'seminas':
                xdata_encoded = []
                for i, arch in enumerate(xdata):
                    encoded = encode(arch, encoding_type=self.encoding_type,
                                 ss_type=self.ss_type)
                    seq = convert_arch_to_seq(encoded['adjacency'],
                                          encoded['operations'],
                                          max_n=self.max_n)
                    xdata_encoded.append(seq)

            full_xdata = [[*x, *xdata_encoded[i]] for i, x in enumerate(full_xdata)]

        return np.array(full_xdata)

    def fit(self, xtrain, ytrain, train_info, learn_hyper=True):
        if self.model_type == 'BONAS':
            self.encoding_type = 'bonas'
        elif self.model_type == 'GCN':
            self.encoding_type = 'gcn'
        else:
            self.encoding_type = 'adjacency_one_hot'

        # if we are below the min train size, use the zero_cost and lce info
        if len(xtrain) < self.min_train_size:
            self.trained = False
            return None
        self.trained = True
        self.train_size = len(xtrain)

        # prepare training data labels
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)
        if self.model_type not in ['BONAS', 'LGB', 'XGB', 'GCN', 'RF', 'GP', 'SparseGP', 'VarSparseGP']:
            print('DATA IS NORMALIZED')
            ytrain = (np.array(ytrain) - self.mean) / self.std
        xtrain = self.prepare_features(xtrain, train_info, train=True)
        if self.hyperparams is not None:
            params = self.hyperparams
        else:
            params = self.get_random_params()

        if self.model_type ==  'XGB':
            self.model = XGBoost(hyperparams=params)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, params=params)
        elif self.model_type == 'BLR':
            self.model = BLR()
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, params=params, omni=True)
        elif self.model_type == 'BANANAS':
            self.model = Ensemble(predictor_type='bananas', num_ensemble=3, hpo_wrapper=False, hyperparams=params)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, omni=True)
        elif self.model_type == 'BOHAMIANN':
            self.model = BOHAMIANN()
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, params=params, omni=True)
        elif self.model_type == 'BONAS':
            self.model = BonasPredictor(hyperparams=params)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, omni=True)
        elif self.model_type == 'DNGO':
            self.model = DNGOPredictor()
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, params=params, omni=True)
        elif self.model_type == 'LGB':
            self.model = LGBoost(hyperparams=params)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, params=params)
        elif self.model_type == 'GCN':
            self.model = GCNPredictor(hyperparams=params, ss_type=self.ss_type)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, omni=True)
        elif self.model_type == 'RF':
            self.model = RandomForestPredictor(hyperparams=params)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, params=params)
        #TODO: Optimize GP hyperparams, maybe Adam lr?
        elif self.model_type == 'GP':
            self.model = GPPredictor(kernel_type=params['kernel'], lengthscale=params['lengthscale']
                                     , optimize_gp_hyper=True)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, omni=True)
        elif self.model_type == 'SparseGP':
            self.model = SparseGPPredictor(kernel_type=params['kernel'], lengthscale=params['lengthscale']
                                           , optimize_gp_hyper=True)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, omni=True)
        elif self.model_type == 'VarSparseGP':
            self.model = VarSparseGPPredictor(kernel_type=params['kernel'], lengthscale=params['lengthscale']
                                              , optimize_gp_hyper=True)
            self.model.set_hyperparams(hyperparams=params)
            self.model.fit(xtrain, ytrain, omni=True)
        print("EID AL-ADHA: %s" % self.model.hyperparams)

    def query(self, xtest, info):
        if self.trained:
            test_data = self.prepare_features(xtest, info, train=False)
            if self.model_type in ['XGB', 'LGB', 'RF']:
                return np.squeeze(self.model.predict(test_data)) * self.std + self.mean
            elif self.model_type in ['BLR', 'BOHAMIANN', 'DNGO', 'BANANAS']:
                return np.squeeze(self.model.query(test_data, omni=True)) * self.std + self.mean
            elif self.model_type in ['BONAS', 'GCN', 'GP', 'SparseGP', 'VarSparseGP']:
                return np.squeeze(self.model.query(test_data, omni=True))
        else:
            logger.info('below the train size, so returning info')
            return info

    def get_data_reqs(self):
        """
        Returns a dictionary with info about whether the predictor needs
        extra info to train/query.
        """
        if len(self.lce_all) > 0:
            # add the metrics needed for the lce predictors
            required_metric_dict = {'sotle': Metric.TRAIN_LOSS, 'valacc': Metric.VAL_ACCURACY}
            self.metric = [required_metric_dict[key] for key in self.lce_all]

            reqs = {'requires_partial_lc': True,
                    'metric': self.metric,
                    'requires_hyperparameters': False,
                    'hyperparams': {},
                    'unlabeled': False,
                    'unlabeled_factor': 0
                    }
        else:
            reqs = super().get_data_reqs()

        return reqs
