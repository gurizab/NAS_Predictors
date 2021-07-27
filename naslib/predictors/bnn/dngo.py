# This is an implementation of the DNGO predictor from the paper:
# Snoek et al., 2015. Scalable Bayesian Optimization using DNNs

from pybnn.dngo import DNGO
import numpy as np
from naslib.predictors.bnn.bnn_base import BNN
from naslib.predictors.trees.ngb import loguniform


class DNGOPredictor(BNN):
    @property
    def default_hyperparams(self):
        params = {
            'batch_size': 128,
            'num_epochs': 500,
            'learning_rate': 0.01,
            'n_units_1': 50,
            'n_units_2': 50,
            'n_units_3': 50,
            'alpha': 1.0,
            'beta': 100,
            # 'early_stopping_rounds': 100,
            # 'verbose': -1
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()
        else:
            params = {
                'batch_size': int(np.random.uniform(low=32, high=256)),
                'num_epochs': int(np.random.uniform(low=100, high=1000)),
                'learning_rate': loguniform(low=0.00001, high=0.1),
                'n_units_1': int(np.random.uniform(low=10, high=100)),
                'n_units_2': int(np.random.uniform(low=10, high=100)),
                'n_units_3': int(np.random.uniform(low=10, high=100)),
                'alpha': np.random.uniform(low=1e-5, high=1e5),
                'beta': np.random.uniform(low=1e-5, high=1e5),
            }
        self.set_hyperparams(params)
        return params

    def get_model(self, params=None, **kwargs):
        if params is not None:
            predictor = DNGO(
                batch_size=params['batch_size'],
                num_epochs=params['num_epochs'],
                learning_rate=params['learning_rate'],
                adapt_epoch=5000,
                n_units_1=params['n_units_1'],
                n_units_2=params['n_units_2'],
                n_units_3=params['n_units_3'],
                alpha=params['alpha'],
                beta=params['beta'],
                prior=None,
                do_mcmc=True,  # turn this off for better sample efficiency
                n_hypers=20,
                chain_length=2000,
                burnin_steps=2000,
                normalize_input=False,
                normalize_output=True
            )
        else:
            if self.hyperparams is not None:
                params = self.hyperparams
                predictor = DNGO(
                    batch_size=params['batch_size'],
                    num_epochs=params['num_epochs'],
                    learning_rate=params['learning_rate'],
                    adapt_epoch=5000,
                    n_units_1=params['n_units_1'],
                    n_units_2=params['n_units_2'],
                    n_units_3=params['n_units_3'],
                    alpha=params['alpha'],
                    beta=params['beta'],
                    prior=None,
                    do_mcmc=True,  # turn this off for better sample efficiency
                    n_hypers=20,
                    chain_length=2000,
                    burnin_steps=2000,
                    normalize_input=False,
                    normalize_output=True
                )
            else:
                predictor = DNGO(
                    batch_size=10,
                    num_epochs=500,
                    learning_rate=0.01,
                    adapt_epoch=5000,
                    n_units_1=50,
                    n_units_2=50,
                    n_units_3=50,
                    alpha=1.0,
                    beta=1000,
                    prior=None,
                    do_mcmc=True, # turn this off for better sample efficiency
                    n_hypers=20,
                    chain_length=2000,
                    burnin_steps=2000,
                    normalize_input=False,
                    normalize_output=True
                )
        return predictor

    def train_model(self, xtrain, ytrain):
        try:
            self.model.train(xtrain, ytrain, do_optimize=True)
        except ValueError:
            self.model.train(xtrain, ytrain, do_optimize=False)

