# This is an implementation of the BOHAMIANN predictor from the paper:
# Springenberg et al., 2016. Bayesian Optimization with Robust Bayesian Neural
# Networks

import torch.nn as nn
import numpy as np
from pybnn.bohamiann import Bohamiann, nll, get_default_network

from naslib.predictors.bnn.bnn_base import BNN
from naslib.predictors.trees.ngb import loguniform

class BOHAMIANN(BNN):

    def get_model(self, params=None, **kwargs):
        self.params = params
        predictor = Bohamiann(
            get_network=get_default_network,
            sampling_method="adaptive_sghmc",
            use_double_precision=True,
            metrics=(nn.MSELoss,),
            likelihood_function=nll,
            print_every_n_steps=10,
            normalize_input=False,
            normalize_output=True
        )
        return predictor
    @property
    def default_hyperparams(self, params=None):
        # default parameters used in Luo et al. 2020
        params = {
            'num_steps': 100,
            'num_burn_in_steps': 10,
            'keep_every': 5,
            'lr': 1e-2,
            'verbose': True
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            # evaluate the default config first during HPO
            params = self.default_hyperparams.copy()
        else:
            params = {
            'num_steps': int(np.random.uniform(low=60, high=500)),
            'num_burn_in_steps': int(np.random.uniform(low=5, high=50)),
            'keep_every': int(np.random.uniform(low=1, high=10)),
            'lr': loguniform(low=0.00001, high=0.1),
            'verbose': True
            }
        self.hyperparams = params

        return params
    def train_model(self, xtrain, ytrain):
        if self.params is not None:
            self.model.train(xtrain, ytrain,
                             num_steps=self.params['num_steps'],
                             num_burn_in_steps=self.params['num_burn_in_steps'],
                             keep_every=self.params['keep_every'],
                             lr=self.params['lr'],
                             verbose=True)
        else:
            self.model.train(xtrain, ytrain,
                         num_steps=100,
                         num_burn_in_steps=10,
                         keep_every=5,
                         lr=1e-2,
                         verbose=True)

