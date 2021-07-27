# This is an implementation of Bayesian Linear Regression

from pybnn.bayesian_linear_regression import BayesianLinearRegression as BLR
from pybnn.bayesian_linear_regression import linear_basis_func, quadratic_basis_func
import numpy as np
from naslib.predictors.bnn.bnn_base import BNN
from naslib.predictors.trees.ngb import loguniform


class BayesianLinearRegression(BNN):
    @property
    def default_hyperparams(self):
        params = {
            'basis_func': 'linear_basis_func',
            'alpha': 1.0,
            'beta': 100,
        }
        return params

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()
        else:
            params = {
                'basis_func': np.random.choice(['linear_basis_func', 'quadratic_basis_func']),
                'alpha': np.random.uniform(low=1e-5, high=1e5),
                'beta': np.random.uniform(low=1e-5, high=1e5),
            }
        self.set_hyperparams(params)
        return params

    def get_model(self, params=None, **kwargs):
        if params is not None:
            if params['basis_func'] == 'linear_basis_func':
                basis = linear_basis_func
            else:
                basis = quadratic_basis_func
            predictor = BLR(
                alpha=params['alpha'],
                beta=params['beta'],
                basis_func=basis,
                prior=None,
                do_mcmc=False,  # turn this off for better sample efficiency
                n_hypers=20,
                chain_length=100,
                burnin_steps=100,
            )
        else:
            predictor = BLR(
                alpha=1.0,
                beta=100,
                basis_func=linear_basis_func,
                prior=None,
                do_mcmc=False, # turn this off for better sample efficiency
                n_hypers=20,
                chain_length=100,
                burnin_steps=100,
            )
        return predictor

    def train_model(self, xtrain, ytrain):
        self.model.train(xtrain, ytrain, do_optimize=True)

