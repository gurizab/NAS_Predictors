from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace


class PredictorConfigSpace:
    def __init__(self, predictor_type):

        self.predictor_type = predictor_type.upper()

    def get_configspace(self):
        cs = ConfigurationSpace()
        # Conditions
        if self.predictor_type in ['NGB', 'OMNI_NGB', 'OMNI_NGB2']:
            n_estimators = UniformIntegerHyperparameter("param:n_estimators", 128, 512, default_value=236, log=True)
            learning_rate = UniformFloatHyperparameter("param:learning_rate", 0.001, 0.1,
                                                       default_value=0.006724595976491001, log=True)
            max_depth = UniformIntegerHyperparameter('base:max_depth', 1, 25, default_value=10)
            max_features = UniformFloatHyperparameter('base:max_features', 0.1, 1.0, default_value=0.7394770722643155)
            # min_samples_leaf = UniformIntegerHyperparameter('base:min_samples_leaf', 2, 20, default_value=6)
            # min_samples_split = UniformIntegerHyperparameter('base:min_samples_split', 2, 20, default_value=4)
            cs.add_hyperparameters(
                [n_estimators, learning_rate, max_depth, max_features])
        if self.predictor_type in ['XGB', 'OMNI_XGB']:
            # XGB
            max_depth = UniformIntegerHyperparameter("max_depth", 1, 15, default_value=6)
            min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
            colsample_bytree = UniformFloatHyperparameter('colsample_bytree', 0.0, 1.0, default_value=1.0)
            colsample_bylevel = UniformFloatHyperparameter('colsample_bylevel', 0.0, 1.0, default_value=1.0)
            learning_rate = UniformFloatHyperparameter('learning_rate', 0.0001, 1.0, default_value=0.1, log=True)
            cs.add_hyperparameters([max_depth, min_child_weight, colsample_bytree, colsample_bylevel, learning_rate])
        elif self.predictor_type == 'BAYES_LIN_REG':
            # BLR
            alpha = UniformFloatHyperparameter('alpha', 1e-5, 1e5, default_value=1.0)
            beta = UniformFloatHyperparameter('beta', 1e-5, 1e5, default_value=100)
            basis_func = CategoricalHyperparameter('basis_func', ['linear_basis_func', 'quadratic_basis_func'],
                                                   default_value='linear_basis_func')
            cs.add_hyperparameters([alpha, beta, basis_func])
        elif self.predictor_type in ['BANANAS', 'MLP']:
            # BANANAS
            num_layers = UniformIntegerHyperparameter('num_layers', 5, 25, default_value=20)
            layer_width = UniformIntegerHyperparameter('layer_width', 5, 25, default_value=20)
            lr_bananas = UniformFloatHyperparameter('lr', 0.00001, 0.1, log=True)
            batch_size_bananas = UniformIntegerHyperparameter('batch_size', 32, 256, default_value=128, log=True)
            regularization = UniformFloatHyperparameter('regularization', 0, 1, default_value=0.2)
            cs.add_hyperparameters([num_layers, layer_width, lr_bananas, batch_size_bananas, regularization])
        elif self.predictor_type == 'BOHAMIANN':
            # BOHAMIANN
            num_steps = UniformIntegerHyperparameter('num_steps', 60, 500, default_value=100)
            keep_every = UniformIntegerHyperparameter('keep_every', 1, 10, default_value=5)
            lr_bohamiann = UniformFloatHyperparameter('lr', 0.00001, 0.1, log=True)
            num_burn_in_steps = UniformIntegerHyperparameter('num_burn_in_steps', 5, 50, default_value=10)
            cs.add_hyperparameters([num_steps, keep_every, lr_bohamiann, num_burn_in_steps])
        elif self.predictor_type in ['BONAS', 'NAO', 'OMNI_SEMINAS', 'SEMINAS']:
            # BONAS, NAO, OMNI_SEMINAS, SEMINAS
            gcn_hidden = UniformIntegerHyperparameter('gcn_hidden', 16, 128, default_value=64, log=True)
            batch_size_bonas = UniformIntegerHyperparameter('batch_size', 32, 256, default_value=128, log=True)
            lr_bonas = UniformFloatHyperparameter('lr', 0.00001, 0.1, log=True)
            cs.add_hyperparameters([gcn_hidden, batch_size_bonas, lr_bonas])
        elif self.predictor_type == 'DNGO':
            batch_size_dngo = UniformIntegerHyperparameter('batch_size', 32, 256, default_value=128, log=True)
            num_epochs = UniformIntegerHyperparameter('num_epochs', 100, 1000, default_value=500)
            lr_dngo = UniformFloatHyperparameter('learning_rate', 0.00001, 0.1, default_value=0.01, log=True)
            n_units_1 = UniformIntegerHyperparameter('n_units_1', 10, 100, default_value=50)
            n_units_2 = UniformIntegerHyperparameter('n_units_2', 10, 100, default_value=50)
            n_units_3 = UniformIntegerHyperparameter('n_units_3', 10, 100, default_value=50)
            alpha_dngo = UniformFloatHyperparameter('alpha', 1e-5, 1e5, default_value=1.0)
            beta_dngo = UniformFloatHyperparameter('beta', 1e-5, 1e5, default_value=100)
            cs.add_hyperparameters(
                [batch_size_dngo, num_epochs, lr_dngo, n_units_1, n_units_2, n_units_3, alpha_dngo, beta_dngo])
        elif self.predictor_type in ['LGB', 'OMNI_LGB']:
            # LGB
            num_leaves = UniformIntegerHyperparameter('num_leaves', 10, 100, default_value=81)
            lr_lgb = UniformFloatHyperparameter('learning_rate', 0.00001, 0.9, default_value=0.009570519683309102,
                                                log=True)
            feature_fraction = UniformFloatHyperparameter('feature_fraction', 0.1, 1, default_value=0.9093860758993939)
            cs.add_hyperparameters([num_leaves, lr_lgb, feature_fraction])
        elif self.predictor_type == 'GCN':
            # GCN
            gcn_hidden_gcn = UniformIntegerHyperparameter('gcn_hidden', 64, 200, default_value=64, log=True)
            batch_size_gcn = UniformIntegerHyperparameter('batch_size', 5, 32, default_value=7, log=True)
            lr_gcn = UniformFloatHyperparameter('lr', 0.00001, 0.1, default_value=0.0001, log=True)
            wd = UniformFloatHyperparameter('wd', 0.00001, 0.1, default_value=3e-4, log=True)
            cs.add_hyperparameters([gcn_hidden_gcn, batch_size_gcn, lr_gcn, wd])
        elif self.predictor_type == 'RF':
            n_estimators = UniformIntegerHyperparameter('n_estimators', 16, 128, default_value=116, log=True)
            max_features = UniformFloatHyperparameter('max_features', 0.1, 0.9, default_value=0.17055852159745608,
                                                      log=True)
            # min_samples_leaf = UniformIntegerHyperparameter('min_samples_leaf', 1, 20, default_value=2)
            # min_samples_split = UniformIntegerHyperparameter('min_samples_split', 2, 20, default_value=2)
            cs.add_hyperparameters([n_estimators, max_features])
        elif self.predictor_type in ['GP', 'GPWL', 'SPARSE_GP', 'VAR_SPARSE_GP']:
            # GP
            kernel_gp = CategoricalHyperparameter('kernel_type', ['RBF', 'Matern32', 'Matern52'], default_value='RBF')
            lengthscale_gp = UniformFloatHyperparameter('lengthscale', 1e-5, 1e5, default_value=10)
            cs.add_hyperparameters([kernel_gp, lengthscale_gp])

        return cs

