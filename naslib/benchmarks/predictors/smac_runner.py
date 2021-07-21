import logging

import numpy as np
import copy
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from naslib.utils import generate_kfold, cross_validation
from naslib.benchmarks.predictors.predictor_config_space import PredictorConfigSpace

logger = logging.getLogger(__name__)


def get_config(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    return cfg


class SMACRunner:

    def __init__(self, xtrain, ytrain, predictor_type, predictor, max_hpo_time, metric='kendalltau'):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.predictor = predictor
        self.predictor_type = predictor_type
        self.metric = metric
        self.max_hpo_time = max_hpo_time

        logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    def predictor_run(self, config):
        logger.info(f'Starting cross validation')
        n_train = len(self.xtrain)
        split_indices = generate_kfold(n_train, 3)

        predictor = copy.deepcopy(self.predictor)

        hyperparams = get_config(config)
        predictor.set_hyperparams(hyperparams)

        print('Hyperparams: ', hyperparams)
        print('----------------------')

        cv_score = np.nan_to_num(cross_validation(self.xtrain, self.ytrain, predictor, split_indices, self.metric), -1)
        print('Cross Validation score: ', cv_score)
        logger.info(f'Finished')

        return 1 - cv_score

    def run(self):
        # Scenario object
        config = PredictorConfigSpace(self.predictor_type)
        cs = config.build_config_space()
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                             "wallclock-limit": self.max_hpo_time,
                             # max. number of function evaluations; for this example set to a low number
                             "cs": cs,  # configuration space
                             "deterministic": "true",
                             "limit_resources": False
                             })
        smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                        tae_runner=self.predictor_run)
        # Optimize, using a SMAC-object
        print("Optimizing! Depending on your machine, this might take a few minutes.")

        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        return get_config(incumbent)

