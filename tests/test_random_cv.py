import unittest

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from pipelinehelper import PipelineHelper


class PipelineHelperTest(unittest.TestCase):
    pass


ignored_metrics = [
    # these are multilabel metrics, but gridsearch does not support multilabel
    'recall_samples',
    'precision_samples',
    'f1_samples',
    'jaccard_samples',
    # require strictly positive y-values (?)
    'neg_mean_poisson_deviance',
    'neg_mean_gamma_deviance',
]
requires_decision_function = ['average_precision', 'roc_auc']


def get_data():
    x = np.random.rand(10, 10)
    y = [0] * 5 + [1] * 5
    return x, y


def get_classifiers_for_function(scorer):
    if scorer in requires_decision_function:
        return [
            ('sgd', SGDClassifier()),
            ('ridge', RidgeClassifier()),
        ]
    return [
        ('knn', KNeighborsClassifier()),
        ('rf', RandomForestClassifier()),
    ]


def create_binary_test(scorer):
    def do_test(self):
        x, y = get_data()
        pipe = Pipeline(
            [('clf', PipelineHelper(get_classifiers_for_function(scorer)))]
        )
        params = {'clf__selected_model': pipe.named_steps['clf'].generate()}
        grid = RandomizedSearchCV(
            pipe,
            params,
            scoring=scorer,
            verbose=0,
            cv=2,
            n_jobs=-1,
            error_score='raise',
        )
        grid.fit(x, y)

    return do_test


for scorer in metrics.SCORERS.keys():
    if scorer in ignored_metrics:
        continue
    # if scorer != 'average_precision':
    method = create_binary_test(scorer)
    method.__name__ = f'test_{scorer}'
    setattr(PipelineHelperTest, method.__name__, method)
