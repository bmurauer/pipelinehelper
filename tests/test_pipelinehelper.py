import unittest

from pipelinehelper import PipelineHelper
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier


class PipelineHelperTest(unittest.TestCase):
    pass


ignored_metrics = [
    # these are multilabel metrics, but gridsearch does not support multilabel
    "recall_samples",
    "precision_samples",
    "f1_samples",
    "jaccard_samples",
    # require strictly positive y-values (?)
    "neg_mean_poisson_deviance",
    "neg_mean_gamma_deviance",
]
requires_decision_function = ["average_precision", "roc_auc"]


def get_classifiers_for_function(scorer):
    if scorer in requires_decision_function:
        return [
            ("sgd", SGDClassifier()),
            ("ridge", RidgeClassifier()),
        ]
    return [
        ("knn", KNeighborsClassifier()),
        ("rf", RandomForestClassifier()),
    ]


def create_binary_test(scorer):
    def do_test(self):
        X, y = datasets.load_breast_cancer(return_X_y=True)
        pipe = Pipeline(
            [("clf", PipelineHelper(get_classifiers_for_function(scorer))),]
        )
        params = {"clf__selected_model": pipe.named_steps["clf"].generate()}
        grid = GridSearchCV(
            pipe,
            params,
            scoring=scorer,
            verbose=0,
            cv=3,
            n_jobs=-1,
            error_score="raise",
        )
        grid.fit(X, y)

    return do_test


for scorer in metrics.SCORERS.keys():
    if scorer in ignored_metrics:
        continue
    # if scorer != 'average_precision':
    test_method = create_binary_test(scorer)
    test_method.__name__ = f"test_{scorer}"
    setattr(PipelineHelperTest, test_method.__name__, test_method)
