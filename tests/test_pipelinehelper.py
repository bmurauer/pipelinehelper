import unittest

from pipelinehelper import PipelineHelper
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

multilabel_metrics = [
    'precision_samples',
    'recall_samples',
    'f1_samples',
]

class PipelineHelperTest(unittest.TestCase):
    pass


def create_multilabel_test(scorer):
    def do_test(self):
        X, y = datasets.make_multilabel_classification()
        lb = LabelBinarizer()
    
        # y = lb.fit_transform(y)
        pipe = Pipeline([
            ('clf', PipelineHelper([
                ('knn', KNeighborsClassifier()),
                ('nb', Pipeline([
                    ('topositive', MinMaxScaler()),
                    ('estimator', OneVsRestClassifier(MultinomialNB())),
                ])),
            ])),
        ])
        params = {
            'clf__selected_model': pipe.named_steps['clf'].generate()
        }
        grid = GridSearchCV(pipe, params, scoring=scorer, verbose=0, cv=3, 
                n_jobs=-1, iid=True, error_score='raise')
        grid.fit(X, y)
    return do_test

def create_binary_test(scorer):
    def do_test(self):
        X, y = datasets.load_breast_cancer(True)
        pipe = Pipeline([
            ('clf', PipelineHelper([
                ('knn', KNeighborsClassifier()),
                ('nb', Pipeline([
                    ('topositive', MinMaxScaler()),
                    ('estimator', MultinomialNB()),
                ])),
            ])),
        ])
        params = {
            'clf__selected_model': pipe.named_steps['clf'].generate()
        }
        grid = GridSearchCV(pipe, params, scoring=scorer, verbose=0, cv=3, 
                n_jobs=-1, iid=True, error_score='raise')
        grid.fit(X, y)
    return do_test

for scorer in metrics.SCORERS.keys():
    if scorer in multilabel_metrics:
        test_method = create_multilabel_test(scorer)
    else:
        test_method = create_binary_test(scorer)
        
    test_method.__name__ = f'test_{scorer}'
    setattr(PipelineHelperTest, test_method.__name__, test_method)
