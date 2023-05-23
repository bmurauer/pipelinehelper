from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC

from pipelinehelper import PipelineHelper


def test_roc_score():
    data, targets = make_classification()

    p = Pipeline([
        ('clf', PipelineHelper([
            ('et', ExtraTreesClassifier()),
            ('rf', RandomForestClassifier()),
            ('gauss_nb', GaussianNB()),
        ]))
    ])

    params = {
        'clf__selected_model': p.named_steps['clf'].generate()
    }

    grid = GridSearchCV(p, params, scoring='roc_auc')
    grid.fit(data, targets)


def test_support():
    data, targets = make_classification()

    pipe = Pipeline([
        ('selector', PipelineHelper([
            ('kb', SelectKBest()),
        ])),
        ('classifier', LinearSVC()),
    ])

    search_space = {
        'selector__selected_model': pipe.named_steps['selector'].generate({
            'kb__k': [2, 3],
        }),
    }

    clf1 = GridSearchCV(pipe, search_space, cv=5,scoring = 'roc_auc')
    clf1 = clf1.fit(data, targets)
    clf1.best_estimator_.named_steps['selector'].get_support()
