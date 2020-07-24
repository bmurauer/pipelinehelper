from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

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
