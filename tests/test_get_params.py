from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pipelinehelper import PipelineHelper


def test_deep_params():
    data, targets = make_classification()

    p = Pipeline([
        ('clf', PipelineHelper([
            ('et', ExtraTreesClassifier()),
        ]))
    ])

    grid = GridSearchCV(p, {
        'clf__selected_model': p.named_steps['clf'].generate({
            'et__n_estimators': [10, 20],
        }),
    })
    grid.fit(data, targets)
    params = grid.best_estimator_.get_params(deep=True)
    assert 'clf__selected_model' in params
    assert 'clf__selected_model__n_estimators' in params
