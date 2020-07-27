from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from pipelinehelper import PipelineHelper


def test_nested_params():
    data, targets = make_classification()

    p = Pipeline([
        ('clf', PipelineHelper([
            ('et', ExtraTreesClassifier()),
            ('rf', RandomForestClassifier())
        ]))
    ])

    grid = GridSearchCV(p, {
        'clf__selected_model': p.named_steps['clf'].generate({
            'et__n_estimators': [10, 20],
            'rf__n_estimators': [10, 20],
        }),
    })
    grid.fit(data, targets)
    params = grid.get_params()

    assert 'estimator__clf__available_models__et' in params
    assert 'estimator__clf__available_models__rf' in params

    best_params = grid.best_estimator_.get_params()
    assert 'clf__selected_model' in best_params
    # no matter the actual model, n_estimators should be in here
    assert 'clf__selected_model__n_estimators' in best_params
