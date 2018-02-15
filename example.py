from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from pipelinehelper import PipelineHelper

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
pipe = Pipeline([
    ('scaler', PipelineHelper([
        # nested! 
        ('std', Pipeline([
            ('std', StandardScaler()),
        ])),
        ('max', MaxAbsScaler()),
    ])),
    ('classifier', PipelineHelper([
        ('svm', LinearSVC()),
        ('rf', RandomForestClassifier()),
    ])),
])

params = {
    'scaler__selected_model': pipe.named_steps['scaler'].generate({
        'std__std__with_mean': [True, False],
        'std__std__with_std': [True, False],
        'max__copy': [True],
    }),
    'classifier__selected_model': pipe.named_steps['classifier'].generate({
        'svm__C': [0.1, 1.0],
        'rf__n_estimators': [100, 20],
    })
}
grid = GridSearchCV(pipe, params, scoring='accuracy', verbose=1)
grid.fit(X_iris, y_iris)
print(grid.best_params_)
print(grid.best_score_)
