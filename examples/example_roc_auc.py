from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from pipelinehelper import PipelineHelper
from sklearn import metrics

from sklearn.datasets import make_classification
X, y = make_classification(100, 23)

pipe = Pipeline([
    # ('scaler', PipelineHelper([
        # ('std', StandardScaler()),
        # ('abs', MaxAbsScaler()),
        # ('minmax', MinMaxScaler()),
        # ('pca', PCA(svd_solver='full', whiten=True)),
    # ])),
    ('classifier', PipelineHelper([
        ('knn', KNeighborsClassifier(weights='distance')),
        # ('gbc', GradientBoostingClassifier())
    ])),
])
# params = {}
params = {
    # 'scaler__selected_model': pipe.named_steps['scaler'].generate(),
        # 'std__with_mean': [True, False],
        # 'std__with_std': [True, False],
        # 'pca__n_components': [0.5, 0.75, 0.9, 0.99],
    # }),
    'classifier__selected_model': pipe.named_steps['classifier'].generate(),
        # 'knn__n_neighbors': [1, 3, 5, 7, 10],#, 30, 50, 70, 90, 110, 130, 150, 170, 190],
        # 'gbc__learning_rate': [0.1, 0.5, 1.0],
        # 'gbc__subsample': [0.5, 1.0],
    # })
}

grid = GridSearchCV(pipe, params, scoring='roc_auc', n_jobs=1, verbose=1, cv=5)
grid.fit(X, y)

