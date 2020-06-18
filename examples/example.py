from sklearn import datasets
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC

from pipelinehelper import PipelineHelper

X, y = datasets.load_iris(True)

pipe = Pipeline(
    [
        (
            "scaler",
            PipelineHelper(
                [("std", StandardScaler()), ("max", MaxAbsScaler())],
                include_bypass=True,
            ),
        ),  # this will produce one setting without scaler
        (
            "classifier",
            PipelineHelper(
                [
                    ("svm", SVC()),
                    ("rf", RandomForestClassifier()),
                    ("ada", AdaBoostClassifier()),
                    ("gb", GradientBoostingClassifier()),
                    ("knn", KNeighborsClassifier()),
                    (
                        "nb_pipe",
                        Pipeline(
                            [
                                # Naivie Bayes needs positive numbers
                                ("scaler", MinMaxScaler()),
                                ("nb", MultinomialNB()),
                            ]
                        ),
                    ),
                ]
            ),
        ),
    ]
)

params = {
    "scaler__selected_model": pipe.named_steps["scaler"].generate(
        {
            "std__with_mean": [True, False],
            "std__with_std": [True, False],
            # no params for 'max' leads to using standard params
        }
    ),
    "classifier__selected_model": pipe.named_steps["classifier"].generate(
        {
            "svm__C": [0.1, 1.0],
            "svm__kernel": ["linear", "rbf"],
            "rf__n_estimators": [10, 20, 50, 100, 150],
            "rf__max_features": ["auto", "sqrt", "log2"],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__bootstrap": [True, False],
            "ada__n_estimators": [10, 20, 40, 100],
            "ada__algorithm": ["SAMME", "SAMME.R"],
            "gb__n_estimators": [10, 20, 50, 100],
            "gb__criterion": ["friedman_mse", "mse", "mae"],
            "gb__max_features": ["auto", "sqrt", None],
            "knn__n_neighbors": [2, 3, 5, 7, 10],
            "knn__leaf_size": [1, 2, 3, 5],
            "knn__weights": ["uniform", "distance"],
            "knn__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "nb_pipe__nb__fit_prior": [True, False],
            "nb_pipe__nb__alpha": [0.1, 0.2],
        }
    ),
}
grid = GridSearchCV(pipe, params, scoring="accuracy", verbose=1, n_jobs=-1)
grid.fit(X, y)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_.decision_function(X))
