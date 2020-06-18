from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.svm import LinearSVC

from pipelinehelper import PipelineHelper


def test_base_case():
    pipe = Pipeline(
        [
            (
                "t",
                PipelineHelper(
                    [("cv", CountVectorizer()), ("tfidf", TfidfVectorizer()), ]
                ),
            ),
            ("c",
             PipelineHelper([("svm", LinearSVC()), ("nb", MultinomialNB())])),
        ]
    )

    actual = {
        "t__selected_model": pipe.named_steps["t"].generate(
            {
                "cv__analyzer": ["char", "word"],
                "cv__use_idf": [True, False],
                "tfidf__analyzer": ["char", "word"],
            }
        ),
        "c__selected_model": pipe.named_steps["c"].generate(
            {"svm__C": [0.1, 1], "nb__alpha": [0.1, 0.2]}
        ),
    }

    expected = {
        "t__selected_model": [
            ("cv", {"analyzer": "char", "use_idf": True}),
            ("cv", {"analyzer": "char", "use_idf": False}),
            ("cv", {"analyzer": "word", "use_idf": True}),
            ("cv", {"analyzer": "word", "use_idf": False}),
            ("tfidf", {"analyzer": "char"}),
            ("tfidf", {"analyzer": "word"}),
        ],
        "c__selected_model": [
            ("svm", {"C": 0.1}),
            ("svm", {"C": 1}),
            ("nb", {"alpha": 0.1}),
            ("nb", {"alpha": 0.2}),
        ],
    }
    assert expected == actual


def test_optional():
    pipe = Pipeline(
        [
            (
                "scaler",
                PipelineHelper(
                    [("minmax", MinMaxScaler()), ("maxabs", MaxAbsScaler()), ],
                    optional=True,
                ),
            ),
            ("svm", LinearSVC()),
        ]
    )

    actual = {"scaler__selected_model": pipe.named_steps["scaler"].generate()}
    expected = {
        "scaler__selected_model": [
            ("minmax", {}),
            ("maxabs", {}),
            (None, {}),  # this is an artifact of the grid cloning
        ]
    }
    assert expected == actual
