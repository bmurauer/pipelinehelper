# Pipeline helper class for scikit #
With this class, elements of a scikit pipeline can be hot-swapped for grid search, along with their parameters. 
This helper is __specifically designed for the use with GridSearch__, but also has been shown to work with RandomizedSearchCV (although not tested as thoroughly).

This class provides the following features:

#### 1. It can hold regular Pipeline objects
This can be useful in cases where a specific element of the pipeline requires additional preprocessing. 
For example, the `StandardScaler` class required the data to be dense, whereas the `MaxAbsScaler` does not. To compare the two elements directly, the PipelineHelper can be used in the following fashion:

```python
pipe = Pipeline([
    ('scaler', PipelineHelper([
        ('maxabs', MaxAbsScaler()),
        ('stdev', Pipeline([
            ('todense', FunctionTransformer(lambda x: x.todense(), allow_sparse=True)),
            ('std', StandardScaler()),
        ])),
    ]))
    ('svm', LinearSVC())
])

params = {
    'scaler__selected_model': pipe.named_steps['scaler'].generate({
        'maxabs__copy': [True, False],
        'stdev__std__with_mean': [True, False],
        ...
    }),
    'svm__C': [0.1, 1.0],
}

```

#### 2. No need to specify default parameters
If no parameters are provided for an element, the default parameters are used.

```python
pipe = Pipeline([
    ('scaler', PipelineHelper([
        ('maxabs', MaxAbsScaler()),
        ('std', StandardScaler()),
    ]))
    ('svc', LinearSVC())
])

params = {
    'scaler__selected_model': pipe.named_steps['scaler'].generate({
        'stdev__std__with_mean': [True, False],
        # MaxAbs will still be tested with default parameters
    }),
    'svm__C': [0.1, 1.0]
}

```

#### 3. Can be used for classifiers or other elements of a pipeline
Not limited to intermediate Transformers:

```python
pipe = Pipeline([
    ('scaler', MaxAbsScaler()),
    ('clf', PipelineHelper([
      ('svm', LinearSVC()),
      ('rf', RandomForestClassifier()),
    ])),
])

params = {
    'clf__selected_model': pipe.named_steps['clf'].generate({
        'svm__C': [0.1, 1.0],
        'rf__n_estimators': [10, 50],
    }),
}
grid = GridSearchCV(pipe, params, scoring='accuracy')
```

## When do I need this?
The scikit search algorithms already support swapping estimators by specifying
them in the parameter grid, like in this example demonstrating [dimensionality reduction](https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html).

However, this has some limitations:
 - it is limited to transformers, so the last part of a pipeline (e.g., a classifier) can't be switched in this manner (at least I wasn't able to, please correct me if I'm wrong)
 
 - If you want to try different parameters for each of the options, you have to specify them separately, which is what we try to avoid in the first place
 
If you run in one of those two limitations, this tool is right for you! 

## Installation

The project is now on PyPI, so it can be installed using:

    pip install pipelinehelper

Then import it:

    from pipelinehelper import PipelineHelper

### Open issues:

 - Nesting PipelineHelpers themselves does not work yet. 
   I'm not sure how useful this would be.
