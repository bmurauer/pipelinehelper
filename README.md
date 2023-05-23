# ARCHIVING

This helper class is no longer maintained, but it is also no longer required, and I don't recommend using it.

The parameters of the grid/random search can be fully specified using a list of parameter grid dictionaries:

```python
pl = Pipeline([
    ('est', LinearSVC())
])
param_grid=[
    {'est': [RandomForestClassifier()],'est__n_estimators':[5,10,25]},
    {'est': [DecisionTreeClassifier()] },
    
]
a = GridSearchCV(pl,param_grid)
```
This way, every feature of this helper class can be modelled.
The only benefit that this helper class would provide is to somewhat change the syntax, which may or may not be more clear to the user.


---

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
The scikit search algorithms already support swapping transformers by specifying them in the parameter grid, like in this example demonstrating [dimensionality reduction](https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html).

However, this has some limitations:
 - I find it confusing that the definition of pipeline steps 


If you run in one of those two limitations, this tool is right for you! 

## Installation

The project is now on PyPI, so it can be installed using:

    pip install pipelinehelper

Then import it:

    from pipelinehelper import PipelineHelper

### Open issues:

 - Nesting PipelineHelpers themselves does not work yet. 
   I'm not sure how useful this would be.
