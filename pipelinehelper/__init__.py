from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from collections import defaultdict
import itertools


class PipelineHelper(BaseEstimator, TransformerMixin, ClassifierMixin):

    def __init__(self, available_models=None, selected_model=None, include_bypass=False):
        self.include_bypass = include_bypass
        self.selected_model = selected_model
        # this is required for the clone operator used in gridsearch 
        if type(available_models) == dict:
            self.available_models = available_models
        # this is the case for constructing the helper initially
        else:
            # a string identifier is required for assigning parameters
            self.available_models = {}
            for (key, model) in available_models:
                self.available_models[key] = model

    def generate(self, param_dict={}):
        per_model_parameters = defaultdict(lambda: defaultdict(list))
        
        # collect parameters for each specified model
        for k, values in param_dict.items():
            model_name = k.split('__')[0]
            param_name = k[len(model_name)+2:]  # might be nested
            if model_name not in self.available_models:
                raise Exception('no such model: {0}'.format(model_name))
            per_model_parameters[model_name][param_name] = values

        ret = []
            
        # create instance for cartesion product of all available parameters for each model
        for model_name, param_dict in per_model_parameters.items():
            parameter_sets = (dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values()))
            for parameters in parameter_sets:
                ret.append((model_name, parameters))

        # for every model that has no specified parameters, add the default model
        for model_name in self.available_models.keys():
            if model_name not in per_model_parameters:
                ret.append((model_name, dict()))

        # check if the stage is to be bypassed as one configuration
        if self.include_bypass:
            ret.append((None, dict(), True))
        return ret
               
    def get_params(self, deep=False):
        return {'available_models': self.available_models,
                'selected_model': self.selected_model,
                'include_bypass': self.include_bypass}

    def set_params(self, selected_model, available_models=None, include_bypass=False):
        include_bypass = len(selected_model) == 3 and selected_model[2]
                    
        if available_models:
            self.available_models = available_models

        if selected_model[0] is None and include_bypass:
            self.selected_model = None
            self.include_bypass = True
        else:
            if selected_model[0] not in self.available_models:
                raise Exception('so such model available: {0}'.format(selected_model[0]))
            self.selected_model = self.available_models[selected_model[0]]
            self.selected_model.set_params(**selected_model[1])

    def fit(self, X, y=None):
        if self.selected_model is None and not self.include_bypass:
            raise Exception('no model was set')
        elif self.selected_model is None:
            return self
        else:
            return self.selected_model.fit(X, y)

    def transform(self, X, y=None):
        if self.selected_model is None and not self.include_bypass:
            raise Exception('no model was set')
        elif self.selected_model is None:
            return X
        else:
            return self.selected_model.transform(X)

    def predict(self, x):
        if self.include_bypass:
            raise Exception('bypassing classifier is not allowed')
        if self.selected_model is None:
            raise Exception('no model was set')
        return self.selected_model.predict(x)


    def predict_proba(self, x):
        if hasattr(self.selected_model, "predict_proba"):
            method = getattr(self.selected_model, "predict_proba", None)
            if callable(method):
                return method(x)
        else:
            raise Exception("Your model does not support predict_proba")


    def decision_function(self, x):
        if hasattr(self.selected_model, 'decision_function'):
            method = getattr(self.selected_model, 'decision_function')
            if callable(method):
                return method(x)

