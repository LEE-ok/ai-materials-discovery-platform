import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

class ModelEngine:
    def __init__(self, model_type='RF', output_dim=2, max_iter=2000):
        self.model_type = model_type
        self.output_dim = output_dim
        self.max_iter = max_iter
        self.model = self._create_model(model_type)
        self.is_trained = False
        
    def _create_model(self, model_type):
        if model_type == 'RF':
            return RandomForestRegressor(n_estimators=100)
        elif model_type == 'GBM':
            # Gradient Boosting wrap
            return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100))
        elif model_type == 'MLP':
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=self.max_iter, 
                                early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
        elif model_type == 'TFP':
            # Bootstrapped MLP Ensemble with Early Stopping enabled
            models = [MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=self.max_iter, 
                                   early_stopping=True, validation_fraction=0.1, 
                                   n_iter_no_change=10, random_state=i) for i in range(5)]
            return models
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
            
    def train(self, X_train, y_train):
        if self.model_type == 'TFP':
            # Train each model in the ensemble with a different bootstrap sample
            n_samples = X_train.shape[0]
            for m in self.model:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                m.fit(X_train[indices], y_train[indices])
        else:
            self.model.fit(X_train, y_train)
            
        self.is_trained = True
        
        class History:
            def __init__(self):
                self.history = {'loss': [0]}
        return History()

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model not trained yet.")
            
        if self.model_type == 'TFP':
            # Collect predictions from all models in the ensemble
            all_preds = np.array([m.predict(X) for m in self.model])
            mean = np.mean(all_preds, axis=0)
            std = np.std(all_preds, axis=0)
            return mean, std
        
        mean = self.model.predict(X)
        
        if self.model_type == 'RF':
            all_tree_preds = []
            for tree in self.model.estimators_:
                all_tree_preds.append(tree.predict(X))
            all_tree_preds = np.array(all_tree_preds)
            std = np.std(all_tree_preds, axis=0)
        else:
            # Heuristic for GBM/MLP (standard deviation estimate)
            std = np.abs(mean) * 0.05
            
        return mean, std

    def save(self, path):
        joblib.dump({"model": self.model, "type": self.model_type}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.model_type = data.get("type", "RF")
        self.is_trained = True
