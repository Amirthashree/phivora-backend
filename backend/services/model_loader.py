import pickle
import os
import sys
sys.path.append(os.path.abspath("."))
from backend.config import settings

class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        d = settings.MODEL_DIR
        with open(f"{d}/xgboost.pkl",       "rb") as f: self.xgboost       = pickle.load(f)
        with open(f"{d}/random_forest.pkl",  "rb") as f: self.random_forest = pickle.load(f)
        with open(f"{d}/sgd.pkl",            "rb") as f: self.sgd           = pickle.load(f)
        with open(f"{d}/naive_bayes.pkl",    "rb") as f: self.naive_bayes   = pickle.load(f)
        with open(f"{d}/feature_cols.pkl",   "rb") as f: self.feature_cols  = pickle.load(f)
        self._loaded = True
        print("All models loaded successfully")

    def is_loaded(self):
        return self._loaded

model_loader = ModelLoader()
