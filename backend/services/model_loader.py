import pickle
import os
import sys
sys.path.append(os.path.abspath("."))
from huggingface_hub import hf_hub_download

HF_REPO_ID = "amirtha1306/phivora-models"

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
        print("Downloading models from Hugging Face...")
        xgboost_path       = hf_hub_download(repo_id=HF_REPO_ID, filename="xgboost.pkl")
        random_forest_path = hf_hub_download(repo_id=HF_REPO_ID, filename="random_forest.pkl")
        sgd_path           = hf_hub_download(repo_id=HF_REPO_ID, filename="sgd.pkl")
        naive_bayes_path   = hf_hub_download(repo_id=HF_REPO_ID, filename="naive_bayes.pkl")
        feature_cols_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="feature_cols.pkl")
        with open(xgboost_path,       "rb") as f: self.xgboost       = pickle.load(f)
        with open(random_forest_path, "rb") as f: self.random_forest = pickle.load(f)
        with open(sgd_path,           "rb") as f: self.sgd           = pickle.load(f)
        with open(naive_bayes_path,   "rb") as f: self.naive_bayes   = pickle.load(f)
        with open(feature_cols_path,  "rb") as f: self.feature_cols  = pickle.load(f)
        self._loaded = True
        print("All models loaded from Hugging Face!")

    def is_loaded(self):
        return self._loaded

model_loader = ModelLoader()
