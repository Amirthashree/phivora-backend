import sys, os
sys.path.append(os.path.abspath('.'))
from ml.training.train_xgboost      import run_train_xgboost
from ml.training.train_random_forest import run_train_random_forest
from ml.training.train_naive_bayes  import run_train_naive_bayes
from ml.training.train_sgd          import run_train_sgd

if __name__ == '__main__':
    print('='*60)
    print('TRAINING ALL MODELS')
    print('='*60)
    print('[1/4] Model 1 - SGD char ngrams'); run_train_xgboost()
    print('[2/4] Model 2 - SGD word tokens'); run_train_random_forest()
    print('[3/4] Model 3 - MultinomialNB');   run_train_sgd()
    print('[4/4] Naive Bayes text');           run_train_naive_bayes()
    print('='*60)
    print('ALL MODELS TRAINED AND SAVED')
    print('='*60)