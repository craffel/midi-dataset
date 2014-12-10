import sys
sys.path.append('../')
import cross_modality_hashing
import hashing_utils
import numpy as np
import os
import hyperopt
import pickle


# Set up paths
base_data_directory = '../data'
training_data_directory = os.path.join(base_data_directory, 'hash_dataset',
                                       'npz')
# Load in the data
X, Y = hashing_utils.load_data(training_data_directory)
# Split into train and validate and standardize
(X_train, Y_train, X_validate,
 Y_validate) = hashing_utils.train_validate_split(X, Y, .99)

# Use this many samples to compute mean reciprocal rank
n_mrr_samples = 500
# Pre-compute indices over which to compute mrr
mrr_samples = np.random.choice(X_validate.shape[0], n_mrr_samples, False)

# Compute layer sizes.  Middle layers are nextpow2(input size)
hidden_layer_size_X = int(2**np.ceil(np.log2(X_train.shape[1])))
hidden_layer_size_Y = int(2**np.ceil(np.log2(Y_train.shape[1])))


def objective(params):
    '''
    Wrapper around cross-modality hashing for hyperopt
    '''
    n_layers = int(params['n_layers'])
    learning_rate = 10**(-params['learning_rate_exp'])
    params = dict([(k, v) for k, v in params.items()
                   if k != 'n_layers' and k != 'learning_rate_exp'])

    epochs = cross_modality_hashing.train_cross_modality_hasher(
        X_train, Y_train, X_validate, Y_validate,
        [hidden_layer_size_X]*(n_layers - 1),
        [hidden_layer_size_Y]*(n_layers - 1),
        mrr_samples=mrr_samples, n_bits=16, learning_rate=learning_rate,
        **params)
    for k, v in params.items():
        print '{} : {},'.format(k, v),
    print 'n_layers: {}'.format(n_layers),
    print 'learning_rate: {}'.format(learning_rate)
    success = np.all([np.isfinite(e['validate_cost']) for e in epochs])
    if len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL, 'epochs': epochs}
    else:
        best_mrr = np.max([e['validate_mrr_pessimist'] for e in epochs])
        print '   Best MRR {} in {} epochs'.format(best_mrr, len(epochs))
        print
        return {'loss': -best_mrr,
                'status': hyperopt.STATUS_OK,
                'epochs': epochs}

space = {'n_layers': 3,  # hyperopt.hp.quniform('n_layers', 3, 4, 1),
         'alpha_XY': hyperopt.hp.lognormal('alpha_XY', 0, 1),
         'alpha_X': hyperopt.hp.lognormal('alpha_X', 0, 1),
         'alpha_Y': hyperopt.hp.lognormal('alpha_Y', 0, 1),
         'm_XY': hyperopt.hp.randint('m_XY', 17),
         'm_X': hyperopt.hp.randint('m_X', 17),
         'm_Y': hyperopt.hp.randint('m_Y', 17),
         'dropout': False,
         'learning_rate_exp': hyperopt.hp.quniform('learning_rate_exp',
                                                   2.5, 8.5, 1),
         'momentum': hyperopt.hp.uniform('momentum', 0, 1)}

trials = hyperopt.Trials()
best = hyperopt.fmin(objective, space=space, algo=hyperopt.tpe.suggest,
                     max_evals=1000, trials=trials)

with open('trials.pkl', 'wb') as f:
    pickle.dump(trials.trials, f)
with open('results.pkl', 'wb') as f:
    pickle.dump(trials.results, f)
with open('best.pkl', 'wb') as f:
    pickle.dump(best, f)
