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
hash_data_directory = os.path.join(base_data_directory, 'hash_dataset')
with open(os.path.join(hash_data_directory, 'train.csv')) as f:
    train_list = f.read().splitlines()
with open(os.path.join(hash_data_directory, 'valid.csv')) as f:
    valid_list = f.read().splitlines()
# Load in the data
(X_train, Y_train, X_validate, Y_validate) = hashing_utils.load_data(
    train_list, valid_list)

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

    epochs = [epoch for epoch, _, _ in
              cross_modality_hashing.train_cross_modality_hasher(
                  X_train, Y_train, X_validate, Y_validate,
                  [hidden_layer_size_X]*(n_layers - 1),
                  [hidden_layer_size_Y]*(n_layers - 1), n_bits=16,
                  learning_rate=learning_rate, **params)]
    success = np.all([np.isfinite(e['validate_cost']) for e in epochs])
    if len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL, 'epochs': epochs}
    else:
        best_objective = np.min([e['validate_objective'] for e in epochs])
        if best_objective < objective.best_objective:
            best_epoch = [e for e in epochs
                          if e['validate_objective'] == best_objective][0]
            print 'New best {}'.format(best_objective, len(epochs))
            for k, v in best_epoch.items():
                print '{} : {},'.format(k, v),
            print
            for k, v in params.items():
                print '{} : {},'.format(k, v),
            print 'n_layers: {}'.format(n_layers),
            print 'learning_rate: {}'.format(learning_rate)
            print
            objective.best_objective = best_objective
        return {'loss': best_objective,
                'status': hyperopt.STATUS_OK,
                'epochs': epochs}

objective.best_objective = np.inf

space = {'n_layers': hyperopt.hp.quniform('n_layers', 3, 4, 1),
         'alpha_XY': hyperopt.hp.lognormal('alpha_XY', 0, 2),
         'm_XY': hyperopt.hp.randint('m_XY', 17),
         'dropout': False,
         'learning_rate_exp': hyperopt.hp.quniform('learning_rate_exp',
                                                   2.5, 8.5, 1),
         'momentum': hyperopt.hp.uniform('momentum', 0, 1)}

if not os.path.exists('../results'):
    os.makedirs('../results')

trials = hyperopt.Trials()
try:
    best = hyperopt.fmin(objective, space=space, algo=hyperopt.tpe.suggest,
                         max_evals=100, trials=trials)
    with open('../results/best.pkl', 'wb') as f:
        pickle.dump(best, f)
except KeyboardInterrupt:
    pass

with open('../results/trials.pkl', 'wb') as f:
    pickle.dump(trials.trials, f)
with open('../results/results.pkl', 'wb') as f:
    pickle.dump(trials.results, f)
