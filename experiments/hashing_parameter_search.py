import sys
sys.path.append('../')
import cross_modality_hashing
import hashing_utils
import numpy as np
import os
import hyperopt
import pickle
from network_structure import hidden_layer_sizes, num_filters, filter_size, ds


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


def objective(params):
    '''
    Wrapper around cross-modality hashing for hyperopt
    '''
    learning_rate = 10**(-params['learning_rate_exp'])
    params = dict([(k, v) for k, v in params.items()
                   if k != 'n_layers' and k != 'learning_rate_exp'])

    for k, v in params.items():
        print '\t{} : {},'.format(k, v),
    print '\tlearning_rate: {}'.format(learning_rate)

    epochs = []
    for epoch, _, _ in cross_modality_hashing.train_cross_modality_hasher(
            X_train, Y_train, X_validate, Y_validate, num_filters, filter_size,
            ds, hidden_layer_sizes, n_bits=16, learning_rate=learning_rate,
            **params):
        if not np.isfinite(epoch['validate_cost']):
            break
        epochs.append(epoch)
    success = np.all([np.isfinite(e['validate_cost']) for e in epochs])
    if len(epochs) == 0 or not success:
        print '    Failed to converge.'
        print
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL, 'epochs': epochs}
    else:
        best_objective = np.min([e['validate_objective'] for e in epochs])
        best_epoch = [e for e in epochs
                      if e['validate_objective'] == best_objective][0]
        for k, v in best_epoch.items():
            print "\t{:>35} | {}".format(k, v)
        if best_objective < objective.best_objective:
            print '### New best {}'.format(best_objective, len(epochs))
            objective.best_objective = best_objective
        print
        return {'loss': best_objective,
                'status': hyperopt.STATUS_OK,
                'epochs': epochs}

objective.best_objective = np.inf

space = {'alpha_XY': hyperopt.hp.uniform('alpha_XY', 0, .25),
         'm_XY': hyperopt.hp.randint('m_XY', 8),
         'dropout': hyperopt.hp.randint('dropout', 2),
         'learning_rate_exp': hyperopt.hp.quniform('learning_rate_exp',
                                                   0.5, 5.5, 1),
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
