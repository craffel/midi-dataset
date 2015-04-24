import sys
sys.path.append('../')
import cross_modality_hashing
import hashing_utils
import numpy as np
import os
import whetlab
import glob

# Set up paths
base_data_directory = '../data'
hash_data_directory = os.path.join(base_data_directory, 'hash_dataset')
train_list = list(glob.glob(os.path.join(
    hash_data_directory, 'train', 'npz', '*.npz')))
valid_list = list(glob.glob(os.path.join(
    hash_data_directory, 'valid', 'npz', '*.npz')))
# Load in the data
(X_train, Y_train, X_validate, Y_validate) = hashing_utils.load_data(
    train_list, valid_list)

# Load whetlab experiment
scientist = whetlab.Experiment(name="Hashing parameter search 6")
# Get hyperparameter suggestion
job = scientist.suggest()
# We will use the suggested parameters to create some other parameters (below)
# so let's create a copy so we can modify the dict but still use it as a job
# key later
params = dict(job)

# Use the # of hidden layers and the hidden layer power to construct a list
# [2^hidden_power, 2^hidden_power, ...n_hidden times...]
hidden_layer_sizes = [2**11]*params['n_hidden']
params['hidden_layer_sizes'] = {'X': hidden_layer_sizes,
                                'Y': hidden_layer_sizes}
# Use the number of convolutional layers to construct a list
# [16, 32 ...n_conv times]
num_filters = [2**(n + 4) for n in xrange(params['n_conv'])]
params['num_filters'] = {'X': num_filters, 'Y': num_filters}
# First filter is 12 semitones tall
params['filter_size'] = {'X': [(5, 12)] + [(3, 3)]*(params['n_conv'] - 1),
                         'Y': [(5, 12)] + [(3, 3)]*(params['n_conv'] - 1)}
# Construct a downsample list [(1, 2), (1, 2), ...n_conv times...]
ds = [(2, 2), (2, 2)] + [(1, 2)]*(params['n_conv'] - 2)
params['ds'] = {'X': ds, 'Y': ds}
# Remove hidden_pow, n_hidden, and n_conv parameters
params = dict([(k, v) for k, v in params.items()
               if k != 'hidden_pow' and k != 'n_hidden' and k != 'n_conv'])

for k, v in params.items():
    print '\t{} : {},'.format(k, v)

# Train hasher
epochs = []
try:
    for epoch, _, _ in cross_modality_hashing.train_cross_modality_hasher(
            X_train, Y_train, X_validate, Y_validate, **params):
        # Stop training of a nan training cost is encountered
        if not np.isfinite(epoch['train_cost']):
            break
        epochs.append(epoch)
        print "{}: {}, ".format(epoch['iteration'],
                                epoch['validate_objective']),
# If there was an error while training, report it to whetlab
except Exception as ex:
    print "ERROR: {}".format(ex)
    scientist.update_as_failed(job)
    sys.exit()
print
# Check that all training costs were not Nan; if not, report error to whetlab
success = np.all([np.isfinite(e['train_cost']) for e in epochs])
if len(epochs) == 0 or not success:
    print '    Failed to converge.'
    print
    scientist.update_as_failed(job)
else:
    # If training was successful, find the maximum validation objective
    best_objective = np.max([e['validate_objective'] for e in epochs])
    best_epoch = [e for e in epochs
                  if e['validate_objective'] == best_objective][0]
    for k, v in best_epoch.items():
        print "\t{:>35} | {}".format(k, v)
    print
    # and report it to whetlab
    scientist.update(job, best_objective)
