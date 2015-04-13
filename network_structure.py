'''
This file defines the structure of the hashing networks.
'''

hidden_layer_sizes = {'X': [2048, 2048], 'Y': [2048, 2048]}
num_filters = {'X': [16, 32], 'Y': [16, 32]}
filter_size = {'X': [(5, 12), (3, 3)],
               'Y': [(5, 3), (3, 3)]}
ds = {'X': [(1, 2), (1, 2)], 'Y': [(1, 2), (1, 2)]}

dropout = False
n_bits = 16
