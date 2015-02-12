'''
Split the hashing training data into training, validation, and test sets.
'''

import os
import glob
import numpy as np

# Proportions of the dataset to use for each set
TRAIN_PROPORTION = .8
VALID_PROPORTION = .1
TEST_PROPORTION = .1

# Directory to hashing dataset
BASE_DATA_DIRECTORY = '../data'
training_data_directory = os.path.join(
    BASE_DATA_DIRECTORY, 'hash_dataset', 'npz')

# Get list of all npz files
files = list(glob.glob(os.path.join(training_data_directory, '*.npz')))

# Lists of training, validation, and test set files to be populated below
train_files = []
valid_files = []
test_files = []

while len(files) > 0:
    # Draw a random number [0 -> 1] to choose which set to place files in
    random = np.random.rand()
    if random <= TRAIN_PROPORTION:
        file_set = train_files
    elif random < TRAIN_PROPORTION + VALID_PROPORTION:
        file_set = valid_files
    else:
        file_set = test_files
    # Pop the last file from the list
    last_file = files[-1]
    del files[-1]
    file_set.append(os.path.abspath(last_file))
    # Get the file basename
    basename = os.path.splitext(os.path.split(last_file)[-1])[0]
    # Get the audio_id from filename (e.g. cal10k_192)
    audio_id = '{}_{}'.format(*basename.split('_')[:2])
    # Get the MIDI MD5 from the filename (e.g. f2e92b9aa..)
    midi_id = basename.split('_')[-1]
    # Add all files with the same audio id or midi md5
    files_to_add = [filename for filename in files
                    if (audio_id in filename or midi_id in filename)]
    file_set += [os.path.abspath(file) for file in files_to_add]
    for filename in files_to_add:
        del files[files.index(filename)]


# Write out
def write_list(filename, l):
    with open(filename, 'wb') as f:
        f.writelines([n + '\n' for n in l])

hash_dataset_directory = os.path.join(BASE_DATA_DIRECTORY, 'hash_dataset')
write_list(os.path.join(hash_dataset_directory, 'train.csv'), train_files)
write_list(os.path.join(hash_dataset_directory, 'valid.csv'), valid_files)
write_list(os.path.join(hash_dataset_directory, 'test.csv'), test_files)
