import csv
import os
import shutil


def load_results(path_to_tsv):
    '''
    Given a tab-separated value file with entries in the format
        Bryan Adams - Summer Of '69.mp3 70.3806 0:00:03 0:03:12 Some egregious pitch bends
    return a list of the files which have start/end times listed and the start/end times.
    '''
    files = []
    with open(path_to_tsv) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for line in csv_reader:
            if len(line) == 5 and len(line[2].split(':')) == 3 and len(line[3].split(':')) == 3:
                files.append(line[0])
    return files


def get_data_folder(filename):
    ''' Given an mp3 filename, returns cal500 or cal10k depending on where the file is (a hack) '''
    if ' ' in filename:
        return 'cal10k'
    else:
        return 'cal500'


# Create the hashing dataset based on which files are aligned, and when they are
if __name__=='__main__':
    # Set up paths
    base_data_path = '../data'
    aligned_path = os.path.join(base_data_path, 'aligned')
    tsv_path = os.path.join(aligned_path, 'results.tsv')
    output_path = os.path.join(base_data_path, 'hash_dataset')
    midi_directory = 'midi-aligned-new-new-dpmod-multiple-files'

    def to_numbered_mid(filename):
        ''' Given an mp3 filename, return the corresponding best alignment .mid name according to the .pdf present '''
        base, _ = os.path.splitext(filename)
        if os.path.exists(os.path.join(aligned_path, base + '.pdf')):
            return filename.replace('mp3', 'mid')
        n = 1
        while not os.path.exists(os.path.join(aligned_path, '{}.{}.pdf'.format(base, n))):
            n += 1
        return '{}.{}.mid'.format(base, n)
    def to_h5_path(filename):
        ''' Given an mp3 filename, returns the path to the corresponding -beats.npy file '''
        return os.path.join(base_data_path, get_data_folder(filename), 'msd', filename.replace('.mp3', '.h5'))
    def to_midi_path(filename):
        ''' Given an mp3 filename, returns the path to the corresponding midi file '''
        return os.path.join(get_data_folder(filename), midi_directory, to_numbered_mid(filename))

    # Load in list of files which were aligned correctly, and the start/end times of the good alignment
    files = load_results(tsv_path)

    for filename in files:
        src_midi = to_midi_path(filename)
        dst_midi = os.path.join(base_data_path, 'aligned_old',
                                get_data_folder(filename),
                                filename.replace('.mp3', '.mid'))
        shutil.copy(os.path.join(base_data_path, src_midi),
                    os.path.join(base_data_path, 'aligned_old',
                                 get_data_folder(filename),
                                 filename.replace('.mp3', '.mid')))
