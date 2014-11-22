import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import pickle
import csv
import shutil
import normalize_names


def safe_copy(old_path, new_path):
    '''
    Copies a file, but if the destination exists it appends a number.
    '''
    if not os.path.exists(new_path):
        shutil.copy(old_path, new_path)
    else:
        n = 1
        while os.path.exists((os.path.splitext(new_path)[0] +
                              '.{}.mid'.format(n))):
            n += 1
        new_path = os.path.splitext(new_path)[0] + '.{}.mid'.format(n)
        shutil.copy(old_path, new_path)
    return new_path

if not os.path.exists('data/clean_midi/mid'):
    os.makedirs('data/clean_midi/mid')

with open('data/Clean MIDIs-md5_to_freebase_artist_title.pickle') as f:
    md5_to_artist_title = pickle.load(f)

with open('data/Clean MIDIs-md5_to_path.pickle') as f:
    md5_to_path = pickle.load(f)

with open('file_lists/clean_midi.txt', 'wb') as f:
    writer = csv.writer(f, delimiter='\t')
    for n, (md5, artist_title) in enumerate(md5_to_artist_title.items()):
        artist = normalize_names.clean(artist_title[0]).replace('/', ' ')
        title = normalize_names.clean(artist_title[1]).replace('/', ' ')
        original_path = os.path.join('data', md5_to_path[md5])
        if not os.path.exists(original_path):
            print "{} not found".format(original_path)
            continue
        if not os.path.exists(os.path.join('data/clean_midi/mid', artist)):
            os.makedirs(os.path.join('data/clean_midi/mid', artist))
        output_path = os.path.join('data/clean_midi/mid', artist,
                                   title[:247] + '.mid')
        output_path = safe_copy(original_path, output_path)
        writer.writerow([n, artist, title, md5,
                         output_path.replace('data/clean_midi/mid/', '')])
