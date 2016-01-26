'''
Create feature files for MSD 7digital 30 second clips
'''
import deepdish
import joblib
import librosa
import os
import sys
sys.path.append('..')
import feature_extraction
import whoosh_search
import traceback

BASE_DATA_PATH = '../data'


def process_one_file(mp3_filename, skip=True):
    '''
    Load in an mp3, get the features, and write the features out

    :parameters:
        - mp3_filename : str
            Path to an mp3 file
        - skip : bool
            Whether to skip files when the h5 already exists
    '''
    # h5 files go in the 'h5' dir instead of 'mp3'
    output_filename = mp3_filename.replace('mp3', 'h5')
    # Skip files already created
    if skip and os.path.exists(output_filename):
        return
    try:
        # Load audio and compute CQT
        audio_data, _ = librosa.load(
            mp3_filename, sr=feature_extraction.AUDIO_FS)
        cqt = feature_extraction.audio_cqt(audio_data)
        # Create subdirectories if they don't exist
        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])
        # Save CQT
        deepdish.io.save(output_filename, {'gram': cqt})
    except Exception as e:
        print "Error processing {}: {}".format(
            mp3_filename, traceback.format_exc(e))

if __name__ == '__main__':
    # Load in all msd entries from whoosh index
    index = whoosh_search.get_whoosh_index(
        os.path.join(BASE_DATA_PATH, 'msd', 'index'))
    with index.searcher() as searcher:
        msd_list = list(searcher.documents())
    # Create list of mp3 file paths
    mp3_files = [os.path.join(BASE_DATA_PATH, 'msd', 'mp3', e['path'] + '.mp3')
                 for e in msd_list]
    joblib.Parallel(n_jobs=10, verbose=51)(
        joblib.delayed(process_one_file)(mp3_filename)
        for mp3_filename in mp3_files)
