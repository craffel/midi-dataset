import pyen
import sys
sys.path.append('../')
import os
import time

ECHONEST_KEY = open('../.echonest_key').read()
BASE_DATA_PATH = '../data/'

en = pyen.Pyen(ECHONEST_KEY)


def wait_for_analysis(id):
    while True:
        response = en.get('track/profile', id=id, bucket=['audio_summary'])
        if response['track']['status'] != 'pending':
            break
        time.sleep(1)

    return response['track']['audio_summary']

uspop_tracks = []
with open('../file_lists/uspop2002.txt') as f:
    for line in f:
        uspop_tracks.append(line.rstrip().split('\t'))

with open('../file_lists/uspop2002_liveness.txt', 'wb') as tsv_file:
    for track in uspop_tracks:
        filename = os.path.join(BASE_DATA_PATH, 'uspop2002', 'mp3', track[-1])
        success = False
        while not success:
            try:
                with open(filename, 'rb') as f:
                    result = en.post('track/upload', track=f, filetype='mp3')
                success = True
            except Exception as e:
                print '  ## Bad connection {}'.format(e.message)
                time.sleep(10)
        try:
            track_id = result['track']['id']
            result = wait_for_analysis(track_id)
            line = track + [str(result['liveness'])]
            print line
            tsv_file.write('\t'.join(line) + '\n')
        except Exception as e:
            print '  ## Bad result: {}\n  {}'.format(filename, e.message)
