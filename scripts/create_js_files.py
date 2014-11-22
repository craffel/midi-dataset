import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import json_utils


def get_sv_list(sv_file, delimiter='\t', skiplines=0):
    '''
    Parses a delimiter-separated value file where each line has the format

    track_id (delimiter) artist (delimiter) title (delimiter) ignored ...

    :parameters:
        - sv_file : str
            Path to the separated value file
        - skiplines : int
            Number of lines to skip at the beginning of the file
        - delimiter : str
            Delimiter used to separate values

    :returns:
        - sv_list : list of list
            Each list contains track_id, artist, title
    '''
    sv_list = []
    with open(sv_file, 'rb') as f:
        for line in f:
            fields = line.split(delimiter)
            sv_list.append(fields)
    # Remove first line - labels
    sv_list = sv_list[skiplines:]
    for n, line in enumerate(sv_list):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        sv_list[n] = line
    return sv_list


data = get_sv_list('file_lists/cal500.txt')
data = [[row[1], row[2], "{}_-_{}".format(row[1], row[2]).replace(' ', '_')]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'title', 'path'],
                     'data/cal500/index.js')

data = get_sv_list('file_lists/EchoNestTrackIDs.tab', skiplines=1)
data = [[row[1], row[2], row[4], u"{} - {}".format(row[1], row[2])]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'title', 'en_id', 'path'],
                     'data/cal10k/index.js')
data = get_sv_list('file_lists/clean_midi.txt')
data = [[row[1], row[2], row[3], row[4].replace('.mid', '')]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'title', 'md5', 'path'],
                     'data/clean_midi/index.js')

data = get_sv_list('file_lists/uspop2002.txt')
data = [[row[1], row[2], row[3], row[4].replace('.mp3', '')]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'album', 'title', 'path'],
                     'data/uspop2002/index.js')

data = get_sv_list('file_lists/unique_tracks.txt', '<SEP>')
to_path = lambda tid: os.path.join(tid[2], tid[3], tid[4], tid)
data = [[row[0], row[1], row[2], row[3], to_path(row[0])] for row in data]
json_utils.create_js(data,
                     ['tracK_id', 'song_id', 'artist', 'title', 'path'],
                     'data/msd/index.js')
