import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import json_utils
import read_sv


data = read_sv.get_sv_list('file_lists/cal500.txt')
data = [[row[1], row[2], "{}-{}".format(row[1], row[2]).replace(' ', '_')]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'title', 'path'],
                     'data/cal500/index.js')

data = read_sv.get_sv_list('file_lists/EchoNestTrackIDs.tab', skiplines=1)
data = [[row[1], row[2], row[4], u"{} - {}".format(row[1], row[2])]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'title', 'en_id', 'path'],
                     'data/cal10k/index.js')
data = read_sv.get_sv_list('file_lists/clean_midi.txt')
data = [[row[1], row[2], row[3], row[4].replace('.mid', '')]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'title', 'md5', 'path'],
                     'data/clean_midi/index.js')

data = read_sv.get_sv_list('file_lists/uspop2002_no_live.txt')
data = [[row[1], row[2], row[3], row[4].replace('.mp3', '')]
        for row in data]
json_utils.create_js(data,
                     ['artist', 'album', 'title', 'path'],
                     'data/uspop2002/index.js')

data = read_sv.get_sv_list('file_lists/unique_tracks.txt', '<SEP>')
to_path = lambda tid: os.path.join(tid[2], tid[3], tid[4], tid)
data = [[row[0], row[1], row[2], row[3], to_path(row[0])] for row in data]
json_utils.create_js(data,
                     ['track_id', 'song_id', 'artist', 'title', 'path'],
                     'data/msd/index.js')
