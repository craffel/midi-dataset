'''
Given a file listing the live tracks in uspop2002 and a file list of all the
tracks, create a list of only the non-live tracks.
'''
import read_sv

all_tracks = read_sv.get_sv_list('../file_lists/uspop2002.txt')
live_tracks = read_sv.get_sv_list('../file_lists/uspop2002_live.txt')

studio_tracks = [track for track in all_tracks if track not in live_tracks]
# Correct indices
studio_tracks = [[str(n)] + track[1:] for n, track in enumerate(studio_tracks)]

with open('../file_lists/uspop2002_no_live.txt', 'wb') as f:
    for track in studio_tracks:
        f.write('\t'.join(track) + '\n')
