import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import normalize_names
import pickle

with open('data/Clean MIDIs-md5_to_artist_title.pickle') as f:
    md5_to_artist_title = pickle.load(f)

with open('data/Clean MIDIs-md5_to_path.pickle') as f:
    md5_to_path = pickle.load(f)

md5_to_freebase_artist_title = {}

for n, md5 in enumerate(md5_to_path):
    artists_titles = md5_to_artist_title[md5]
    artists = [artist_title[0] for artist_title in artists_titles]
    titles = [artist_title[1] for artist_title in artists_titles]
    for n, title in enumerate(titles):
        # Some titles have " l" appended to the end which trips up freebase
        if title[-2:] == ' l':
            titles[n] = title[:-2]
    print artists, titles
    resolved_artists = normalize_names.echonest_normalize_artist(artists)
    if resolved_artists is not None:
        resolved_artist, resolved_title = \
            normalize_names.freebase_normalize_title(resolved_artists, titles)
        if resolved_artist is not None and resolved_title is not None:
            md5_to_freebase_artist_title[md5] = [resolved_artist,
                                                 resolved_title]
            print '\t', resolved_artist, '-', resolved_title
            print

with open('data/Clean MIDIs-md5_to_freebase_artist_title.pickle', 'wb') as f:
    pickle.dump(md5_to_freebase_artist_title, f)
