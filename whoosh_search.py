# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import csv
import whoosh, whoosh.fields, whoosh.index, whoosh.analysis, whoosh.qparser
from whoosh.support.charset import accent_map
import os

# <codecell>

# Code from Brian McFee
def createIndexWriter(indexPath):
    '''
    Constructs a whoosh index writer for the MSD
    
    Input:
        indexPath - Path to whoosh index to be written
    Output:
        index - Whoosh index writer
    '''
    if not os.path.exists(indexPath):
        os.mkdir(indexPath)
        pass

    A = whoosh.analysis.StemmingAnalyzer() | whoosh.analysis.CharsetFilter(accent_map)

    Schema = whoosh.fields.Schema(  track_id    =   whoosh.fields.ID(stored=True),
                                    song_id     =   whoosh.fields.TEXT(stored=True),
                                    artist      =   whoosh.fields.TEXT(stored=True, analyzer=A),
                                    title       =   whoosh.fields.TEXT(stored=True, analyzer=A))

    index = whoosh.index.create_in(indexPath, Schema)
    return index.writer()

# <codecell>

def get_msd_list(csv_file):
    '''
    Parses the unique_tracks.txt file into a python list of lists.
    
    Input:
        csv_file - path to unique_tracks.txt
    Output:
        msd_list - list of lists, each list contains track_id, song_id, artist, title
    '''
    msd_list = []
    with open(csv_file, 'rb') as f:
        msd_list = [line.split('<SEP>') for line in f]
    for n, line in enumerate( msd_list ):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        msd_list[n] = line
    return msd_list

# <codecell>

# Code from Brian McFee
def createIndex(index_path, csv_file):
    '''
    Creates a whoosh index directory for the MSD

    Input:
        index_path - where to create the whoosh index
        csv_file - path to unique_tracks.txt
    '''
    
    writer = createIndexWriter(index_path)
    msd_list = get_msd_list( csv_file )
    
    for (track_id, song_id, artist_name, song_name) in msd_list:
        writer.add_document(    track_id    = track_id,
                                song_id     = song_id,
                                artist      = artist_name,
                                title       = song_name)

        pass
    writer.commit()
    pass

# <codecell>

def get_whoosh_index(index_path):
    '''
    Get a whoosh searcher object from a whoosh index path
    
    Input:
        index_path - path to whoosh index
    Output:
        index - whoosh index
    '''
    return whoosh.index.open_dir(index_path)

# <codecell>

def search( index, string ):
    '''
    Search a whoosh searcher for a string, and return the best match

    Input:
        searcher - whoosh searcher object
        string - whoosh index
    Output:
        best_matches - matches for the search, ordered
    '''
    searcher = index.searcher()
    q = whoosh.qparser.MultifieldParser(['title', 'artist'], index.schema).parse(unicode(string, encoding='utf-8'))
    results = searcher.search(q)
    if len(results) > 0:
        best_matches = [[r['track_id'], r['song_id'], r['artist'], r['title']] for r in results]
        return best_matches
    else:
        return None

# <codecell>

index = get_whoosh_index('whoosh_index')
import pprint
pprint.pprint( search( index, 'thunderstruck' ) )

# <codecell>

index.

# <codecell>

if __name__=='__main__':
    createIndex( 'whoosh_index', 'unique_tracks.txt' )

