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
def create_index_writer(index_path):
    '''
    Constructs a whoosh index writer, which has an ID field as well as artist and title
    
    Input:
        index_path - Path to whoosh index to be written
    Output:
        index - Whoosh index writer
    '''
    if not os.path.exists(index_path):
        os.mkdir(index_path)

    A = whoosh.analysis.StemmingAnalyzer() | whoosh.analysis.CharsetFilter(accent_map)

    Schema = whoosh.fields.Schema(  track_id    =   whoosh.fields.ID(stored=True),
                                    artist      =   whoosh.fields.TEXT(stored=True, analyzer=A),
                                    title       =   whoosh.fields.TEXT(stored=True, analyzer=A))

    index = whoosh.index.create_in(index_path, Schema)
    return index.writer()

# <codecell>

def get_msd_list(csv_file):
    '''
    Parses the unique_tracks.txt file into a python list of lists.
    
    Input:
        csv_file - path to unique_tracks.txt
    Output:
        msd_list - list of lists, each list contains track_id, artist, title
    '''
    msd_list = []
    with open(csv_file, 'rb') as f:
        for line in f:
            fields = line.split('<SEP>')
            msd_list += [[fields[0], fields[2], fields[3]]]
    for n, line in enumerate( msd_list ):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        msd_list[n] = line
    return msd_list

# <codecell>

def get_cal10k_list(csv_file):
    '''
    Parses the EchoNestTrackIDs.tab file into a python list of lists.
    
    Input:
        csv_file - path to unique_tracks.txt
    Output:
        cal10k_list - list of lists, each list contains track_id, artist, title
    '''
    cal10k_list = []
    with open(csv_file, 'rb') as f:
        for line in f:
            fields = line.split('\t')
            cal10k_list += [[fields[0], fields[2], fields[1]]]
    # Remove first line - labels
    cal10k_list = cal10k_list[1:]
    for n, line in enumerate( cal10k_list ):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        cal10k_list[n] = line
    return cal10k_list

# <codecell>

# Code from Brian McFee
def createIndex(index_path, track_list):
    '''
    Creates a whoosh index directory for the MSD

    Input:
        index_path - where to create the whoosh index
        track_list - list of lists, each list contains track_id, artist, title
    '''
    
    writer = createIndexWriter(index_path)
    
    for (track_id, artist_name, song_name) in track_list:
        writer.add_document(    track_id    = track_id,
                                artist      = artist_name,
                                title       = song_name)

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
        best_match - best match for the search, or None of no match
    '''
    searcher = index.searcher()
    q = whoosh.qparser.MultifieldParser(['title', 'artist'], index.schema).parse(unicode(string, encoding='utf-8'))
    results = searcher.search(q)
    if len(results) > 0:
        r = results[0]
        return [r['track_id'], r['artist'], r['title']]
    else:
        return None

# <codecell>

if __name__=='__main__':
    import os
    if not os.path.exists('Whoosh Indices/cal10k_index/'):
        createIndex('Whoosh Indices/cal10k_index/', get_cal10k_list('File Lists/EchoNestTrackIDs.tab') )
    index = get_whoosh_index('Whoosh Indices/cal10k_index/')
    print search( index, '50 cent' )

