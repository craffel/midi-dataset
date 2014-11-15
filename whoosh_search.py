# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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


def get_tsv_list(tsv_file, skiplines=0):
    '''
    Parses the EchoNestTrackIDs.tab file into a python list of lists.

    Input:
        tsv_file - path to EchoNestTrackIDs.tab
    Output:
        cal10k_list - list of lists, each list contains track_id, artist, title
    '''
    tsv_list = []
    with open(tsv_file, 'rb') as f:
        for line in f:
            fields = line.split('\t')
            tsv_list += [[fields[0], fields[1], fields[2]]]
    # Remove first line - labels
    tsv_list = tsv_list[skiplines:]
    for n, line in enumerate(tsv_list):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        tsv_list[n] = line
    return tsv_list

# <codecell>

# Code from Brian McFee
def create_index(index_path, track_list):
    '''
    Creates a whoosh index directory for the MSD

    Input:
        index_path - where to create the whoosh index
        track_list - list of lists, each list contains track_id, artist, title
    '''
    
    writer = create_index_writer(index_path)
    
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

def search( searcher, schema, artist, title, threshold=20 ):
    '''
    Search for an artist - title pair and return the best match

    Input:
        searcher - whoosh searcher (create with index.searcher() then close it yourself)
        schema - whoosh schema (index.schema)
        artist - Artist name
        title - Song name
        threshold - Score threshold for a match
    Output:
        best_match - best match for the search, or None of no match
    '''
    arparser = whoosh.qparser.QueryParser('artist', schema)
    tiparser = whoosh.qparser.QueryParser('title', schema)
    q = whoosh.query.And([arparser.parse(unicode(artist, encoding='utf-8')), tiparser.parse(unicode(title, encoding='utf-8'))])
    results = searcher.search(q)
    result = None
    
    if len(results) > 0:
        r = results[0]
        if r.score > threshold:
            result = [r['track_id'], r['artist'], r['title']]
    
    return result


if __name__ == '__main__':
    import os
    if not os.path.exists('whoosh_indices/cal500_index/'):
        create_index('whoosh_indices/cal500_index/',
                     get_tsv_list('File Lists/cal500.txt'))
    if not os.path.exists('whoosh_indices/cal10k_index/'):
        create_index('whoosh_indices/cal10k_index/',
                     get_tsv_list('File Lists/EchoNestTrackIDs.tab', 1))
    if not os.path.exists('whoosh_indices/msd_index/'):
        create_index('whoosh_indices/msd_index/',
                     get_msd_list('File Lists/unique_tracks.txt'))
    index = get_whoosh_index('whoosh_indices/cal500_index/')
    with index.searcher() as searcher:
        print 'cal500:\t{}'.format(search(searcher, index.schema,
                                         'bon jovi', 'livin on a prayer'))
    index = get_whoosh_index('whoosh_indices/cal10k_index/')
    with index.searcher() as searcher:
        print 'cal10k:\t{}'.format(search(searcher, index.schema,
                                         'bon jovi', 'livin on a prayer'))
    index = get_whoosh_index('whoosh_indices/msd_index/')
    with index.searcher() as searcher:
        print 'msd:\t{}'.format(search(searcher, index.schema,
                                      'bon jovi', 'livin on a prayer'))
