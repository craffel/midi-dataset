import os
import sys
sys.path.append('..')
import whoosh_search

FILE_LIST_PATH = '../file_lists/'
BASE_DATA_PATH = '../data'


def get_sv_list(sv_file, delimiter='\t', skiplines=0, field_indices=None):
    '''
    Parses a delimiter-separated value file

    :parameters:
        - sv_file : str
            Path to the separated value file
        - skiplines : int
            Number of lines to skip at the beginning of the file
        - delimiter : str
            Delimiter used to separate values
        - field_indices : list of int or NoneType
            Desired field indices, if None then return all fields

    :returns:
        - sv_list : list of list
            One list per row of the sv file
    '''
    sv_list = []
    with open(sv_file, 'rb') as f:
        for line in f:
            fields = line.split(delimiter)
            if field_indices is None:
                sv_list.append(fields)
            else:
                sv_list.append([fields[n] for n in field_indices])
    # Remove first line - labels
    sv_list = sv_list[skiplines:]
    for n, line in enumerate(sv_list):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        sv_list[n] = line
    return sv_list

if not os.path.exists(os.path.join(BASE_DATA_PATH, 'cal500', 'index')):
    # Load in cal500 list
    cal500_list = get_sv_list(
        os.path.join(FILE_LIST_PATH, 'cal500.txt'))
    # Construct list of dicts of entries
    cal500_list = [
        # cal500.txt doesn't include unique IDs; create them from their index
        {'id': unicode(n), 'artist': row[1], 'title': row[2],
         # cal500.txt doesn't include path; we must construct it
         'path': u"{}-{}".format(row[1], row[2]).replace(' ', '_')}
        for n, row in enumerate(cal500_list)]
    whoosh_search.create_index(
        os.path.join(BASE_DATA_PATH, 'cal500', 'index'), cal500_list)

if not os.path.exists(os.path.join(BASE_DATA_PATH, 'cal10k', 'index')):
    cal10k_list = get_sv_list(
        os.path.join(FILE_LIST_PATH, 'cal10k.txt'), skiplines=1)
    cal10k_list = [
        # cal10k.txt doesn't include unique IDs; create them from their index
        {'id': unicode(n), 'artist': row[1], 'title': row[2],
         # cal10k.txt doesn't include path; we must construct it
         'path': u"{} - {}".format(row[1], row[2])}
        for n, row in enumerate(cal10k_list)]
    whoosh_search.create_index(
        os.path.join(BASE_DATA_PATH, 'cal10k', 'index'), cal10k_list)

if not os.path.exists(os.path.join(BASE_DATA_PATH, 'msd', 'index')):
    # Simple function which converts track ID to path
    to_path = lambda tid: os.path.join(tid[2], tid[3], tid[4], tid)
    msd_list = get_sv_list(
        os.path.join(FILE_LIST_PATH, 'msd.txt'), delimiter='<SEP>')
    msd_list = [{'id': row[0], 'artist': row[2], 'title': row[3],
                 'path': to_path(row[0])} for row in msd_list]
    whoosh_search.create_index(
        os.path.join(BASE_DATA_PATH, 'msd', 'index'), msd_list)

if not os.path.exists(os.path.join(BASE_DATA_PATH, 'clean_midi', 'index')):
    clean_midi_list = get_sv_list(
        os.path.join(FILE_LIST_PATH, 'clean_midi.txt'))
    # We'll use row[3], the md5, as the ID
    clean_midi_list = [{'id': row[3], 'artist': row[1], 'title': row[2],
                        # Must remove .mid from the path to get generic path
                        'path': row[4].replace('.mid', '')}
                       for row in clean_midi_list]
    whoosh_search.create_index(
        os.path.join(BASE_DATA_PATH, 'clean_midi', 'index'), clean_midi_list)

if not os.path.exists(os.path.join(BASE_DATA_PATH, 'uspop2002', 'index')):
    uspop2002_list = get_sv_list(
        os.path.join(FILE_LIST_PATH, 'uspop2002.txt'))
    uspop2002_list = [{'id': unicode(n), 'artist': row[1], 'title': row[3],
                       # Must remove .mp3 from the path to get generic path
                       'path': row[4].replace('.mp3', '')}
                      for n, row in enumerate(uspop2002_list)]
    whoosh_search.create_index(
        os.path.join(BASE_DATA_PATH, 'uspop2002', 'index'), uspop2002_list)

# Quick test
artist = 'bon jovi'
title = 'livin on a prayer'

index = whoosh_search.get_whoosh_index(
    os.path.join(BASE_DATA_PATH, 'cal500', 'index'))
with index.searcher() as searcher:
    print 'cal500:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                        artist, title))

index = whoosh_search.get_whoosh_index(
    os.path.join(BASE_DATA_PATH, 'cal10k', 'index'))
with index.searcher() as searcher:
    print 'cal10k:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                        artist, title))

index = whoosh_search.get_whoosh_index(
    os.path.join(BASE_DATA_PATH, 'msd', 'index'))
with index.searcher() as searcher:
    print 'msd:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                                 artist, title))

index = whoosh_search.get_whoosh_index(
    os.path.join(BASE_DATA_PATH, 'uspop2002', 'index'))
with index.searcher() as searcher:
    print 'uspop2002:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                            artist, title))

index = whoosh_search.get_whoosh_index(
    os.path.join(BASE_DATA_PATH, 'clean_midi', 'index'))
with index.searcher() as searcher:
    print 'clean_midi:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                            artist, title))
