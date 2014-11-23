import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import whoosh_search
import read_sv

if not os.path.exists('data/cal500/index/'):
    whoosh_search.create_index('data/cal500/index/',
        read_sv.get_sv_list('file_lists/cal500.txt',
                            field_indices=[0, 1, 2]))
if not os.path.exists('data/cal10k/index/'):
    whoosh_search.create_index('data/cal10k/index/',
        read_sv.get_sv_list('file_lists/EchoNestTrackIDs.tab',
                            skiplines=1, field_indices=[0, 1, 2]))
if not os.path.exists('data/msd/index/'):
    whoosh_search.create_index('data/msd/index/',
        read_sv.get_sv_list('file_lists/unique_tracks.txt',
                            delimiter='<SEP>',
                            field_indices=[0, 2, 3]))
if not os.path.exists('data/clean_midi/index'):
    whoosh_search.create_index('data/clean_midi/index',
        read_sv.get_sv_list('file_lists/clean_midi.txt',
                            field_indices=[0, 1, 2]))
if not os.path.exists('data/uspop2002/index'):
    whoosh_search.create_index('data/uspop2002/index',
        read_sv.get_sv_list('file_lists/uspop2002.txt',
                            field_indices=[0, 1, 3]))

artist = 'bon jovi'
title = 'livin on a prayer'

index = whoosh_search.get_whoosh_index('data/cal500/index/')
with index.searcher() as searcher:
    print 'cal500:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                        artist, title))

index = whoosh_search.get_whoosh_index('data/cal10k/index/')
with index.searcher() as searcher:
    print 'cal10k:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                        artist, title))

index = whoosh_search.get_whoosh_index('data/msd/index/')
with index.searcher() as searcher:
    print 'msd:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                                 artist, title))

index = whoosh_search.get_whoosh_index('data/uspop2002/index/')
with index.searcher() as searcher:
    print 'uspop2002:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                            artist, title))

index = whoosh_search.get_whoosh_index('data/clean_midi/index/')
with index.searcher() as searcher:
    print 'clean_midi:\t{}'.format(whoosh_search.search(searcher, index.schema,
                                            artist, title))
