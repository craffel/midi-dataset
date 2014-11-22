import whoosh
import whoosh.fields
import whoosh.index
import whoosh.analysis
import whoosh.qparser
from whoosh.support.charset import accent_map
import os


# Code from Brian McFee
def create_index_writer(index_path):
    '''
    Constructs a whoosh index writer, which has ID, artist and title fields

    :parameters:
        - index_path : str
            Path to whoosh index to be written

    :returns:
        - index : whoosh.writing.IndexWriter
            Whoosh index writer
    '''
    if not os.path.exists(index_path):
        os.mkdir(index_path)

    A = (whoosh.analysis.StemmingAnalyzer() |
         whoosh.analysis.CharsetFilter(accent_map))

    Schema = whoosh.fields.Schema(
        track_id=whoosh.fields.ID(stored=True),
        artist=whoosh.fields.TEXT(stored=True, analyzer=A),
        title=whoosh.fields.TEXT(stored=True, analyzer=A))

    index = whoosh.index.create_in(index_path, Schema)
    return index.writer()


def get_sv_list(sv_file, delimiter='\t', skiplines=0, field_indices=None):
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
        - field_indices : list of int
            Field indices for [id, artist, title], default [0, 1, 2]

    :returns:
        - sv_list : list of list
            Each list contains track_id, artist, title
    '''
    sv_list = []
    if field_indices is None:
        field_indices = [0, 1, 2]
    with open(sv_file, 'rb') as f:
        for line in f:
            fields = line.split(delimiter)
            sv_list.append([fields[n] for n in field_indices])
    # Remove first line - labels
    sv_list = sv_list[skiplines:]
    for n, line in enumerate(sv_list):
        line = [unicode(a.rstrip(), encoding='utf-8') for a in line]
        sv_list[n] = line
    return sv_list


# Code from Brian McFee
def create_index(index_path, track_list):
    '''
    Creates a whoosh index directory for the MSD

    :parameters:
        - index_path : str
            where to create the whoosh index
        - track_list : list of list of str
            list of lists, each list contains track_id, artist, title
    '''

    writer = create_index_writer(index_path)

    for (track_id, artist_name, song_name) in track_list:
        writer.add_document(track_id=track_id,
                            artist=artist_name,
                            title=song_name)

    writer.commit()
    pass


def get_whoosh_index(index_path):
    '''
    Get a whoosh searcher object from a whoosh index path

    :parameters:
        index_path - path to whoosh index

    :returns:
        index - whoosh index
    '''
    return whoosh.index.open_dir(index_path)


def search(searcher, schema, artist, title, threshold=20):
    '''
    Search for an artist - title pair and return the best match

    :usage:
        >>> index = whoosh_search.get_whoosh_index('/path/to/index/')
        >>> with index.searcher() as searcher:
        >>>     whoosh_search.search(searcher, index.schema, 'artist', 'title')

    :parameters:
        - searcher : whoosh.searching.Searcher
            Create with index.searcher() then close it yourself
        - schema : whoosh.fields.Schema
            E.g. index.schema
        - artist : str
            Artist name to search for
        - title : str
            Song title to search for
        - threshold : float
            A result must have a score higher than this to be a match

    :returns:
        - matches : list of list
            List of match lists of the form [id, artist, title]
    '''
    arparser = whoosh.qparser.QueryParser('artist', schema)
    tiparser = whoosh.qparser.QueryParser('title', schema)
    q = whoosh.query.And([arparser.parse(unicode(artist, encoding='utf-8')),
                          tiparser.parse(unicode(title, encoding='utf-8'))])
    results = searcher.search(q)

    if len(results) > 0:
        return [[r['track_id'], r['artist'], r['title']] for r in results if
                r.score > threshold]
    else:
        return None


if __name__ == '__main__':
    import os
    if not os.path.exists('data/cal500/index/'):
        create_index('data/cal500/index/',
                     get_sv_list('file_lists/cal500.txt'))
    if not os.path.exists('data/cal10k/index/'):
        create_index('data/cal10k/index/',
                     get_sv_list('file_lists/EchoNestTrackIDs.tab',
                                 skiplines=1))
    if not os.path.exists('data/msd/index/'):
        create_index('data/msd/index/',
                     get_sv_list('file_lists/unique_tracks.txt',
                                 delimiter='<SEP>',
                                 field_indices=[0, 2, 3]))
    if not os.path.exists('data/clean_midi/index'):
        create_index('data/clean_midi/index',
                     get_sv_list('file_lists/clean_midi.txt'))
    if not os.path.exists('data/uspop2002/index'):
        create_index('data/uspop2002/index',
                     get_sv_list('file_lists/uspop2002.txt',
                                 field_indices=[0, 1, 3]))

    artist = 'bon jovi'
    title = 'livin on a prayer'

    index = get_whoosh_index('data/cal500/index/')
    with index.searcher() as searcher:
        print 'cal500:\t{}'.format(search(searcher, index.schema,
                                          artist, title))

    index = get_whoosh_index('data/cal10k/index/')
    with index.searcher() as searcher:
        print 'cal10k:\t{}'.format(search(searcher, index.schema,
                                          artist, title))

    index = get_whoosh_index('data/msd/index/')
    with index.searcher() as searcher:
        print 'msd:\t{}'.format(search(searcher, index.schema, artist, title))

    index = get_whoosh_index('data/uspop2002/index/')
    with index.searcher() as searcher:
        print 'uspop2002:\t{}'.format(search(searcher, index.schema,
                                             artist, title))

    index = get_whoosh_index('data/clean_midi/index/')
    with index.searcher() as searcher:
        print 'clean_midi:\t{}'.format(search(searcher, index.schema,
                                              artist, title))
