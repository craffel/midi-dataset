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

    Input:
        index_path - Path to whoosh index to be written
    Output:
        index - Whoosh index writer
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
            fields = line.split('\t')
            sv_list.append([fields[0], fields[1], fields[2]])
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

    Input:
        index_path - where to create the whoosh index
        track_list - list of lists, each list contains track_id, artist, title
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

    Input:
        index_path - path to whoosh index
    Output:
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

    :return:
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
    if not os.path.exists('whoosh_indices/cal500_index/'):
        create_index('whoosh_indices/cal500_index/',
                     get_sv_list('file_lists/cal500.txt'))
    if not os.path.exists('whoosh_indices/cal10k_index/'):
        create_index('whoosh_indices/cal10k_index/',
                     get_sv_list('file_lists/EchoNestTrackIDs.tab',
                                 skiplines=1))
    if not os.path.exists('whoosh_indices/msd_index/'):
        create_index('whoosh_indices/msd_index/',
                     get_sv_list('file_lists/unique_tracks.txt',
                                 delimiter='<SEP>'))
    if not os.path.exists('whoosh_indices/clean_midis_index/'):
        create_index('whoosh_indices/clean_midis_index/',
                     get_sv_list('file_lists/clean_midis.txt'))

    artist = 'bon jovi'
    title = 'livin on a prayer'

    index = get_whoosh_index('whoosh_indices/cal500_index/')
    with index.searcher() as searcher:
        print 'cal500:\t{}'.format(search(searcher, index.schema,
                                          artist, title))
    index = get_whoosh_index('whoosh_indices/cal10k_index/')
    with index.searcher() as searcher:
        print 'cal10k:\t{}'.format(search(searcher, index.schema,
                                          artist, title))
    index = get_whoosh_index('whoosh_indices/msd_index/')
    with index.searcher() as searcher:
        print 'msd:\t{}'.format(search(searcher, index.schema, artist, title))

    index = get_whoosh_index('whoosh_indices/clean_midis_index/')
    with index.searcher() as searcher:
        print 'clean_midis:\t{}'.format(search(searcher, index.schema,
                                               artist, title))
