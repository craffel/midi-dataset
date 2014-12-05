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
    if type(artist) != unicode:
        artist = unicode(artist, encoding='utf-8')
    if type(title) != unicode:
        title = unicode(title, encoding='utf-8')
    arparser = whoosh.qparser.QueryParser('artist', schema)
    tiparser = whoosh.qparser.QueryParser('title', schema)
    q = whoosh.query.And([arparser.parse(artist), tiparser.parse(title)])
    results = searcher.search(q)

    if len(results) > 0:
        return [[r['track_id'], r['artist'], r['title']] for r in results if
                r.score > threshold]
    else:
        return []
