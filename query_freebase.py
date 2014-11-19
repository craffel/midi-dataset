import json
import urllib
import unicodedata
API_KEY = 'AIzaSyAMo_6FhlxCgdTXq5lo9_hm4zLCXtEETOc'
FREEBASE_URL = 'https://www.googleapis.com/freebase/v1/search'


def normalize(artists, titles):
    '''
    Normalize an artist/title pair using freebase

    :parameters:
        - artist : str or list of str
            Query artist name or list of potential artist names
        - title : str or list of str
            Query title or list of potential song titles

    :returns:
        - artist : str or NoneType
            Freebase's purported artist name or None if no match
        - title: str or NoneType
            Freebase's purported title or None if no match
    '''
    def clean(string):
        ''' Removes non-ascii characters from a string '''
        if type(string) == str:
            string = unicode(string, 'utf-8', 'ignore')
        return unicodedata.normalize('NFKD', string).encode('ascii', 'ignore')

    def artist_match(artist, old_correction=None):
        ''' Match an artist using freebase '''
        filter_str = '(all name:"{}" type:/music/artist)'
        params = {'query': clean(artist),
                  'filter': filter_str.format(clean(artist)),
                  'limit': 1,
                  'key': API_KEY,
                  'spell': 'always'}
        url = FREEBASE_URL + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read())
        if 'correction' in response:
            correction = response['correction'][0]
            if correction == old_correction:
                return correction
            else:
                return artist_match(response['correction'][0], correction)
        if len(response['result']) > 0:
            return response['result'][0]['name']
        else:
            return None

    def title_match(artist, title, old_correction=None):
        ''' Match a song title with some artist using freebase '''
        filter_str = '(all type:/music/recording /music/recording/artist:"{}")'
        params = {'query': clean(title),
                  'filter': filter_str.format(clean(artist)),
                  'limit': 1,
                  'key': API_KEY,
                  'spell': 'always'}
        url = FREEBASE_URL + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read())
        if 'correction' in response:
            correction = response['correction'][0]
            if correction == old_correction:
                return correction
            else:
                return title_match(artist,
                                   response['correction'][0],
                                   correction)
        if len(response['result']) > 0:
            return response['result'][0]['name']
        else:
            return None

    if type(artists) == str or type(artists) == unicode:
        artists = [artists]

    for query_artist in artists:
        artist = artist_match(query_artist)
        if artist is not None:
            break
    if artist is None:
        return None, None

    if type(titles) == str or type(artists) == unicode:
        titles = [titles]

    for query_title in titles:
        title = title_match(artist, query_title)
        if title is not None:
            break

    return artist, title
