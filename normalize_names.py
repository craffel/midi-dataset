import json
import urllib
import unicodedata
import pyen
import collections
FREEBASE_KEY = open('.freebase_key').read()
ECHONEST_KEY = open('.echonest_key').read()
FREEBASE_URL = 'https://www.googleapis.com/freebase/v1/search?'


def clean(string):
    '''
    Removes non-ascii characters from a string in a semi-smart way

    :parameters:
        - string : str or unicode
            String to clean

    :returns:
        - clean_string : str
            ASCII string
    '''
    if type(string) == str:
        string = unicode(string, 'utf-8', 'ignore')
    return unicodedata.normalize('NFKD', string).encode('ascii', 'ignore')


def echonest_normalize_artist(artists):
    '''
    Normalize artist names using echonest

    :parameters:
        - artists : str or list of str
            Query artist name or list of potential artist names
        - titles : str or list of str
            Query title or list of potential song titles

    :returns:
        - artists : list of str
            Unique list of matching artists
    '''
    if type(artists) == str or type(artists) == unicode:
        artists = [artists]

    matched_artists = []

    en = pyen.Pyen(api_key=ECHONEST_KEY)

    for query_artist in artists:
        success = False
        while not success:
            try:
                response = en.get('artist/search',
                                  name=clean(query_artist),
                                  results=5,
                                  fuzzy_match='true')
            except pyen.PyenException as e:
                print e.message, e.args
                continue
            success = True
        if len(response['artists']) > 0:
            for matched_artist in response['artists']:
                matched_artists.append(matched_artist['name'])
    if len(matched_artists) == 0:
        return None
    matched_artists = list(collections.OrderedDict.fromkeys(matched_artists))
    return matched_artists


def freebase_normalize_title(artists, titles):
    '''
    Normalize a song title using freebase

    :parameters:
        - artists : str or list of str
            Query artist name or list of potential artist names
        - titles : str or list of str
            Query title or list of potential song titles

    :returns:
        - artist : str or NoneType
            Freebase's chosen title out of the supplied list or None if no match
        - title: str or NoneType
            Freebase's purported title or None if no match
    '''
    def title_match(artist, title, old_correction=False):
        ''' Match a song title with some artist using freebase '''
        filter_str = '(all type:/music/recording /music/recording/artist:"{}")'
        params = {'query': clean(title),
                  'filter': filter_str.format(clean(artist).replace('"', '')),
                  'limit': 1,
                  'key': FREEBASE_KEY,
                  'spell': 'always'}
        url = FREEBASE_URL + urllib.urlencode(params)
        success = False
        while not success:
            try:
                response = json.loads(urllib.urlopen(url).read())
            except Exception as e:
                print e.message, e.args
                continue
            if 'result' in response:
                success = True
            else:
                print 'result not in response: {}'.format(response)
        if len(response['result']) > 0:
            return response['result'][0]['name']
        if 'correction' in response:
            if old_correction:
                return None
            else:
                return title_match(artist,
                                   response['correction'][0],
                                   True)
        return None

    if type(artists) == str or type(artists) == unicode:
        artists = [artists]

    if type(titles) == str or type(artists) == unicode:
        titles = [titles]

    for query_artist in artists:
        for query_title in titles:
            title = title_match(query_artist, query_title)
            if title is not None:
                return query_artist, title
    return None, None
