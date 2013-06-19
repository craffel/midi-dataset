# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import json
import urllib
API_KEY = 'AIzaSyAMo_6FhlxCgdTXq5lo9_hm4zLCXtEETOc'

# <codecell>

def query( query, collection ):
    '''
    Send a query to freebase, and get the name or correction

    Input:
        query - the term to search for, eg 'brintey_spares'
        collection - the collection to search in, eg '/music/artist' or '/music/recording'
    Output:
        name - the name of the entity, corrected and cleaned up
    '''
    service_url = 'https://www.googleapis.com/freebase/v1/search'
    params = {
    'query': query,
    'filter': '(all type:{})'.format( collection ),
    'limit': 10,
    'indent': True,
    'key': API_KEY,
    'spell': 'always'
    }
    url = service_url + '?' + urllib.urlencode(params)
    response = json.loads(urllib.urlopen(url).read())
    if response.has_key( 'correction' ):
        return str( response['correction'][0] )
    if len( response['result'] ) > 0:
        return response['result'][0]['name']

# <codecell>

if __name__ == '__main__':
    print query( 'brittneyspaers', '/music/artist' )
    print query( '_bleeding_me', '/music/recording' )

