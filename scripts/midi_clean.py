# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import sys
import numpy as np
import hashlib
import pickle

# <codecell>

path = '../data/Clean MIDIs/'

# <codecell>

def split_all_extensions( f ):
    '''
    Returns a filename with all extensions removed
    '''
    while os.path.splitext(f)[1] != '':
        f = os.path.splitext(f)[0]
    return f

# <codecell>

def safe_rename( old_path, new_path ):
    ''' 
    Moves a file, but if the destination exists it appends a number to the filename.
    '''
    if not os.path.exists( new_path ):
        os.renames( old_path, new_path )
    else:
        n = 1
        new_path = split_all_extensions(new_path) + os.path.splitext(new_path)[1]
        while os.path.exists( os.path.splitext(new_path)[0] + '.{}.mid'.format( n ) ):
            n += 1
        new_path = os.path.splitext(new_path)[0] + '.{}.mid'.format( n )
        os.renames( old_path, new_path )

# <codecell>

def convert_camelCase( string ):
    '''
    Replaces any camelCase with camel Case
    '''
    lowers = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    uppers = [s.upper() for s in lowers]
    camel_case_spots = np.flatnonzero(np.array([b in lowers and c in uppers for b, c in zip( string[:-1], string[1:] )]))
    if camel_case_spots.shape == (0,):
        return None
    shift = 1
    for n in camel_case_spots:
        string = string[:n + shift] + ' ' + string[n + shift:]
        shift += 1
    return string

# <codecell>

# Remove small and non-midi files, and rename .kar to .mid
for root, subdirectories, files in os.walk( path ):
    for f in files:
        if os.path.splitext(f)[1].lower() == '.kar':
            os.rename( os.path.join( root, f ), os.path.join( root, os.path.splitext(f)[0] + '.mid' ) )
        elif os.path.splitext(f)[1].lower() != '.mid':
            os.remove( os.path.join(root, f) )
        elif os.path.getsize( os.path.join(root, f) ) < 2000:
            os.remove( os.path.join(root, f) )

# <codecell>

# Flatten subdirectories
for root, subdirectories, files in os.walk( path ):
    for f in files:
        if len( os.path.join( root, f ).split('/') ) > 5:
            new_path = '/'.join( os.path.join( root, f ).split('/')[:4] + [f] )
            safe_rename( os.path.join( root, f ), new_path )

# <codecell>

# Remove empty subdirectories
for root, subdirectories, files in os.walk( path ):
    for subdirectory in subdirectories:
        if os.listdir( os.path.join(root, subdirectory) ) == [] or os.listdir( os.path.join(root, subdirectory) ) == ['.DS_Store']:
            os.rmdir( os.path.join(root, subdirectory) )

# <codecell>

# Remove duplicates
md5dict = {}
for root, subdirectories, files in os.walk( path ):
    for f in files:
        md5 = hashlib.md5( open( os.path.join( root, f )).read() )
        md5 = md5.hexdigest()
        if md5 in md5dict:
            os.remove( os.path.join(root, f) )
            md5dict[md5] += [os.path.join( root, f )]
        else:
            md5dict[md5] = [os.path.join( root, f )]

# <codecell>

# Convert CamelCase to Camel Case in subdirectories
for root, subdirectories, files in os.walk( path ):
    for subdirectory in subdirectories:
        if convert_camelCase( subdirectory ) is not None:
            safe_rename( os.path.join(root, subdirectory), os.path.join(root, convert_camelCase(subdirectory) ) )

# <codecell>

# Convert CamelCase to Camel Case in files
for root, subdirectories, files in os.walk( path ):
    for f in files:
        if convert_camelCase( f ) is not None:
            safe_rename( os.path.join(root, f), os.path.join(root, convert_camelCase(f) ) )

# <codecell>

# Replace _ and - with space
for root, subdirectories, files in os.walk( path ):
    for f in files:
        if f.find('_') > -1 or f.find('-') > -1:
            safe_rename( os.path.join(root, f), os.path.join( root, f.replace('_', ' ').replace('-', ' ') ) )

# <codecell>

# Remove files which were just artist names (oops)
for root, subdirectories, files in os.walk( path ):
    for f in files:
        if f[:4] == '.mid':
            os.remove( os.path.join(root, f) )

# <codecell>

# Replace . with space
for root, subdirectories, files in os.walk( path ):
    for f in files:
        title = os.path.splitext( f )[0]
        if title.find('.') > -1:
            safe_rename( os.path.join(root, f), os.path.join(root, title.replace('.', ' ') + '.mid') )

# <codecell>

# Change duplicate numbering with space to period (yesterday 7.mid -> yesterday.7.mid)
for root, subdirectories, files in os.walk( path ):
    for f in files:
        title = os.path.splitext(f)[0]
        while len(title) > 2 and title[-2] == " " and title[-1] in [str(n) for n in xrange(10)]:
            title = title[:-2]
        if title != os.path.splitext(f)[0]:
            safe_rename( os.path.join(root, f), os.path.join(root, title + '.mid') )

# <codecell>

# Flatten all directories
for root, subdirectories, files in os.walk( path ):
    for f in files:
        if len( os.path.join( root, f ).split('/') ) > 4:
            start = '/'.join( os.path.join(root, f).split('/')[:2] )
            end = '/'.join( os.path.join(root, f).split('/')[-2:] )
            new_path = os.path.join( start, end )
            safe_rename( os.path.join( root, f ), new_path )

# <codecell>

# Remove artist name from track title
for root, subdirectories, files in os.walk( path ):
    for f in files:
        artist = os.path.split(root)[1]
        if artist in f:
            safe_rename( os.path.join(root, f), os.path.join( root, f.replace(artist, '').lstrip() ) )
            f = f.replace(artist, '').lstrip()
        for word in artist.split(' '):
            if len(word) > 3 and word in f:
                safe_rename( os.path.join(root, f), os.path.join( root, f.replace(word, '').lstrip() ) )
                f = f.replace(word, '').lstrip()

# <codecell>

# Strip spaces at the beginning and end of filenames
for root, subdirectories, files in os.walk( path ):
    for f in files:
        new_f = os.path.splitext( f )[0].lstrip().rstrip() + os.path.splitext(f)[1]
        if new_f != f:
            safe_rename( os.path.join(root, f), os.path.join(root, new_f) )

# <codecell>

# Remove multiple spaces
for root, subdirectories, files in os.walk( path ):
    for f in files:
        new_f = f
        while new_f.find('  ') > -1:
            new_f = new_f.replace('  ',' ')
        if new_f != f:
            safe_rename( os.path.join(root, f), os.path.join(root, new_f) )

# <codecell>

def normalize_string(string):
    '''
    Make it lowercase and unicode
    '''
    return unicode(string.lower(), encoding='utf-8')

# <codecell>

# Make the md5->[[artist, title]] dict
md5_to_artist_title = {}
md5_to_paths = pickle.load( open('../data/Clean MIDIs-md5_to_paths.pickle') )
for root, subdirectories, files in os.walk(path):
    for f in files:
        if '.mid' not in f:
            continue
        md5 = hashlib.md5( open( os.path.join(root, f) ).read() )
        md5 = md5.hexdigest()
        title = split_all_extensions(f)
        artist = os.path.split(root)[1]
        title = normalize_string(title)
        artist = normalize_string(artist)
        md5_to_artist_title[md5] = [[artist, title]]
        for some_path in md5_to_paths[md5]:
            rem, title = os.path.split(some_path)
            title = os.path.splitext(title)[0]
            artist = os.path.split(rem)[1]
            if convert_camelCase(title) is not None:
                title = convert_camelCase(title)
            if convert_camelCase(artist) is not None:
                artist = convert_camelCase(artist)
            title = title.replace("_", " ").replace("-"," ")
            artist = artist.replace("_", " ").replace("-"," ")
            if len(title) > 2 and title[-2] == " " and title[-1] in [str(n) for n in xrange(10)]:
                title = title[:-2]
            if artist in title:
                title = title.replace(artist, "")
            for word in artist.split(' '):
                if len(word) > 3 and word in title:
                    title = title.replace(word, "")
            while title.find('  ') > -1:
                title = title.replace('  ',' ')
            while artist.find('  ') > -1:
                artist = artist.replace('  ',' ')
            artist = artist.lstrip().rstrip()
            title = title.lstrip().rstrip()
            title = normalize_string(title)
            artist = normalize_string(artist)
            if [artist, title] not in md5_to_artist_title[md5]:
                md5_to_artist_title[md5] += [[artist, title]]

# <codecell>

if __name__ == '__main__':
    import whoosh_search
    index = whoosh_search.get_whoosh_index('../data/cal500/index/')
    searcher = index.searcher()
    match_list = []
    for root, subdirectories, files in os.walk(path):
        for f in files:
            if '.mid' not in f.lower():
                break
            title = split_all_extensions(f)
            artist = os.path.split(root)[1]
            result = whoosh_search.search(searcher, index.schema, artist, title)
            if result is not None:
                match_list += [[os.path.join(artist, f), "{}-{}.mp3".format( result[1].replace(' ', '_'), result[2].replace(' ', '_') )]]
    searcher.close()
    pickle.dump( match_list, open('../data/Clean MIDIs-path_to_cal500_path.pickle', 'w') )

