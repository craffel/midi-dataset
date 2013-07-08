# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import whoosh_search
import csv
import sys
import query_freebase
import string

# <codecell>

#create whoosh index
#whoosh_search.createIndex( 'whoosh_index', 'unique_tracks.txt' )
index = whoosh_search.get_whoosh_index('whoosh_index')

# <codecell>

# Load the midi collection file into python list
# Three different lists: myId, myArtist, mySong

TOTAL_MIDI = 15559    #15559 midi files with a good format
myIds = []
myArtists = []
mySongs = []
with open('midiList.csv', 'rb') as md:
    reader = csv.reader(md)
    for rows in reader:
        blank, no, artist, song = rows
        myIds.append(no)
        myArtists.append(artist.lower())
        mySongs.append(song.lower())

# <codecell>

# Auto-correction of the midi name
count = 0
with open('match-id.csv', 'wb') as outfile:
    songwriter = csv.writer(outfile, delimiter=',',)
    for i in range(TOTAL_MIDI):
        query_info = myArtists[i] + ' ' + mySongs[i]
        result = whoosh_search.search(index,query_info)
        # Write the suggested result to the csv file 
        if result != None:
            msdID1 = result[0]
            msdID2 = result[1]
            msdSinger = result[2]
            msdSong = result[3]
            #songwriter.writerow([msdID1,msdID2,msdSinger, msdSong, mySongs[i],myArtists[i],myIds[i] ])
            # Some unicode error regarding the song name, so only the ids are written at this time
            songwriter.writerow([msdID1,msdID2, mySongs[i],myArtists[i],myIds[i] ])
            count += 1
            #print count
print count

#recognized
#5481 in total

# <codecell>

# Load the msd csv file into python list
# Four different lists: track_id, song_id, artists, songs
csv.field_size_limit(sys.maxsize)
TOTAL_ROWS = 999365
track_id = []
song_id = []
artists = []
songs = []

with open('msdList.csv', 'rb') as msd:
    reader = csv.reader(msd)
    for row in reader:
        id1, id2, singer, title = row
        track_id.append(id1)
        song_id.append(id2)
        artists.append(singer.lower())
        songs.append(title.lower())


# <codecell>

# load match-id, extract song and artist information
msd_track_id = []
msd_song_id = []
midiSongs = []
midiArtists = []
midiId = []

with open('match-id.csv','rb') as brn:
    reader = csv.reader(brn)
    for row in reader:
        id1,id2,mySong,myArtist,myId = row
        msd_track_id.append(id1)
        msd_song_id.append(id2)
        midiSongs.append(mySong)
        midiArtists.append(myArtist)
        midiId.append(myId)

# <codecell>

TOTAL_MATCH = 5481
with open('match-new-all','wb') as outfile:
    songwriter = csv.writer(outfile, delimiter=',',)
    songwriter.writerow(['MSD Track Id','MSD Song Id','MSDSong','MSD Artist','MIDISong','MIDIArtist','No'])
    for i in range(TOTAL_MATCH):
        idx = track_id.index(msd_track_id[i])
        print idx
        songwriter.writerow([track_id[idx],song_id[idx],artists[idx],songs[idx],midiSongs[i],midiArtists[i],midiId[i] ])

