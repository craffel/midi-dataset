File Lists
==========

This directory contains text files listing all of the files in the various datasets used in this project.  Included are
  * `cal500.txt` - ID, artist, and title for each entry in cal500.
  * `cal10k.txt` - ID, artist, title, and EchoNest ID/md5 (ignored) for each entry in cal10k.
  * `clean_midi.txt` - ID, artist, title, md5, and path for each entry in the "clean MIDI subset".
  * `uspop2002.txt` - ID, artist, title, and path for each *non-live* entry in uspop2002.  Songs were determined to be live or not based on their Echo Nest "loudness" score; please see [this script](https://gist.github.com/craffel/125f15a8c4f33d6c70d5).

Not included:
  * `msd.txt` - Track ID, song ID, artist, and title for each entry in the Million Song Dataset.  This file is distributed with the MSD as `unique_tracks.txt`.  You can also download it [here](http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/unique_tracks.txt) and rename `msd.txt`.

