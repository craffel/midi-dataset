## MIDI Dataset

The goal of this project is to match and align a very large collection of MIDI files to very large collections of audio files so that the MIDI data can be used to infer ground truth information about the audio.

### Datasets

Each dataset is contained in a folder in `data`.  Each dataset has the following:

- A .js file describing the contents.  This json file is essentially a list of dicts, where each dict describes one song in the dataset, including its metadata and path.  See below for information about how these were created.
- A whoosh index.  This allows the dataset to be searched in a (barely) fuzzy manner.  See below for information about how these were created.
- Data.  The data is organized by filetype, e.g. `data/cal500/mp3` contains all of the .mp3 files for cal500.  Each dataset has at least one of the following: mp3, mid, h5 (MSD metadata/features), npz (e.g. cached features computed with `librosa` or alignment results).

The datasets are as follows:

MIDI:

1. Full MIDI dataset, in `data/midi`.  This is the full collection of MIDI files scraped off the internet.  Because we're assuming we don't know anything about the metadata of these files, there's no .js file or whoosh index (yet?).
1. Clean MIDI dataset, in `data/clean_midi`.  This is a subset of the full MIDI dataset which has reasonable artist/title file naming, i.e. the song "Some Song" by the artist "An Artist" should be in `data/clean_midi/An Artist/Some Song.mid`.  For information about how this dataset was collected, see below (under "process").  Entries in the .js file include artist, title, md5 of the MIDI file, and file base path.
1. Clean MIDI dataset aligned to baseline mp3 collections (see below), in `data/aligned_midi`.  This data is created as part of this experiment and includes both aligned MIDI files and .npz files which contain the results of the alignment.  Entries in the .js file include the ID (list index) of the entry for the MIDI file in the clean_midi js file, the baseline mp3 collection from which the mp3 file came (one of `cal500`, `cal10k`, or `uspop2002`), the ID (list index) of the entry for the mp3 file in the js file from the baseline mp3 collection, and the aligned file base path.

Baseline mp3 collections matched to Echonest/MSD analysis (see [here](http://labrosa.ee.columbia.edu/millionsong/pages/additional-datasets))

1. cal500, in `data/cal500/`.  This is the cal500 dataset, meant for music autotagging.  Entries in the .js file include artist, title, and file base path.
1. cal10k, in `data/cal10k/`.  The cal10k dataset, also for autotagging.  Entries in the .js file include artist, title, matched echonest ID (from `EchoNestTrackIDs.tab`, found by querying Echonest), and file base path.
1. uspop2002, in `data/uspop2002`.  The uspop2002 dataset, for artist recognition/similarity estimation.  Entries in the .js file include artist, title, album, and file base path.

The MSD:

1. The Million Song Dataset, in `data/msd`.  Includes metadata, Echonest features, and 7digital preview mp3s for 1,000,000 songs.  Entries in the .json file include artist, title, song ID, track ID, and base file path.


### Process

1. About 500,000 MIDI files were scraped from the internet.  About half of them are duplicates.  Metadata (specifically, song title and artist) is sometimes included in MIDI files as text meta events, but more frequently the filename/subdirectory is used as a way of denoting the artist/song title.
1. We manually found about 25,000 non-duplicate MIDI files which had clean metadata information in their filename/subdirectory.  We did some manual cleaning/merging to make their "metadata" as sane as possible.
1. For each file in the "clean MIDI subset", we resolved the purported artist/title name against Freebase and the Echonest.  Code for this step is in `normalize_names.py`.  This resulted in about 17,000 MIDI files for about 9,000 unique tracks.
1. Using the file lists (typically supplied with the datasets), we created .js files describing the contents of each dataset.  This is primarily to keep track of metadata for each file in the dataset in a sane/consistent way.  Code for this step is in `json_utils.py`.
1. For each of the datasets, we created Whoosh indices.  These are used to match an entry in one dataset (usually the clean MIDI dataset) to an entry in another (one of cal500, cal10k, uspop2002, or MSD).  Code for this step is in `whoosh_search.py`.
1. Using Whoosh, we matched each entry in the clean MIDI subset to mp3 files from cal500/cal10k/uspop2002.  For the files that matched, we attempted to align the MIDI file to the matching audio file (using [align_midi](https://github.com/craffel/align_midi)).  Code for this step is in `midi_alignment.py`.  This resulted in about 6,000 MIDI-MP3 matches, about 2,500 of which correspond to unique songs.
1. Once we ran the alignment, we manually verified whether the alignment was successful or not.  The resulting valid/invalid labels are collected in xyz.tsv (TODO - verify alignment output)
1. For those audio/MIDI pairs which had good alignments, we extracted the Echonest analysis features from the accompanying .h5 file and a piano-roll representation from the .mid file, thereby creating a dataset of aligned MIDI and Echonest/MSD features.  Code for this step is in `create_hashing_dataset.py` (TODO-re-write with the new dataset format)
1. Using this cross-modality hashing dataset, we trained a siamenese neural net architecture to map aligned feature vectors from each modality (Echonest/MSD features and MIDI piano roll) to a common binary space (see e.g. "Multimodal similarity-preserving hashing", Masci et al.).  Code for this step is in `cross_modality_hashing.py` (TODO: re-write using nntools, different architectures)
1. Using the trained neural network, we mapped MIDI files with no good metadata to entries in the MSD. (TODO)
1. Once we obtained matched MIDI file <-> MSD entry pairs, we aligned the corresponding 7digital/wcd audio to the MIDI file.  From here, we can obtain (approximate) ground-truth information (e.g. chords, beats, transcription, key, tempo, etc) for the corresponding MIDI files. (TODO)
