'''
The h5 files distrbuted with the MSD to accompany cal10k and cal500 are named
according to their entry in the MSD, not in conjunction with the cal10k/cal500
naming scheme.  uspop2002 is distributed with h5 files with the excention
.mp3.h5.  This fixes these issues.
'''

import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import read_sv
import shutil
import glob

data = read_sv.get_sv_list('file_lists/cal500_EchoNest_IDs.txt')
for row in data:
    shutil.move('data/cal500/h5/{}.h5'.format(row[1]),
                'data/cal500/h5/{}.h5'.format(row[0]))

data = read_sv.get_sv_list('file_lists/EchoNestTrackIDs.tab')
for row in data:
    if row[4][:5] == 'music':
        msd_id = row[4].split('/')[-1]
        shutil.move('data/cal10k/h5/{}.h5'.format(msd_id),
                    'data/cal10k/h5/{}.h5'.format(row[3]))

for filename in glob.glob('data/uspop2002/h5/*/*/*.mp3.h5'):
    shutil.move(filename, filename.replace('.mp3', ''))
