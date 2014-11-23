'''
The h5 files distrbuted with the MSD to accompany cal10k and cal500 are named
according to their entry in the MSD, not in conjunction with the cal10k/cal500
naming scheme.  This fixes that.
'''

import os
os.chdir('..')
import sys
sys.path.append(os.getcwd())
import read_sv
import shutil

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
