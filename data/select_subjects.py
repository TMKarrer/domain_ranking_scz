'''
Selection of the subjects with both complete VBM and RS images.
'''

# packages
import os
import glob
import numpy as np
import pandas as pd

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'
dataname = 'SCZ_data.xlsx'
target = '/data/interim/'

# generate list of subjects with both complete vbm and rs data
beh = pd.read_excel(project_dir + '/data/raw/' + dataname)

vbm_path = []
rs_path = []
rs_subj = np.zeros(len(beh))
vbm_subj = np.zeros(len(beh))

for basedir, subname, i in zip(beh['folder'], beh['name'], range(len(beh))):
    print 'Subject number %i' % i
    if isinstance(subname, np.int):
        sub = '%07d' % subname
    else:
        sub = subname

    rs_path_mask = data_dir + '/data/raw/rs_data/' + basedir + os.sep + sub + '/RS/rsfix*'
    sub_rsfiles = sorted(glob.glob(rs_path_mask))

    if len(sub_rsfiles) == 0:
        print 'NOT FOUND'
        rs_path.append(sub_rsfiles)
    else:
        print 'Found %i RS images...' % len(sub_rsfiles)
        rs_subj[i] = 1
        rs_path.append(sub_rsfiles)

    vbm_path_mask = data_dir + '/data/raw/' + 'vbm_data/' + basedir + os.sep + sub + '/3D/sm0wrp1*'
    sub_vbmfile = sorted(glob.glob(vbm_path_mask))

    if len(sub_vbmfile) == 0:
        print 'NOT FOUND'
        vbm_path.append(sub_vbmfile)
    else:
        print 'Found'
        vbm_subj[i] = 1
        vbm_path.append(sub_vbmfile[0])

complete_data = np.logical_and(vbm_subj, rs_subj)

vbm_path = np.array(vbm_path)
vbm_path = vbm_path[complete_data]
vbm_path = [path.replace(data_dir, '~') for path in vbm_path]
data = beh[complete_data]
data['raw_vbm_paths'] = vbm_path

print 'Found %i subjects with complete VBM and RS data' % np.array(complete_data).sum()
data.to_excel(project_dir + target + 'complete_subjects.xlsx')
print 'Saved data and paths of complete subjects into %s' % target
