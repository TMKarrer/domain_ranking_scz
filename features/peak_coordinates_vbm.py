'''
Meta-prior guided peak-coordinates of the VBM data.
'''

# packages
import glob
import os
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import nilearn
from nilearn.input_data import NiftiMasker

# reproducibility
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

# parameters
method = 'ha_pc'
taxonomy = ['BD', 'PC']
k_list = 5000
n_folds = 10
n_jobs = 12
modality = 'vbm'

# peak coordinate methods
if method == 'ha_pc':
    title = 'highest-activation_voxels'
    mapname = '_maps_ha_pc'
elif method == 'md_pc':
    title = 'max-distance-to-mean_voxels'
    mapname = '_maps_md_pc'
elif method == 'ts_pc':
    title = 'top-standardized_voxels'
    mapname = '_maps_ts_pc'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
vbm_path = np.array(data['vbm_paths'])
vbm_path = [path.replace('~', data_dir) for path in vbm_path]
diags = np.array(data[99])
gm_mask = nib.load(project_dir + '/data/raw/icbm_fsl.nii')

# function using prior for classification
def prior_classifier(meta_file, meta_category):
    np.random.seed(42)  # reproducible parallelisation
    niimask = nib.load(project_dir + '/data/processed/' + prior_group + mapname + os.sep + prior_group + '_' + str(k) + os.sep + prior_group + '_' + meta_category + '_' + str(k) + '_' + title + '.nii.gz')
    masker = NiftiMasker(niimask)
    masker.fit()

    voxels_in_metaprior = (niimask.get_data() > 0).sum()
    print('%s: %i non-zero voxels' % (meta_category, voxels_in_metaprior))

    FS_stack = None
    batch_size = 50.
    n_batches = np.ceil(len(vbm_path) / batch_size)
    for i_batch in range(int(n_batches)):
        i_start = i_batch * batch_size
        i_end = min(i_start + batch_size, len(vbm_path))

        FS_batch = masker.transform(vbm_path[int(i_start):int(i_end)])
        if FS_stack is None:
            print 'starting a batch'
            FS_stack = FS_batch
        else:
            FS_stack = np.vstack((FS_stack, FS_batch))

    from sklearn.preprocessing import StandardScaler
    FS_stack = StandardScaler().fit_transform(FS_stack)
    FS_prior = FS_stack
    print FS_prior.shape
    return (meta_category, FS_prior)

for prior_group in taxonomy:
    print prior_group

    for k in str(k_list).split(','):
        print str(k)
        k = int(k)

        target_dir = project_dir + '/features/%s/%s/' % (modality, method)
        if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        meta_file_pathes = joblib.load(project_dir + '/data/interim/%s_Archiv_order.npy' % prior_group)
        meta_files = [path.replace('~', project_dir) for path in meta_file_pathes]
        meta_categories = [nifti_path.split(os.sep)[-1].split('_')[-1].split('.nii')[0] for nifti_path in meta_files]

        # calculate classifier for each meta prior
        result = Parallel(n_jobs=n_jobs)(delayed(prior_classifier)(meta_file, meta_category) for meta_file, meta_category in zip(meta_files, meta_categories))
        listed_meta_categories, FS_prior = zip(*result)

        assert (list(listed_meta_categories) == meta_categories)
        assert (np.array(FS_prior).shape) == (len(meta_files), len(data), k)
        joblib.dump(result, target_dir + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy')
        print result
        print 'Saved feature space.'
