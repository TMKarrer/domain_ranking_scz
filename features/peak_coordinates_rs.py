'''
Meta-prior guided peak-coordinates of the RS data.
'''

# packages
import glob
import os
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn.input_data import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
import joblib
from joblib import Parallel, delayed

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
modality = 'rs'

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
diags = np.array(data[99])
tmpnii = nib.load(project_dir + '/data/raw/ref_rs_bin.nii.gz')

# function using prior for classification
def prior_classifier(meta_file, meta_category):
    np.random.seed(42)  # reprodicible parallelisation
    print 'Current Prior is ' + prior_group + ' ' + str(k) + ' ' + meta_category
    raw_niimask = nib.load(project_dir + '/data/processed/' + prior_group + mapname + os.sep + prior_group + '_' + str(k) + os.sep + prior_group + '_' + meta_category + '_' + str(k) + '_' + title + '_labels.nii.gz')
    niimask = nilearn.image.resample_img(raw_niimask, target_affine=tmpnii.get_affine(),
                                                 target_shape=tmpnii.shape, interpolation='nearest')
    masker = NiftiMasker(tmpnii)
    masker.fit()
    niimask_labels = np.squeeze(masker.transform(niimask))

    FS_single_prior = []

    for i, sub_rs_files in enumerate(data.name):
        if isinstance(data['name'].iloc[i], np.int):
            sub = '%07d' % data['name'].iloc[i]
        else:
            sub = data['name'].iloc[i]
        print 'subject ' + str(i+1) + ' of ' + str(len(data.name)) + ' subjects'
        number_rs_files = glob.glob(data_dir + '/data/interim/rs_data_conags/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix*.nii.gz')
        rs_files = []
        for num in range(len(number_rs_files)):
            img_num = '%03d' % (num + 1)
            rs_files.append(data_dir + '/data/interim/rs_data_conags/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix_img' +img_num + '.nii.gz')

        subj_labels = []
        for i, rs_file in enumerate(rs_files):
            print 'rsfile ' + str(i+1) + ' of ' + str(len(rs_files)) + ' rsfiles'
            rs_array = np.squeeze(masker.transform(rs_file))
            subj_values = []
            for label in np.unique(niimask_labels):
                if label == 0:
                    continue
                label = float(label)
                clvalues = rs_array[niimask_labels == label]
                feature = np.sum(clvalues).astype(np.float64)
                subj_values.append(feature / len(clvalues))
            subj_labels.append(subj_values)
        subj_labels = np.array(subj_labels)
        assert (subj_labels.shape == (len(rs_files), k/250))

        connectivity_measure1 = ConnectivityMeasure(kind='correlation')
        cross_corr = connectivity_measure1.fit_transform([np.array(subj_labels)])[0]
        tril_inds = np.tril_indices_from(cross_corr, k=-1)
        cc_ravel = cross_corr[tril_inds]
        assert (len(cc_ravel) == ((k/250*k/250-k/250)/2))

        FS_single_prior.append(cc_ravel)

    assert (len(FS_single_prior) == len(data))

    print 'Deconfounding Feature Space for Age, Gender, & Site'  # at earliest point of time
    deconf_FS_single_prior = nilearn.signal.clean(np.array(FS_single_prior),
                                                  confounds=[pd.get_dummies(data['Site']).as_matrix(),
                                                  pd.get_dummies(data['Gender']).as_matrix(), np.array(data['Age'])], standardize=False)

    FS_prior = deconf_FS_single_prior

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
        assert (np.array(FS_prior).shape == (len(meta_files), len(data), (k/250*k/250-k/250)/2))
        joblib.dump(zip(listed_meta_categories, FS_prior), target_dir + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy')

        print zip(listed_meta_categories, FS_prior)
        print 'Saved feature space.'
