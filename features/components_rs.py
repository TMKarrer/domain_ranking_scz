'''
Meta-prior guided matrix decomposition of the RS data.
'''

# packages
import os
import glob
import time
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn.image import load_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map
from sklearn.decomposition import PCA, FastICA, SparsePCA
from nilearn.connectome import ConnectivityMeasure
import joblib
from joblib import Parallel, delayed

# reproducibility
np.random.seed(42)

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

# parameters
method = 'pca'
taxonomy = ['BD', 'PC']
k_list = 100
n_folds = 10
n_jobs = 12
modality = 'rs'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
diags = np.array(data[99])
tmpnii = nib.load(project_dir + '/data/raw/ref_rs_bin.nii.gz')

# function using prior for matrix decomposition
def prior_classifier(meta_file, meta_category):
    np.random.seed(42)  # reproducible parallelisation
    print 'Current Prior is ' + prior_group + ' ' + str(k) + ' ' + meta_category
    raw_img = load_img(meta_file)
    img = nilearn.image.resample_img(raw_img, target_affine=tmpnii.get_affine(),
                                     target_shape=tmpnii.shape, interpolation='nearest')

    img_data = img.get_data()
    img_data[img.get_data() < 0] = 0
    img_data[img.get_data() > 0] = 1
    prior_mask = new_img_like(img, img_data, affine=img.get_affine())

    subsampled_group_rsfiles = []
    FS_single_prior = []

    seed = np.random.RandomState(42)  # reproducible group components
    for i, sub_rs_files in enumerate(data.name):
        if isinstance(data['name'].iloc[i], np.int):
            sub = '%07d' % data['name'].iloc[i]
        else:
            sub = data['name'].iloc[i]

        number_rs_files = glob.glob(data_dir + '/data/interim/rs_data_conags/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix*.nii.gz')
        rs_files = []
        for num in range(len(number_rs_files)):
            img_num = '%03d' % (num + 1)
            rs_files.append(data_dir + '/data/interim/rs_data_conags/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix_img' + img_num + '.nii.gz')

        inds_subsample = np.arange(len(rs_files))
        seed.shuffle(inds_subsample)
        subsample = inds_subsample[:10]
        for sample in subsample:
            subsampled_group_rsfiles.append(rs_files[sample])
    assert (len(subsampled_group_rsfiles) == len(data)*10)

    print 'Masking subsampled group rsfiles'
    t0 = time.clock()
    masker = NiftiMasker(prior_mask)
    masker.fit()
    FS_stack = None
    batch_size = 50.
    n_batches = np.ceil(len(subsampled_group_rsfiles) / batch_size)
    for i_batch in range(int(n_batches)):
        i_start = i_batch * batch_size
        i_end = min(i_start + batch_size, len(subsampled_group_rsfiles))
        FS_batch = masker.transform(subsampled_group_rsfiles[int(i_start):int(i_end)])
        if FS_stack is None:
            print 'starting a batch'
            FS_stack = FS_batch
        else:
            FS_stack = np.vstack((FS_stack, FS_batch))
    group_files = FS_stack
    t1 = (time.clock() - t0)/60
    print 'Time for masking subsampled group rs files: ' + str(t1)

    # matrix decomposition method
    if method == 'pca':
        matdecomp = PCA(n_components=k, random_state=42)
    elif method == 'ica':
        matdecomp = FastICA(n_components=k, random_state=42)
    elif method == 'sparse_pca':
        matdecomp = SparsePCA(n_components=k, tol=0.1, random_state=42)

    print 'Calculating group components'
    t0 = time.clock()
    matdecomp.fit(group_files)

    group_matdecomp_weights = matdecomp.components_
    group_img = masker.inverse_transform(group_matdecomp_weights)
    t1 = (time.clock() - t0)/60
    print 'Time for calculating group components: ' + str(t1)

    # visualisation of components
    for i in range(0, k, 9):
        print 'visualisation of group component ' + str(i+1)
        cur_nifti = index_img(group_img, i)
        fname = '%s_%s_%i_groupcomp_%i' % (meta_category, method, k, i+1)
        vis_dir1 = data_dir + '/data/processed/%s_%s_%s_%i/%s' % (modality, method, prior_group, k, meta_category)
        if not os.path.exists(vis_dir1):
            os.makedirs(vis_dir1)
        plot_stat_map(cur_nifti,
                      display_mode='ortho', colorbar=True,
                      black_bg=True, draw_cross=True,
                      bg_img=project_dir + '/data/raw/colin.nii',
                      output_file='%s/%s.png' % (vis_dir1, fname))
        cur_nifti.to_filename('%s/%s.nii.gz' % (vis_dir1, fname))

    for i, sub_rs_files in enumerate(data.name):
        print sub_rs_files
        if isinstance(data['name'].iloc[i], np.int):
            sub = '%07d' % data['name'].iloc[i]
        else:
            sub = data['name'].iloc[i]
        print 'current subject is ' + sub
        number_rs_files = glob.glob(data_dir + '/data/interim/rs_data_conags/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix*.nii.gz')
        rs_files = []
        for num in range(len(number_rs_files)):
            img_num = '%03d' % (num + 1)
            rs_files.append(data_dir + '/data/interim/rs_data_conags/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix_img' + img_num + '.nii.gz')

        print 'Masking single subject rsfiles'
        masker = NiftiMasker(prior_mask)
        masker.fit()
        masked_files = masker.transform(rs_files)

        print 'Generating component values for one subject'
        new_loadings = matdecomp.transform(masked_files)
        assert (new_loadings.shape == (len(rs_files), k))

        print 'Generating lower triangle of cross-correlations of component values'
        connectivity_measure1 = ConnectivityMeasure(kind='correlation')
        cross_corr = connectivity_measure1.fit_transform([new_loadings])[0]
        tril_inds = np.tril_indices_from(cross_corr, k=-1)
        cc_ravel = cross_corr[tril_inds]
        assert (len(cc_ravel) == ((k*k-k)/2))

        FS_single_prior.append(cc_ravel)

    assert (len(FS_single_prior) == len(data))

    print 'Deconfounding Feature Space for Age, Gender, & Site'  # at earliest point of time
    deconf_FS_single_prior = nilearn.signal.clean(np.array(FS_single_prior),
                                                  confounds=[pd.get_dummies(data['Site']).as_matrix(),
                                                  pd.get_dummies(data['Gender']).as_matrix(), np.array(data['Age'])], standardize=False)

    FS_prior = deconf_FS_single_prior
    return (meta_category, FS_prior)

for prior_group in taxonomy:

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
        assert (np.array(FS_prior).shape == (len(meta_files), len(data), (k*k-k)/2))

        joblib.dump(zip(listed_meta_categories, FS_prior), target_dir + prior_group + '_' + method + '_feature_space_' + str(k) + '.npy')

        print zip(listed_meta_categories, FS_prior)
        print 'Saved feature space.'
