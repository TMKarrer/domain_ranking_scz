'''
Meta-prior guided matrix decomposition of the VBM data.
'''

# packages
import os
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
from nilearn.image import load_img, new_img_like
from nilearn.input_data import NiftiMasker
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, SparsePCA
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
modality = 'vbm'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
vbm_path = np.array(data['vbm_paths'])
vbm_path = [path.replace('~', data_dir) for path in vbm_path]
tmpnii = nib.load(project_dir + '/data/raw/ref_vbm_bin.nii.gz')
diags = np.array(data[99])

# function using prior for matrix decomposition
def prior_classifier(meta_file, meta_category):
    np.random.seed(42)  # reprodicible parallelisation
    print '-' * 80
    print 'Current meta prior is %s' % meta_category
    img = load_img(meta_file)
    img_data = img.get_data()
    img_data[img.get_data() < 0] = 0
    img_data[img.get_data() > 0] = 1
    prior_mask = new_img_like(img, img_data, affine=img.get_affine())

    print 'Created NiftiMasker with %i voxels for %s' % ((prior_mask.get_data() > 0).sum(), meta_category)
    masker = NiftiMasker(prior_mask)
    masker.fit()
    masked_data = masker.transform(vbm_path)
    z_vbm_data_array = StandardScaler().fit_transform(masked_data)

    # matrix decomposition method
    if method == 'pca':
        matdecomp = PCA(n_components=k, random_state=42)
    elif method == 'ica':
        matdecomp = FastICA(n_components=k, random_state=42)
    elif method == 'sparse_pca':
        matdecomp = SparsePCA(n_components=k, tol=0.1, random_state=42)

    matdecomp_loadings = matdecomp.fit_transform(z_vbm_data_array)

    # visualisation of components
    matdecomp_weights = matdecomp.components_
    nifti_4d_components = masker.inverse_transform(matdecomp_weights)

    for ii in range(0, k, 9):
        print 'visualisation of component %i of %i' % (ii+1, k)
        cur_nifti = index_img(nifti_4d_components, ii)
        fname = '%s_%s_%i_comp_%i' % (meta_category, method, k, ii+1)
        vis_dir1 = data_dir + '/data/processed/%s_%s_%s_%i/%s' % (modality, method, prior_group, k, meta_category)
        if not os.path.exists(vis_dir1):
            os.makedirs(vis_dir1)
        plot_stat_map(cur_nifti, cut_coords=(0, 0, 0),
                      display_mode='ortho', colorbar=True,
                      black_bg=True, draw_cross=True,
                      bg_img=project_dir + '/data/raw/colin.nii', output_file='%s/%s.png' % (vis_dir1, fname))
        cur_nifti.to_filename('%s/%s.nii.gz' %(vis_dir1, fname))

    FS_prior = matdecomp_loadings
    return (meta_category, FS_prior)

for prior_group in taxonomy:
    print prior_group

    for k in str(k_list).split(','):
        k = int(k)
        print str(k) + ' components'

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
