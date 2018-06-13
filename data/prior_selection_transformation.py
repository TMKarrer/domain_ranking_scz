'''
Selection and transformation of the meta-priors based on =< 50 experiments.
'''

# packages
import numpy as np
import pandas as pd
import glob
import os
import sys
import nibabel as nib
import nilearn
from nilearn.image import load_img, new_img_like
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing

from IPython import get_ipython
ipython = get_ipython()

src_dir = os.path.join(os.getcwd(), 'src/_utility')
sys.path.append(src_dir)

if '_IPYTHON_' in globals():
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 1')
    ipython.magic('aimport nilearn_dev')

project_dir = 'scz_ranking_project'
taxonomy = ['BD', 'PC']

tmpnii = nib.load(project_dir + '/data/raw/ref_vbm_bin.nii.gz')

for prior_group in taxonomy:
    print prior_group
    prior_list = sorted(glob.glob(project_dir + '/data/raw/%s_Archiv_images_numbers/*.png' % prior_group))

    number = []
    for i in range(len(prior_list)):
        nr = prior_list[i][prior_list[i].find('(')+1:prior_list[i].find('exp)')]
        number.append(int(nr.split('(')[-1]))
        prior_list[i] = prior_list[i].split('_ppc')[0].split(os.sep)[-1]

    raw = pd.DataFrame([prior_list, number])
    raw = raw.transpose()
    raw[1] = np.array(raw[1])
    raw.columns = ['MetaPrior', '# exp']
    raw = raw.sort('# exp', ascending=False)

    path = project_dir + '/data/raw/%s_Archiv_alescores' % prior_group
    nii_list = sorted(glob.glob(project_dir + path + os.sep + '*.nii'))

    for i in range(len(nii_list)):
        img = nib.load(nii_list[i])
        img.to_filename(nii_list[i] + '.gz')
    for i in range(len(nii_list)):
        os.remove(nii_list[i])

    # delete priors with less than 50 experiments
    delete = raw[raw['# exp'] < 50]
    for i in range(len(delete)):
        print 'Will delete %s prior since too few experiments' % delete.iloc[i]
        if os.path.exists(project_dir + '/data/raw/' + prior_group + '_Archiv_alescores' + os.sep + delete['MetaPrior'].iloc[i] + '.nii.gz'):
            os.remove(project_dir + '/data/raw/' + prior_group + '_Archiv_alescores' + os.sep + delete['MetaPrior'].iloc[i] + '.nii.gz')

    nii_list = sorted(glob.glob(path + os.sep + '*.nii.gz'))
    for i in range(len(nii_list)):
        title = nii_list[i].split(os.sep)[-1]
        print '%i from %i: %s' % (i, len(nii_list), title)
        img = load_img(nii_list[i])
        img_data = _safe_get_data(img, ensure_finite=True)
        clean_img = new_img_like(img, img_data, affine=img.get_affine())

        vbm_masker = NiftiMasker(tmpnii)
        clean_data = vbm_masker.fit_transform(clean_img)
        clean_masked_img = vbm_masker.inverse_transform(clean_data)

        res_img = nilearn.image.resample_img(clean_masked_img,
                                             target_affine=tmpnii.get_affine(),
                                             target_shape=tmpnii.shape,
                                             interpolation='nearest')
        res_data = vbm_masker.transform(res_img)
        t_data = preprocessing.StandardScaler().fit_transform(np.squeeze(res_data))
        t_img = vbm_masker.inverse_transform(t_data)

        filepath = project_dir + '/data/interim/%s_Archiv_zvalues_new' % prior_group
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        t_img.to_filename(filepath + os.sep + title)
