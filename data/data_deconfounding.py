'''
Deconfounding and resampling of the imaging data.
'''

# packages
import numpy as np
import pandas as pd
import os
import nibabel as nib
import nilearn
from nilearn.input_data import NiftiMasker

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
vbm_path = np.array(data['raw_vbm_paths'])
vbm_path = [path.replace('~', data_dir) for path in vbm_path]
tmpnii = nib.load(project_dir + '/data/raw/ref_vbm_bin.nii.gz')
all_new_filenames = []

# deconfounding
print 'Deconfounding vbm data for age, gender and site'
masker = NiftiMasker(tmpnii)
masker.fit()
masked_data = masker.transform(vbm_path, confounds=[pd.get_dummies(data['Site']).as_matrix(),
                               pd.get_dummies(data['Gender']).as_matrix(), np.array(data['Age'])])

# resampling
for i in range(len(masked_data)):
    filepath = data_dir + '/data/interim/vbm_data_conags' + os.sep + '/'.join(vbm_path[i].split(os.sep)[-5:-1])
    filename = data_dir + '/data/interim/vbm_data_conags' + os.sep + '/'.join(vbm_path[i].split(os.sep)[-5:])
    all_new_filenames.append(filename)
    img = masker.inverse_transform(masked_data[i])
    img = nilearn.image.resample_img(nib.Nifti1Image(np.array(img.get_data()), img.get_affine()),
                                     target_affine=tmpnii.get_affine(), target_shape=tmpnii.shape,
                                     interpolation='nearest')
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    img.to_filename(filepath + os.sep + vbm_path[i].split('/')[-1])
    print 'printed subject %i of %i to data/interim' % (i+1, len(vbm_path))

vbm_path = [path.replace('/scz_ranking_project', '~') for path in all_new_filenames]
data['vbm_paths'] = vbm_path
data.to_excel(data_dir + '/data/interim/complete_subjects.xlsx')
