'''
Preprocessing of the resting state images.
'''

# packages
import os
import glob
import numpy as np
import pandas as pd
from scipy import io
import nibabel as nib
from nilearn.input_data import NiftiMasker

project_dir = 'scz_ranking_project'
data_dir = 'scz_data'

data = pd.read_excel(project_dir + '/data/interim/complete_subjects.xlsx')
tmpnii = nib.load(project_dir + '/data/raw/ref_rs_bin.nii.gz')
masker = NiftiMasker(tmpnii, standardize=True, smoothing_fwhm=12)
masker.fit()

# preprocessing
all_new_filenames = []
for i, sub_rs_files in enumerate(data.name):
    if isinstance(data['name'].iloc[i], np.int):
        sub = '%07d' % data['name'].iloc[i]
    else:
        sub = data['name'].iloc[i]
    print 'current subject is %s (%i of %i)' % (sub, i+1, len(data.name))
    number_rs_files = glob.glob(data_dir + '/data/raw/rs_data/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix*.nii.gz')
    rs_files = []
    for num in range(len(number_rs_files)):
        img_num = '%03d' % (num + 1)
        rs_files.append(data_dir + '/data/raw/rs_data/' + data.folder.iloc[i] + os.sep + sub + '/RS/rsfix_img' + img_num + '.nii.gz')

    confound_files = glob.glob(data_dir + '/data/raw/rs_data/' + data.folder.iloc[i] + os.sep + sub + '/RS/Counfounds*.mat')
    if len(confound_files) > 1:
        print 'too many confound files'
        stop
    else:
        confound_file = io.loadmat(confound_files[0])

    new_filenames = []

    for rs_file in rs_files:
        last_part = os.sep.join(rs_file.split(os.sep)[-5:])
        new_filenames.append(data_dir + '/data/interim/rs_data_conags/' + last_part)

    masked_data = masker.transform(rs_files, confounds=[confound_file['rp'],
                                   confound_file['rpp'], confound_file['rp']**2,
                                   confound_file['rpp']**2])

    for i in range(len(masked_data)):
        img = masker.inverse_transform(masked_data[i])

        if not os.path.exists(os.sep.join(new_filenames[i].split(os.sep)[:-1])):
                os.makedirs(os.sep.join(new_filenames[i].split(os.sep)[:-1]))
        img.to_filename(new_filenames[i])
