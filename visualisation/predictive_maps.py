'''
Visualization of the average predictive importance across all meta-priors.
'''

# packages
import os
import joblib
import numpy as np
import pandas as pd
import nilearn
import nilearn.plotting
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img

# settings
modalities = ['vbm', 'rs', 'vbm_rs']
taxonomy = ['BD', 'PC']
project_dir = os.path.expanduser('~/sciebo/Meta_Priors_teresa/')
tmpnii_bin = nib.load(project_dir + 'data/raw/ref_vbm_bin.nii.gz')
tmpnii = nib.load(project_dir + 'data/raw/colin_vbm_r.nii.gz')

for modality in modalities:
    print modality
    for prior_group in taxonomy:
        print prior_group

        meta_file_pathes = joblib.load(project_dir + '/data/interim/%s_Archiv_order.npy' % prior_group)
        meta_files = [path.replace('~', project_dir) for path in meta_file_pathes]
        meta_categories = [nifti_path.split(os.sep)[-1].split('_')[-1].split('.nii')[0] for nifti_path in meta_files]
        feat_imp = pd.read_csv(project_dir + '/models/LogReg_RF/' + modality + '/_meta_ranking/all_methods/' + prior_group + '_' + modality + '_all_methods_BT_data_mean.csv')
        feat_imp = pd.DataFrame(zip(feat_imp.means, feat_imp.priors), columns=['means', 'priors'])

        feat_imp_imgs = []
        for meta_file, meta_category in zip(meta_files, meta_categories):
            print meta_category
            prior = nib.load(meta_file)
            masker = NiftiMasker(tmpnii_bin)
            prior_data = masker.fit_transform(prior)
            prior_data[prior_data<0] = 0
            prior_data[prior_data>1] = prior_data[prior_data>1] * float(feat_imp[feat_imp.priors == meta_category].means)
            feat_imp_imgs.append(prior_data)

        result_data = np.mean(feat_imp_imgs, axis=0)
        result_img = masker.inverse_transform(result_data)

        nib.save(result_img, project_dir + 'visualisation/predictive_maps/%s_%s_predictive_map.nii.gz' % (prior_group, modality))
        result_img_smooth = smooth_img(result_img, fwhm=6)
        result_img_smooth.to_filename(project_dir + 'visualisation/predictive_maps/%s_%s_predictive_map_6mm.nii.gz' % (prior_group, modality))

        nilearn.plotting.plot_glass_brain(result_img_smooth,  #cut_coords=(21, 40, 45),
                                       display_mode='ortho', colorbar=True,
                                       black_bg=False,
                                       draw_cross=False,
                                       annotate=False,
                                       output_file=project_dir + 'visualisation/predictive_maps/%s_%s_predictive_map.png' % (prior_group, modality))
