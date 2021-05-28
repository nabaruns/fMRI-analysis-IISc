import os
import argparse
import shutil
import subprocess
import json
import warnings
import sys, glob

from multiprocessing import Pool

import pandas as pd
import numpy as np
from os.path import abspath, join, pardir

from nilearn import datasets
# from roi_mean_sub import roi_mean_interface
# from bids.layout import BIDSLayout
from nipype.pipeline import Node, MapNode, Workflow
from nipype.interfaces.io import DataSink, DataGrabber
from nipype.algorithms.confounds import TSNR
from nipype.interfaces.utility import Function, IdentityInterface
from nilearn.input_data import NiftiLabelsMasker
from nipype.interfaces import fsl
from nipype.interfaces.utility import Rename
from nilearn import image
from nilearn.connectome import ConnectivityMeasure
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
import nibabel as nib


def _run_copy(source, output_path):
    cmd_str = "cp -r {} {}".format(source, output_path)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _run_untar(outdir, srcfile):
    cmd_str = "tar -C {} -zxf {}".format(outdir, srcfile)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _rm_tar(srcfile):
    cmd_str = "rm -rf {}".format(srcfile)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _scp_fmriprep(srcfile, base_dir):
    cmd_str = "scp -q -r nabarun@10.36.17.186:~/RADC/sym_radc/fmriprep/{} {}/RADC/fmriprep".format(srcfile, base_dir)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def collect_data(layout, participant_label, bids_validate=True):
    queries = {
        "func": {"datatype": "func", "suffix": "bold"},
        "confounds": {"datatype": "func", "suffix": ['timeseries','regressors']},
        "flair": {"datatype": "anat", "suffix": "FLAIR"},
        "t2w": {"datatype": "anat", "suffix": "T2w"},
        "t1w": {"datatype": "anat", "suffix": "T1w"},
        "roi": {"datatype": "anat", "suffix": "roi"},
    }

    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                extension=["nii", "nii.gz", "tsv"],
                **query
            )
        )
        for dtype, query in queries.items()
    }

    return subj_data

def extract_confounds(confound_file, confound_vars):
    confound_df = pd.read_csv(confound_file, delimiter='\t')
    confound_df = confound_df[confound_vars]
    for col in confound_df.columns:

        #Example X --> X_dt
        new_name = '{}_dt'.format(col)

        #Compute differences for each pair of rows from start to end.
        new_col = confound_df[col].diff()

        #Make new column in our dataframe
        confound_df[new_name] = new_col
    return confound_df.values
  
# pool = Pool()

# for subject_id in df[0]:
def f(idir, odir, subject_id):
    print("Target: {}\n".format(subject_id))

    # layout = BIDSLayout(idir,validate=False)
    # subjects = layout.get_subjects()
    
    pooled_subjects = []
    tr_drop = 4

    # for sub in subjects:
    #Get functional file and confounds file
    # subj_data = collect_data(layout, sub, False)
    # func_file = subj_data['func'][0]

    func_file = glob.glob(os.path.join(idir,'*bold.nii.gz'))
    confound_file = glob.glob(os.path.join(idir,'*.tsv'))
    # confound_file = subj_data['confounds'][0]
    print(func_file,confound_file)

    #Load functional file and perform TR drop
    func_img = image.load_img(func_file[0])
    func_img = func_img.slicer[:,:,:,tr_drop+1:]

    #Convert cnfounds file into required format
    confound_vars = ['trans_x', 'trans_y', 'trans_z', 
                    'rot_x', 'rot_y', 'rot_z',
                    'global_signal','a_comp_cor_01','a_comp_cor_02']
    confounds = extract_confounds(confound_file[0], confound_vars)

    #Drop TR on confound matrix
    confounds = confounds[tr_drop+1:,:]
    #Apply cleaning, parcellation and extraction to functional data
    time_series = masker.fit_transform(func_img,confounds)
    pooled_subjects.append(time_series)
    print("Timeseries saved with shape",np.shape(time_series))
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform(pooled_subjects)
    np.save(odir, correlation_matrix)
    print("Corr matrix saved with shape",np.shape(correlation_matrix))

    print("\n")

subject_id = sys.argv[1]
in_dir = os.path.join(sys.argv[2],subject_id)
out_dir = os.path.join(sys.argv[3],subject_id)

# df = pd.read_csv(base_dir+'/fmriprep_list', header=None)

########## schaefer 400/1000 ROI #########
# dataset = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, 
#                             resolution_mm=1, verbose=0)
# atlas_filename = dataset.maps
# labels = dataset.labels
# masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
#                            memory='nilearn_cache', verbose=0)


########## Willard 499 ROI #########
parcel_file = '/home/nabaruns/willard_fROIs_atlas.nii.gz'
atlas = image.load_img(parcel_file)
masker = NiftiMapsMasker(
    atlas, resampling_target="data", t_r=2.5, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)

f(in_dir, out_dir, subject_id)
# p = Pool()
# p.map(f, df[0])