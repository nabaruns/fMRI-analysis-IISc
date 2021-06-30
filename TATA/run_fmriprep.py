import os
import argparse
import shutil
import subprocess
import json
import warnings
from pathlib import Path
# import natsort

def _cli_parser():
    """Reads command line arguments and returns input specifications"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, metavar='dicom_dir',
                        help="The path to the directory which contains all "
                             "subjects' dicom images.")
    parser.add_argument('-o', type=str, metavar='output_dir',
                        help="The path to the directory which contains all "
                             "subjects' BIDS data.")
    parser.add_argument('--ignore', nargs='+', type=str, metavar='ignored_dirs',
                        help="Subdirectories in `-d` to ignore.")
    parser.add_argument('-s', type=str, metavar='session',
                        help="Session number, e.g., 'ses-01'")
    parser.add_argument('-c', type=str, metavar='config',
                        help='Configuration .json file for dcm2bids. Refer to '
                             'dcm2bids documentation for examples.')
    parser.add_argument('--force-run-labels', action='store_true', 
                        help='Force all functional runs to have a run number. '
                             'This means that singleton runs, i.e. tasks that '
                             'have only one functional run will be labeled '
                             'as `run-01`. This is a necessary workaround for '
                             'fmriprep 1.4.0 or greater. Otherwise, singleton '
                             'runs will not have a run number/label, which is '
                             'the default for dcm2bids.')
    parser.add_argument('-m', type=str.lower, metavar='mapping',
                        help='.json file containing specific mappings between '
                             'input dicom folders (keys) and subject IDs (values). '
                             'Useful for multi-session data in which different '
                             'dicom folders belong to the same subject.')
    return parser.parse_args()

def _rm_tar(srcfile):
    cmd_str = "rm -rf {}".format(srcfile)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _run_dcm2bids_scaffold():
    cmd_str = "dcm2bids_scaffold"
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _run_dcm2bids(sub_id, config, output_path, dicom_path, session=None):
    cmd_str = "dcm2bids -p {} -c {} -o {} -d '{}'".format(sub_id, config,
                                                          output_path, dicom_path)
    if session is not None:
        cmd_str += " -s {}".format(session)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _run_fmriprep(in_path, out_path, sub_id):
    cmd_str = "sudo fmriprep-docker {} {} --participant_label {} --fs-license-file /home/varsha/Shreya/BIDS/freesurfer.txt --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2".format(in_path, out_path, sub_id)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _run_fmriprep_singularity(data_path, in_path, out_path, sub_id):
    cmd_str = "singularity run --bind {}:/data --cleanenv ~/my_images/fmriprep-1.4.1.simg \
                {} {} participant \
                --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2 \
                --participant-label {} --fs-license-file /data/freesurfer.txt \
                --omp-nthreads 8 --nthreads 12 --mem_mb 30000".format(data_path, in_path, out_path, sub_id)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

# params = vars(_cli_parser())
# print(params)

# dirname = params['d']

file1 = open('fmriprep_todo_list.txt', 'r') 
count = 0
  
while True: 
    count += 1
  
    # Get next line from file 
    line = file1.readline() 
  
    # if line is empty 
    # end of file is reached 
    if not line: 
        break
    print("Target{}: {}".format(count, line.strip())) 

    dirname = line.strip()

    sid = dirname.split("_")
    # indir = os.path.join("/home/nabarun/RADC/sym_radc/BIDS/", dirname)
    # outdir = os.path.join("/home/nabarun/RADC/sym_radc/fmriprep", dirname)

    datadir = "/home/nabarun/RADC/sym_radc/"
    indir = os.path.join("/data/BIDS/", dirname)
    outdir = os.path.join("/data/fmriprep/", dirname)

    # _run_fmriprep(indir, outdir, sid[0])
    _run_fmriprep_singularity(datadir, indir, outdir, sid[0])

    print("Done")
    hs = open(os.path.join("/home/nabarun/RADC/", "fmriprep_hist.txt"), "a")
    hs.write(str(dirname) + "\n")
    hs.close() 
