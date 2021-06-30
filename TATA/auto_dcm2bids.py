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

params = vars(_cli_parser())
print(params)
sub_data = []
sub_count = 1

directories = sorted(os.listdir(params['d']))
for dirname in directories:
    in_path = os.path.join(params['d'], dirname)
    if params['ignore'] is not None:
        if dirname in params['ignore']:
            print('Skipping {}'.format(in_path))
            continue
    else:
        print('Processing {}'.format(in_path))

    x = dirname.split("_")

    sub_id = x[0]
    sess = x[1]

    sub_count += 1
    sub_data.append(dirname)

    outdir = os.path.join(params['o'], dirname)
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    _run_dcm2bids_scaffold()

    _run_dcm2bids(sub_id, params['c'], outdir, in_path)

    tmpdir = os.path.join(outdir, "tmp_dcm2bids")
    _rm_tar(tmpdir)

    rm = open(os.path.join(outdir, "README"),"a")
    rm.write("RADC BIDS")
    rm.close()

    # os.makedirs(os.path.join(params['o'], 'derivatives'), exist_ok=True)
    # Path(os.path.join(params['o'], 'README')).touch()
    # Path(os.path.join(params['o'], 'CHANGES')).touch()
    # Path(os.path.join(params['o'], '.bidsignore')).touch()
    # Path(os.path.join(params['o'], 'dataset_description.json')).touch()

print(sub_data)
hs = open(os.path.join("../../", "auto_dcm_hist.txt"), "a")
hs.write(str(sub_data) + "\n")
hs.close() 