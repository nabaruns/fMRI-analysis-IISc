import os
import argparse
import shutil
import subprocess
import json
import warnings

def _run_copy(source, output_path):
    cmd_str = "cp -r {} {}".format(source, output_path)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _rm_tar(srcfile):
    cmd_str = "rm -rf {}".format(srcfile)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

destpath = "/media/varsha/Seagate\ Backup\ Plus\ Drive/MRI_DEVARAJAR/RADC_Dicom/"
srcpath = "~/RADC/BIDS/"
# print('Copying {} to {}'.format(srcpath, destpath))
_run_copy(srcpath, destpath)

directories = sorted(os.listdir("BIDS"))
for dirname in directories:
    # print(dirname)
    destpath = "~/RADC/Data/"
    srcFile = os.path.join(destpath, dirname)
    # print('Removing {} from {}'.format(srcFile, destpath))
    _rm_tar(srcFile)

    destpath = "~/RADC/BIDS/"
    srcFile = os.path.join(destpath, dirname)
    # print('Removing {} from {}'.format(srcFile, destpath))
    _rm_tar(srcFile)
