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

def _run_untar(outdir, srcfile):
    cmd_str = "tar -C {} -zxf {}".format(outdir, srcfile)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

def _rm_tar(srcfile):
    cmd_str = "rm -rf {}".format(srcfile)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

# Using readline() 
file1 = open('copy10.txt', 'r') 
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

    subject_id = line.strip()
    srcpath = os.path.join("/media/varsha/Seagate\ Backup\ Plus\ Drive/MRI_DEVARAJAR/RADC_Dicom/mg/160627/",subject_id+".tar.gz")
    destpath = "~/RADC/Data/"
    # print('Copying {} to {}'.format(srcpath, destpath))
    _run_copy(srcpath, destpath)

    srcFile = os.path.join(destpath, subject_id+".tar.gz")
    # print('Unzipping {} to {}'.format(srcFile, destpath))
    _run_untar(destpath, srcFile)

    # print('Removing {} from {}'.format(srcFile, destpath))
    _rm_tar(srcFile)

    print("\n")

    
  
file1.close() 
