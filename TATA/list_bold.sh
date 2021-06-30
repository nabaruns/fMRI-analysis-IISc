#!/bin/bash
echo "task_rest,folder,sub,file,parent_folder" > Tata_files_server.csv
BASEDIR="/media/nabarun/TATA_MRI_Data_RAW"
types="task"
for d in $BASEDIR/BIDS_$types/*; do
    folder=$( echo $d | sed "s/[A-Za-z0-9_]*\///g" )
    sub=$( echo $folder | sed "s/_//g" | cut -c 1-5 )
    files=$( ls -S $d/sub-*/func/ | head -1 )
    echo "$types,$folder,$sub,$files,$d" >> Tata_files_server.csv
done
types="rest"
for d in $BASEDIR/BIDS_$types/*; do
    folder=$( echo $d | sed "s/[A-Za-z0-9_]*\///g" )
    sub=$( echo $folder | sed "s/_//g" | cut -c 1-5 )
    files=$( ls -S $d/sub-*/func/ | head -1 )
    echo "$types,$folder,$sub,$files,$d" >> Tata_files_server.csv
done