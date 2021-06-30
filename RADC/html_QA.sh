#!/bin/bash
BASEDIR="/media/varsha/Seagate\ Backup\ Plus\ Drive/MRI_DEVARAJAR/RADC_Dicom/fmriprep"
cd /media/varsha/Seagate\ Backup\ Plus\ Drive/MRI_DEVARAJAR/RADC_Dicom/fmriprep
TODIR="$HOME/RADC/html_files"
mkdir -p $TODIR
todaydate=$( date +'%Y-%m-%d' )
filename="$TODIR/$todaydate-FMRIprep-output-on-BIDS-format-of-RADC-data.md"
echo "---" > $filename
echo "toc: true" >> $filename
echo "layout: post" >> $filename
echo "description: FMRIprep output on BIDS format of RADC data" >> $filename
echo "categories: [neuroscience]" >> $filename
echo "title: FMRIprep output on BIDS format of RADC data" >> $filename
echo "---" >> $filename
echo "<table class=\"tableizer-table\">" >> $filename
echo "<thead><tr class=\"tableizer-firstrow\"><th>Subject ID </th></tr></thead><tbody>" >> $filename
for d in ./*/fmriprep/sub-*.html; do
    v=$(($RANDOM % 100))
    if [ $v -le 4 ]; then
        folder=$( echo $d | sed "s/[A-Za-z0-9_.]*\///g" )
        folder=$( echo $folder | sed "s/.html//" )
        folderpath=$( echo $d | sed "s/.html//" )
        echo $d $folderpath
        mkdir -p $TODIR/$folder
        cp -r $folderpath/figures $TODIR/$folder
        echo "<tr><td><a href = 'https://nabarunsarkar.com/img/fmriprep_RADC/$folder.html'>$folder</a> </td></tr>" >> $filename
        cp $d $TODIR
    fi
done
echo "</tbody></table>" >> $filename