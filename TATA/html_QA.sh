#!/bin/bash
BASEDIR="/media/nabarun/TATA_MRI_Data_RAW/fmriprep_task"
TODIR="$HOME/TATA/html_files"
mkdir -p $TODIR
todaydate=$( date +'%Y-%m-%d' )
filename="$TODIR/$todaydate-FMRIprep-output-on-BIDS-format-of-TATA-task-data.md"
echo "---" >> $filename
echo "toc: true" >> $filename
echo "layout: post" >> $filename
echo "description: FMRIprep output on BIDS format of TATA task data" >> $filename
echo "categories: [neuroscience]" >> $filename
echo "title: FMRIprep output on BIDS format of TATA task data" >> $filename
echo "---" >> $filename
echo "<table class=\"tableizer-table\">" >> $filename
echo "<thead><tr class=\"tableizer-firstrow\"><th>Subject ID </th></tr></thead><tbody>" >> $filename
cp -r $BASEDIR/*/fmriprep/*.html $TODIR
for d in $BASEDIR/*/fmriprep/sub-*; do
    if [[ "$d" != *.html ]]; then
        folder=$( echo $d | sed "s/[A-Za-z0-9_]*\///g" )
        mkdir -p $TODIR/$folder
        cp -r $d/figures $TODIR/$folder
        echo "<tr><td><a href = 'https://nabarunsarkar.com/img/fmriprep_TATA_task/$folder.html'>$folder</a> </td></tr>" >> $filename
    fi
done
echo "</tbody></table>" >> $filename