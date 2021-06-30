#!/bin/bash

SUBJ=(48104907_00 55598394_00 71598000_00)

singularity run --cleanenv -B /mnt:/mnt /mnt/data/singularity_images/fmriprep-latest.simg \
      /mnt/data/loic2/RSBIDS4 /mnt/data/loic2/fmriprep_output_tw_less participant \
      --participant-label ${SUBJ[$SLURM_ARRAY_TASK_ID-1]} --low-mem --stop-on-first-crash \
      --medial-surface-nan --use-aroma --cifti-output --notrack \
      --output-space template fsaverage5 --fs-license-file /mnt/data/loic2/license.txt \
      --omp-nthreads 8 --nthreads 12 --mem_mb 30000