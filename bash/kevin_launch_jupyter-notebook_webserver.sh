#!/bin/bash

RT_DIR=$SLURM_JOB_TMP/runtime_dir
if ! [ -d "$RT_DIR" ]; then
  mkdir $RT_DIR
fi
export XDG_RUNTIME_DIR=$RT_DIR

#  Startup jupyter-notebook assuming that there is a jupyter-notebook in the PATH
#  Set Notebook memory to 48 Gigabytes
jupyter-notebook --ip=$(hostname -f) --no-browser --NotebookApp.max_buffer_size=48000000000
