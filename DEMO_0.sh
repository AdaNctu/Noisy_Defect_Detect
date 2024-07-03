#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  echo $RESULTS_PATH
  SAVE_IMAGES=$1; shift
  echo $SAVE_IMAGES
  GPUS=($@)

  
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c1 $RESULTS_PATH 246 200 0.005 2 True True 10 3 4.0 0.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c2 $RESULTS_PATH 246 200 0.005 2 True True 10 3 4.0 2.0 ${GPUS[1]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c3 $RESULTS_PATH 246 200 0.005 2 True 2.0 True 10 3 4.0 4.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c4 $RESULTS_PATH 246 200 0.005 2 True 4.0 True 10 3 4.0 4.0 ${GPUS[1]} &
  #wait
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_nc3 $RESULTS_PATH 246 200 0.005 2 True 2.0 True 10 2 4.0 4.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_nc4 $RESULTS_PATH 246 200 0.005 2 True 4.0 True 10 2 4.0 4.0 ${GPUS[1]} &
  #wait
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c5 $RESULTS_PATH 246 200 0.005 2 True 2.0 True 10 3 4.0 2.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c6 $RESULTS_PATH 246 200 0.005 2 True 4.0 True 10 3 4.0 2.0 ${GPUS[1]} &
  #wait
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_nc5 $RESULTS_PATH 246 200 0.005 2 True 2.0 True 10 2 4.0 2.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_nc6 $RESULTS_PATH 246 200 0.005 2 True 4.0 True 10 2 4.0 2.0 ${GPUS[1]} &
  
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c6 $RESULTS_PATH 246 200 0.005 2 True 4.0 True 10 3 4.0 2.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_nc6 $RESULTS_PATH 246 200 0.005 2 True 4.0 True 10 2 4.0 2.0 ${GPUS[1]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c7 $RESULTS_PATH 246 200 0.001 2 True 4.0 True 10 3 4.0 2.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c8 $RESULTS_PATH 246 200 0.001 2 True 4.0 True 10 3 4.0 4.0 ${GPUS[1]} &
  #wait
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c9 $RESULTS_PATH 246 200 0.001 2 True 1.0 True 10 3 4.0 2.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c10 $RESULTS_PATH 246 200 0.001 2 True 1.0 True 10 3 4.0 1.0 ${GPUS[1]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c1 $RESULTS_PATH 246 200 0.001 2 True 1.0 True 10 3 4.0 4.0 ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_easy_c2 $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 3 4.0 4.0 ${GPUS[1]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_cheat  $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 3 4.0 0.0 False ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_ncheat $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 2 4.0 0.0 False ${GPUS[1]} &
  #wait
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_cheat2  $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 3 4.0 2.0 False ${GPUS[0]} &
  #train_single $SAVE_IMAGES PCB $PCB_PATH N_nc_co1  $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 2 4.0 0.0 True ${GPUS[1]} &
  train_single $SAVE_IMAGES PCB $PCB_PATH N_nc_co1  $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 2 4.0 0.0 True ${GPUS[0]} &
  train_single $SAVE_IMAGES PCB $PCB_PATH N_nc_co1  $RESULTS_PATH 246 200 0.005 2 True 1.0 True 10 2 4.0 0.0 True ${GPUS[1]} &
  wait

}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results4 True "${GPUS[@]}"

