#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh


run_DEMO_EXPERIMENTS()
{
  RESULTS_PATH=$1; shift
  echo $RESULTS_PATH
  SAVE_IMAGES=$1; shift
  echo $SAVE_IMAGES
  GPUS=($@)

  
  train_DAGM $SAVE_IMAGES DAGM $DAGM_PATH clean   $RESULTS_PATH 1000 150 0.001 2 True 1.0 0.50 True  False 0.00 "${GPUS[@]}"
  train_DAGM $SAVE_IMAGES DAGM $DAGM_PATH noisy   $RESULTS_PATH 1000 150 0.001 2 True 1.0 0.50 False False 0.00 "${GPUS[@]}"
  train_DAGM $SAVE_IMAGES DAGM $DAGM_PATH co_50   $RESULTS_PATH 1000 150 0.001 2 True 1.0 0.50 False True  0.50 "${GPUS[@]}"
  
  train_single $SAVE_IMAGES PCB $PCB_PATH clean $RESULTS_PATH 335 150 0.001 1 True 1.0 0.50 True  False 0.00 ${GPUS[0]} &
  train_single $SAVE_IMAGES PCB $PCB_PATH noisy $RESULTS_PATH 335 150 0.001 1 True 1.0 0.50 False False 0.00 ${GPUS[1]} &
  train_single $SAVE_IMAGES PCB $PCB_PATH co_50 $RESULTS_PATH 335 150 0.001 1 True 1.0 0.50 False True  0.50 ${GPUS[2]} &
  wait
  
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH clean_40 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.40 True  False 0.00 ${GPUS[0]} &
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH noisy_40 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.40 False False 0.00 ${GPUS[1]} &
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH co_40_40 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.40 False True  0.40 ${GPUS[2]} &
  wait
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH clean_50 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.50 True  False 0.00 ${GPUS[0]} &
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH noisy_50 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.50 False False 0.00 ${GPUS[1]} &
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH co_50_50 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.50 False True  0.50 ${GPUS[2]} &
  wait
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH clean_60 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.60 True  False 0.00 ${GPUS[0]} &
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH noisy_60 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.60 False False 0.00 ${GPUS[1]} &
  train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH co_60_60 $RESULTS_PATH 246 150 0.001 1 True 1.0 0.60 False True  0.60 ${GPUS[2]} &
  wait
}

# Space delimited list of GPU IDs which will be used for training
GPUS=($@)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi
run_DEMO_EXPERIMENTS   ./results True "${GPUS[@]}"

