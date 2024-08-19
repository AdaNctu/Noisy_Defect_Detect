#!/usr/bin/env bash

DAGM_PATH="./datasets/DAGM/"
KSDD2_PATH="./datasets/KSDD2/"
PCB_PATH="./datasets/PCB/"

train_single()
{
    SAVE_IMAGES=$1;shift
    DATASET=$1; shift
    DATASET_PATH=$1; shift
    RUN_NAME=$1; shift
    RESULTS_PATH=$1; shift
    NUM_SEGMENTED=$1; shift

    EPOCHS=$1; shift
    LEARNING_RATE=$1; shift
    BATCH_SIZE=$1; shift
    WEIGHTED_SEG_LOSS=$1; shift
    WEIGHTED_DEFECT=$1; shift
    NOISE_RATE=$1; shift
    CLEAN_TRAIN=$1; shift
    COTRAIN=$1; shift
    DROP_RATE=$1; shift
    GMM_SINGLE=$1; shift

    GPU=$1; shift

    RUN_ARGS="--SAVE_IMAGES=$SAVE_IMAGES --DATASET_PATH=$DATASET_PATH --NUM_SEGMENTED=$NUM_SEGMENTED --RUN_NAME=$RUN_NAME --RESULTS_PATH=$RESULTS_PATH --DATASET=$DATASET --EPOCHS=$EPOCHS --LEARNING_RATE=$LEARNING_RATE --BATCH_SIZE=$BATCH_SIZE --WEIGHTED_SEG_LOSS=$WEIGHTED_SEG_LOSS --VALIDATE=True --NOISE_RATE=$NOISE_RATE --CLEAN_TRAIN=$CLEAN_TRAIN --WEIGHTED_DEFECT=$WEIGHTED_DEFECT --COTRAIN=$COTRAIN --DROP_RATE=$DROP_RATE --GMM_SINGLE=$GMM_SINGLE"
    
    LOG_REDIRECT=$RESULTS_PATH/$DATASET/$RUN_NAME/training_log.txt

    mkdir -p $RESULTS_PATH/$DATASET/$RUN_NAME/ && python -u train_net.py --GPU=$GPU $RUN_ARGS | /usr/bin/tee $LOG_REDIRECT


}

train_DAGM()
{
    SAVE_IMAGES=$1;shift
    DATASET=$1; shift
    DATASET_PATH=$1; shift
    RUN_NAME=$1; shift
    RESULTS_PATH=$1; shift
    NUM_SEGMENTED=$1; shift

    EPOCHS=$1; shift
    LEARNING_RATE=$1; shift
    BATCH_SIZE=$1; shift
    WEIGHTED_SEG_LOSS=$1; shift
    WEIGHTED_DEFECT=$1; shift
    NOISE_RATE=$1; shift
    CLEAN_TRAIN=$1; shift
    COTRAIN=$1; shift
    DROP_RATE=$1; shift
    GMM_SINGLE=$1; shift

    GPUS=($@)
    N=${#GPUS[*]}
    echo Will evaluate on "$N" GPUS!
    class=1
    
    RUN_ARGS="--SAVE_IMAGES=$SAVE_IMAGES --DATASET_PATH=$DATASET_PATH --NUM_SEGMENTED=$NUM_SEGMENTED --RUN_NAME=$RUN_NAME --RESULTS_PATH=$RESULTS_PATH --DATASET=$DATASET --EPOCHS=$EPOCHS --BATCH_SIZE=$BATCH_SIZE --WEIGHTED_SEG_LOSS=$WEIGHTED_SEG_LOSS --VALIDATE=True --NOISE_RATE=$NOISE_RATE --CLEAN_TRAIN=$CLEAN_TRAIN --MEMORY_FIT=$BATCH_SIZE --COTRAIN=$COTRAIN --DROP_RATE=$DROP_RATE --LEARNING_RATE=$LEARNING_RATE --GMM_SINGLE=$GMM_SINGLE"
    
    for (( ;; ));
    do
      for j in $(seq 0 $(( $N - 1 )));
      do
          LOG_REDIRECT=$RESULTS_PATH/DAGM/$RUN_NAME/FOLD_$class/training_log.txt
          if [ "$class" -eq 2 ]; then
            WEIGHTED_DEFECT2=$(echo "$WEIGHTED_DEFECT * 4.0" | bc)
          elif [ "$class" -eq 4 ]; then
            WEIGHTED_DEFECT2=$(echo "$WEIGHTED_DEFECT * 2.0" | bc)
          else
            WEIGHTED_DEFECT2=$(echo "$WEIGHTED_DEFECT * 1.0" | bc)
          fi
          RUN_ARGS2="--WEIGHTED_DEFECT=$WEIGHTED_DEFECT2"
          mkdir -p $RESULTS_PATH/DAGM/$RUN_NAME/FOLD_$class && python -u train_net.py --GPU=${GPUS[$j]} --FOLD=$class $RUN_ARGS $RUN_ARGS2 | /usr/bin/tee $LOG_REDIRECT &

          class=$(( $class + 1 ))
          [[ $class -eq 7 ]] && break
      done
      sleep 1
      wait
      [[ $class -eq 7 ]] && break
    done
    wait

}
