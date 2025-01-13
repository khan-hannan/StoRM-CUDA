#!/bin/bash

# Log files
NAND_LOG_FILE="prism_nand_run.log"
BRP_LOG_FILE="prism_brp_run.log"
CROWDS_LOG_FILE="prism_crowds_run.log"
EGL_LOG_FILE="prism_egl_run.log"
HERMAN_LOG_FILE="prism_herman_run.log"
LEADER_SYNC_LOG_FILE="prism_leader_sync_run.log"

# Clear the log files (optional)
> $NAND_LOG_FILE
> $BRP_LOG_FILE
> $CROWDS_LOG_FILE
> $EGL_LOG_FILE
> $HERMAN_LOG_FILE
> $LEADER_SYNC_LOG_FILE

# Arrays for nand parameters
NAND_N_VALUES=(20 40 60)
NAND_K_VALUES=(1 2 3 4)

# Arrays for brp parameters
BRP_N_VALUES=(16 32 64)
BRP_MAX_VALUES=(2 3 4 5)

# Arrays for crowds parameters
CROWDS_RUN_VALUES=(3 4 5 6)
CROWDS_SIZE_VALUES=(5 10 15 20)

# Arrays for egl parameters
EGL_N_VALUES=(5)
EGL_L_VALUES=(2 4 6 8)

# Herman models
HERMAN_MODELS=("herman3.pm" "herman5.pm" "herman7.pm" "herman9.pm" "herman11.pm" "herman13.pm" "herman15.pm")

# Leader_sync models
LEADER_SYNC_MODELS=(
  "leader_sync3_2.pm" "leader_sync3_3.pm" "leader_sync3_4.pm"
  "leader_sync4_2.pm" "leader_sync4_3.pm" "leader_sync4_4.pm"
  "leader_sync5_2.pm" "leader_sync5_3.pm" "leader_sync5_4.pm"
  "leader_sync6_2.pm" "leader_sync6_3.pm" "leader_sync6_4.pm"
  "leader_sync6_5.pm" "leader_sync6_6.pm" "leader_sync6_8.pm"
)

# Run nand commands
echo "Running NAND Commands..."
for N in "${NAND_N_VALUES[@]}"; do
  for K in "${NAND_K_VALUES[@]}"; do
    COMMAND="bin/prism benchmarks/dtmcs/nand/nand.pm -const N=$N,K=$K benchmarks/dtmcs/nand/px.pctl -gpu"
    echo "Executing: $COMMAND"
    $COMMAND | tee -a $NAND_LOG_FILE
  done
done

# Run brp commands
echo "Running BRP Commands..."
for N in "${BRP_N_VALUES[@]}"; do
  for MAX in "${BRP_MAX_VALUES[@]}"; do
    COMMAND="bin/prism benchmarks/dtmcs/brp/brp.pm -const N=$N,MAX=$MAX benchmarks/dtmcs/brp/px.pctl -gpu"
    echo "Executing: $COMMAND"
    $COMMAND | tee -a $BRP_LOG_FILE
  done
done

# Run crowds commands
echo "Running CROWDS Commands..."
for RUNS in "${CROWDS_RUN_VALUES[@]}"; do
  for SIZE in "${CROWDS_SIZE_VALUES[@]}"; do
    COMMAND="bin/prism benchmarks/dtmcs/crowds/crowds.pm -const TotalRuns=$RUNS,CrowdSize=$SIZE benchmarks/dtmcs/crowds/px.pctl -gpu"
    echo "Executing: $COMMAND"
    $COMMAND | tee -a $CROWDS_LOG_FILE
  done
done

# Run egl commands
echo "Running EGL Commands..."
for N in "${EGL_N_VALUES[@]}"; do
  for L in "${EGL_L_VALUES[@]}"; do
    COMMAND="bin/prism benchmarks/dtmcs/egl/egl.pm -const N=$N,L=$L benchmarks/dtmcs/egl/px.pctl -gpu"
    echo "Executing: $COMMAND"
    $COMMAND | tee -a $EGL_LOG_FILE
  done
done

# Run herman commands
echo "Running HERMAN Commands..."
for MODEL in "${HERMAN_MODELS[@]}"; do
  COMMAND="bin/prism benchmarks/dtmcs/herman/$MODEL benchmarks/dtmcs/herman/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $HERMAN_LOG_FILE
done

# Run leader_sync commands
echo "Running LEADER_SYNC Commands..."
for MODEL in "${LEADER_SYNC_MODELS[@]}"; do
  COMMAND="bin/prism benchmarks/dtmcs/leader_sync/$MODEL benchmarks/dtmcs/leader_sync/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $LEADER_SYNC_LOG_FILE
done

echo "All tasks completed. Check the log files:"
echo "  NAND Log: $NAND_LOG_FILE"
echo "  BRP Log: $BRP_LOG_FILE"
echo "  CROWDS Log: $CROWDS_LOG_FILE"
echo "  EGL Log: $EGL_LOG_FILE"
echo "  HERMAN Log: $HERMAN_LOG_FILE"
echo "  LEADER_SYNC Log: $LEADER_SYNC_LOG_FILE"

