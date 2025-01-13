#!/bin/bash

# Log files
CLUSTER_LOG_FILE="prism_cluster_run.log"
EMBEDDED_LOG_FILE="prism_embedded_run.log"
FMS_LOG_FILE="prism_fms_run.log"
KANBAN_LOG_FILE="prism_kanban_run.log"
MAPK_LOG_FILE="prism_mapk_run.log"
POLL_LOG_FILE="prism_poll_run.log"
TANDEM_LOG_FILE="prism_tandem_run.log"

# Clear the log files (optional)
> $CLUSTER_LOG_FILE
> $EMBEDDED_LOG_FILE
> $FMS_LOG_FILE
> $KANBAN_LOG_FILE
> $MAPK_LOG_FILE
> $POLL_LOG_FILE
> $TANDEM_LOG_FILE

# Arrays for cluster parameters
CLUSTER_N_VALUES=(2 4 8 16 32 64 128 256 512)

# Arrays for embedded parameters
EMBEDDED_MAX_COUNT_VALUES=(2 3 4 5 6 7 8)

# Arrays for fms parameters
FMS_N_VALUES=(1 2 3 4 5 6 7 8 9 10)

# Arrays for kanban parameters
KANBAN_T_VALUES=(1 2 3 4 5 6 7)

# Arrays for mapk_cascade parameters
MAPK_N_VALUES=(1 2 3 4 5 6)

# Poll models
POLL_MODELS=("poll3.sm" "poll4.sm" "poll5.sm" "poll10.sm" "poll11.sm" "poll12.sm")

# Arrays for tandem parameters
TANDEM_C_VALUES=(5 7 15 31 63 127 255 511 1023 2047 4095)

# Run cluster commands
echo "Running CLUSTER Commands..."
for N in "${CLUSTER_N_VALUES[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/cluster/cluster.sm -const N=$N benchmarks/ctmcs/cluster/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $CLUSTER_LOG_FILE
done

# Run embedded commands
echo "Running EMBEDDED Commands..."
for MAX_COUNT in "${EMBEDDED_MAX_COUNT_VALUES[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/embedded/embedded.sm -const MAX_COUNT=$MAX_COUNT benchmarks/ctmcs/embedded/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $EMBEDDED_LOG_FILE
done

# Run fms commands
echo "Running FMS Commands..."
for N in "${FMS_N_VALUES[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/fms/fms.sm -const n=$N benchmarks/ctmcs/fms/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $FMS_LOG_FILE
done

# Run kanban commands
echo "Running KANBAN Commands..."
for T in "${KANBAN_T_VALUES[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/kanban/kanban.sm -const t=$T benchmarks/ctmcs/kanban/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $KANBAN_LOG_FILE
done

# Run mapk_cascade commands
echo "Running MAPK_CASCADE Commands..."
for N in "${MAPK_N_VALUES[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/mapk_cascade/mapk_cascade.sm -const N=$N benchmarks/ctmcs/mapk_cascade/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $MAPK_LOG_FILE
done

# Run poll commands
echo "Running POLL Commands..."
for MODEL in "${POLL_MODELS[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/poll/$MODEL benchmarks/ctmcs/poll/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $POLL_LOG_FILE
done

# Run tandem commands
echo "Running TANDEM Commands..."
for C in "${TANDEM_C_VALUES[@]}"; do
  COMMAND="bin/prism benchmarks/ctmcs/tandem/tandem.sm -const c=$C benchmarks/ctmcs/tandem/px.pctl -gpu"
  echo "Executing: $COMMAND"
  $COMMAND | tee -a $TANDEM_LOG_FILE
done

echo "All tasks completed. Check the log files:"
echo "  CLUSTER Log: $CLUSTER_LOG_FILE"
echo "  EMBEDDED Log: $EMBEDDED_LOG_FILE"
echo "  FMS Log: $FMS_LOG_FILE"
echo "  KANBAN Log: $KANBAN_LOG_FILE"
echo "  MAPK_CASCADE Log: $MAPK_LOG_FILE"
echo "  POLL Log: $POLL_LOG_FILE"
echo "  TANDEM Log: $TANDEM_LOG_FILE"

