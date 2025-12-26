
DATA_ROOT="/data3/tonyw/toy_cube_benchmark/cube_gold"
RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05/cube_gold"

# Set bash to exit immediately if any command fails 
set -e

echo "DATA_ROOT: ${DATA_ROOT}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "start visualizing openpi05...! "
read -p "Press Enter to continue"
uv run viz/pipeline.py $DATA_ROOT $RESULTS_ROOT  