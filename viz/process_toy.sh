

CAMERA="right"
DATASET="cube_gold" # all, cube_gold
DATA_ROOT="/mnt/sda/edward/projects/toy_cube_benchmark/${DATASET}"
RESULTS_ROOT="/mnt/sda/edward/projects/pi05_vis/${DATASET}/${CAMERA}"


# DATA_ROOT="/data3/tonyw/toy_cube_benchmark/${DATASET}"
# RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05_vis/${DATASET}"

# Set bash to exit immediately if any command fails 
set -e

echo "DATA_ROOT: ${DATA_ROOT}"
echo "CAMERA: ${CAMERA}"
echo "DATASET: ${DATASET}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "start visualizing openpi05...! "
read -p "Press Enter to continue"
uv run viz/pipeline.py $DATA_ROOT $RESULTS_ROOT  --no-counterfactual