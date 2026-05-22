

CAMERA="right"
DATASET="faraz" # all, cube_gold
MODEL="pi05_robocasa365"
DATA_ROOT="/mnt/sda/edward/projects/toy_cube_benchmark/${DATASET}"
RESULTS_ROOT="/mnt/sda/edward/data_attn/${MODEL}/${DATASET}_action/${CAMERA}"

# export GEMINI_API_KEY=AIzaSyDWbMdvQBaScuPrj1Za8wzqWtFGTd0tp0M # AIzaSyDWbMdvQBaScuPrj1Za8wzqWtFGTd0tp0M
# DATA_ROOT="/data3/tonyw/toy_cube_benchmark/${DATASET}"
# RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05_vis/${DATASET}"

# Set bash to exit immediately if any command fails 
set -e

echo "DATA_ROOT: ${DATA_ROOT}"
echo "CAMERA: ${CAMERA}"
echo "DATASET: ${DATASET}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "start labeling objects...! "
read -p "Press Enter to continue"
# uv run viz/perception_pipeline.py $DATA_ROOT
echo "start visualizing openpi05...! "
uv run viz/pipeline.py $DATA_ROOT $RESULTS_ROOT --checkpoint ./checkpoints/viz/${MODEL}_pytorch --no-counterfactual