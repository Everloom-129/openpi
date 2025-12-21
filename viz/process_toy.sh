
DATA_ROOT="/data3/tonyw/toy_cube_benchmark"


# Set bash to exit immediately if any command fails 
set -e

echo "DATA_ROOT: ${DATA_ROOT}"
echo "start visualizing openpi05...! "
read -p "Press Enter to continue"
uv run viz/pipeline.py $DATA_ROOT  