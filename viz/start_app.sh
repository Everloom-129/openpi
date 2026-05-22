if [ "$(whoami)" = "edward" ]; then
  echo "Hello Edward! Starting Pi0.5 Attention Visualization Dashboard..."
    export DATA_ATTN_ROOT="/mnt/sda/edward/data_attn"
    export RESULTS_ROOT="${DATA_ATTN_ROOT}/pi05_droid/cube_gold"

else
  echo "Warning: You are running as $(whoami), use default results root /data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold"
  export RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold"
fi


uv run streamlit run viz/dashboard/app.py 