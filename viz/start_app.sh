if [ "$(whoami)" = "edward" ]; then
  echo "Hello Edward! Starting Pi0.5 Attention Visualization Dashboard..."
    export RESULTS_ROOT="/mnt/sda/edward/projects/pi05_vis/cube_gold_cf"

else
  echo "Warning: You are running as $(whoami), use default results root /data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold"
  export RESULTS_ROOT="/data3/tonyw/toy_cube_benchmark/pi05_vis/cube_gold"
fi


uv run streamlit run viz/dashboard/app.py 