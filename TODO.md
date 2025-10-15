# TODO list on 

## data for testing
data/visualization/duck
- calibration.json   # ZED intrinsics and extrinsics
- frames/  
│   ├── hand_camera
│   │   ├── 00000.jpg
│   │   ├── ...
│   │   └── 00090.jpg
│   ├── varied_camera_1
│   │   ├── 00000.jpg
│   │   ├── ...
│   │   └── 00090.jpg
│   └── varied_camera_2
│       ├── 00000.jpg
│       ├── ...
│       └── 00090.jpg
├── videos
│   ├── hand_camera.mp4
│   ├── varied_camera_1.mp4
│   └── varied_camera_2.mp4


## convert jax model to pytorch
- [x] done
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/tony/.cache/openpi/openpi-assets/checkpoints/pi05_droid/ \
    --config_name pi05_droid \
    --output_path /home/tony/projects/openpi/checkpoints/viz/pi05_droid_pytorch
```

- [x] this is not supported for convert script 
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/tony/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid/ \
    --config_name pi0_fast_droid \
    --output_path /home/tony/projects/openpi/checkpoints/viz/pi0_fast_droid_pytorch


```