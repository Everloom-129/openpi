# TODO list 

- [x] hacky viusalize overall map
    - [x] overlay heatmap on image
    - [x] resize image to 224x224

- [ ] viusalize all individual heads
    - [x] vibe checking

- [ ] improve output into single big image
- [x] combine output into video

- [ ] understand embedding type
- [ ] viusalize each step?
    - [ ] understand denoising effect 
    - [ ] 

### Q&A
- [ ] Ask Edward
- [ ] Ask Manling
- [ ] Ask Dinesh
- [ ] Ask 


### writing 
- [x] overleaf start
    - [ ] 
    - [ ] reference list 
- [ ] literature review
- [ ]


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

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/exx/.cache/openpi/openpi-assets/checkpoints/pi05_droid/ \
    --config_name pi05_droid \
    --output_path /home/exx/Project_pi05/openpi/checkpoints/viz/pi05_droid_pytorch
```

- [x] this is not supported for convert script 
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/tony/.cache/openpi/openpi-assets/checkpoints/pi0_fast_droid/ \
    --config_name pi0_fast_droid \
    --output_path /home/tony/projects/openpi/checkpoints/viz/pi0_fast_droid_pytorch


```