# GravityFace Training Code

This directory contains PyTorch code for face recognition training and evaluation. It includes iResNet and MobileFaceNet backbones, margin-based face losses, a dual-view gravitational loss module, dataset helpers, evaluation scripts, and data conversion utilities.

## Project Structure

- `train.py`: main training entry point.
- `configs/ms1mv3.py`: default training configuration.
- `dataset.py`: loads a pickle file containing image paths and class labels.
- `backbones/`: supported model backbones, including `r18`, `r34`, `r50`, `r100`, `r200`, `mbf`, and `mbf_large`.
- `losses.py`: ArcFace, CosFace, AdaFace, SFace, CurricularFace, and related loss implementations.
- `partial_fc_v2.py`: `DualViewGravitationalLoss` implementation used by the current training script.
- `logger.py`: console, file, TensorBoard, and CSV training logging.
- `test/`: evaluation scripts for IJBB, IJBC, TinyFace, and common verification sets.
- `utils/`: dataset list generation and MXNet `.rec` / `.bin` conversion utilities.

## Environment

The provided `requirement.txt` targets Python 3.9.

```bash
conda create -n gravityface39 python=3.9
conda activate gravityface39
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
```

For GPU training, install a CUDA-compatible PyTorch build for your machine. The pinned PyTorch versions in `requirement.txt` are chosen for Python 3.9 compatibility.

MXNet is only required by the conversion utilities in `utils/rec2img.py` and `utils/bin2img.py`. The CUDA MXNet package listed in `requirement.txt` is limited to Linux wheels; on Windows, prepare converted pickle/image data first or run the MXNet conversion tools in Linux/WSL.

## Training Data Format

`dataset.py` expects `cfg.rec` to point to a pickle file. The pickle should contain a list of image-label pairs:

```python
[
    ("path/to/image_0001.jpg", 0),
    ("path/to/image_0002.jpg", 0),
    ("path/to/image_1001.jpg", 1),
]
```

Each image path must be readable by Pillow, and each label should be an integer class id.

The helper scripts in `utils/` can generate or modify pickle image lists from image folders. Update their hard-coded paths before running them.

## Configuration

Edit `configs/ms1mv3.py` before training. The most important fields are:

- `rec`: path to the training pickle file.
- `num_classes`: number of identities/classes in the training set.
- `num_image`: number of training images.
- `network`: backbone name, such as `r50` or `r100`.
- `batch_size`: batch size per training run.
- `embedding_size`: feature dimension, usually `512`.
- `device`: `cuda` or `cpu`.
- `output`: directory for logs and saved weights.
- `num_epoch`: number of training epochs.

## Run Training

From this directory:

```bash
python train.py --config ms1mv3
```

The training script writes logs and artifacts under `cfg.output`, including:

- `training.log`
- `tensorboard/`
- `loss.csv`
- `checkpoint_gpu_<epoch>.pt` when `save_all_states=True`
- `model.pt` from the final save block

View TensorBoard logs with:

```bash
tensorboard --logdir Output/tensorboard
```

## Evaluation

Evaluation scripts are located in `test/`. They assume CUDA by default and expect a backbone state dict compatible with `backbones.get_model()`.

Example commands:

```bash
python test/eval_ijbb.py --model_path Output/model.pt --result_dir Output/EVAL --network r100 --embedding_size 512
python test/eval_ijbc.py --model_path Output/model.pt --result_dir Output/EVAL --network r100 --embedding_size 512
python test/validate_tinyface.py --model_path Output/model.pt --result_dir Output/EVAL --network r100 --embedding_size 512 --data_root E:\Dataset\TinyFace
python test/verif.py --model_path Output/model.pt --result_dir Output/EVAL --network r100
```

Before running evaluation, update hard-coded dataset roots in the scripts:

- `test/eval_ijbb.py`: `image_path`
- `test/eval_ijbc.py`: `image_path`
- `test/validate_tinyface.py`: `--data_root`
- `test/verif.py`: `DATA_ROOT`
- `test/run_all_evals.py`: `BASE_DIR`

Evaluation summaries are written into the result directory through `test/eval_summary.py`.

## Notes

- Most training and evaluation paths are currently configured as absolute local paths. Change them to match your dataset layout before running.
- The default config uses `device = "cuda"` and `fp16 = True`.
- If Python package installation fails around MXNet, first install the core training dependencies and skip MXNet unless you need `.rec` or `.bin` conversion.
- Keep the network argument used for evaluation consistent with the backbone used during training.
