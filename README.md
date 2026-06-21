# Pix2Pix Maps

Satellite-to-map image translation using a Pix2Pix-style UNet with PatchGAN discriminator. Trained on the [Berkeley Maps dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).

## Setup

```bash
# Python 3.12 + CUDA 12.6
pip install -r requirements-cuda126.txt
pip install pyyaml
```

## Download Dataset

```bash
python download_pix2pix_maps.py
```

## Training

```bash
# Fresh training (all 4 GPUs)
./train.sh

# Custom settings
python train_pix2pix.py --data-root datasets/maps --epochs 200 --batch-size 2

# Resume from latest checkpoint
./resume-latest.sh

# Resume from best checkpoint
./resume-best.sh
```

## Inference

```bash
./infer.sh
# or
python infer_pix2pix.py \
  --checkpoint runs/pix2pix_maps/checkpoints/best.pt \
  --input-path datasets/maps/val \
  --output-dir runs/inference \
  --paired-input \
  --source-side left
```

## Google Colab

Open `colab_train.ipynb` in Colab, enable GPU, and run all cells. Checkpoints save to Google Drive.

## Project Structure

```
configs/              YAML config files
data/                 Dataset loader
models/               UNet generator + PatchGAN discriminator
train_pix2pix.py      Training script (DDP, AMP, early stopping)
infer_pix2pix.py      Inference script
download_pix2pix_maps.py
```
