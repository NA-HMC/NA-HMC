# Noise-space HMC (N-HMC)


## Getting started 

### 1) Clone the repository

```
git clone https://github.com/Sunsett5/Noise-space-HMC.git

cd Noise-space-HMC
```


### 2) Download pretrained checkpoint


```
pip3 install gdown
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O ./models/ffhq_10m.pt
gdown https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O ./models/imagenet256.pt
```

Download the checkpoint "GOPRO_wVAE.pth"

```
gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O ./experiments/pretrained/
```


### 3) Set environment


Install dependencies. Change {DIR} in sed command to your root location.

```
conda env create -f environment.yml
conda activate NHMC
sed -i 's/torch\._six\.string_classes/str/g' /{DIR}/miniconda3/envs/NHMC/lib/python3.8/site-packages/torchvision/datasets/vision.py
sed -i "s/torch\.load(model_path, map_location='cpu')/torch\.load(model_path, map_location='cpu', weights_only=True)/" /{DIR}/miniconda3/envs/NHMC/lib/python3.8/site-packages/lpips/lpips.py
```

If encounter this bug "ImportError: cannot import name 'VectorQuantizer2' from 'taming.modules.vqvae.quantize'". Download [quantize.py](https://github.com/CompVis/stable-diffusion/issues/72). Then replace this file miniconda/envs/NHMC/lib/python3.8/site-packages/taming/modules/vqvae/quantize.py

### 4) Run experiment
```
python3 main_sampling.py \
    --dataset ffhq \
    --timesteps 2 \
    --deg inpaint_random \
    --noise_type gaussian \
    --sigma_y 0.05 \
    --unknown_noise \
    --image_folder exp/samples/ffhq/inpaint_random \
    --verbose
```


- `--timesteps INT`  
  Number of timesteps. Default: `2`.

- `--deg {sr4, sr16, hdr, random_inpaint, deblur_aniso, deblur_nonlinear, phase_retrieval}`  
  Forward operator.

- `--noise_type {gaussian, speckle, impulse}`  
  Measurement noise type.

- `--sigma_y FLOAT`  
  Standard deviation of measurement noise.

- `--unknown_noise` *(flag)*  
  Use noise-adaptive algorithm.  
  Default: `False`. Set to `True` if provided.

- `--verbose` *(flag)*  
  Enable verbose output.  
  Default: `False`.

- `--image_folder PATH`  
  Output directory for generated images.


## References
This repo is developed based on [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [BlindDPS](https://github.com/BlindDPS/blind-dps), especially for forward operations. We also use the external codes for [motion-blurring](https://github.com/LeviBorodenko/motionblur) and [non-linear deblurring](https://github.com/VinAIResearch/blur-kernel-space-exploring). Please also consider citing them if you use this repo.
