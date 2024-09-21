# Running SAM

## Setup

Create a new Anaconda environment.

```https://github.com/shenkev/fc-clip.git
conda create -n sam python=3.9
```

```
source activate sam
pip3 install torch torchvision torchaudio
```

Clone our fork of the SAM repo.

```
https://github.com/shenkev/segment-anything.git
cd segment-anything; pip install -e .
```

```
pip install opencv-python matplotlib jupyter numpy pycocotools
```

Download the [model weights](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) and save it in the `checkpoints` folder.

## Running inference


```
CUDA_VISIBLE_DEVICES=2 python scripts/amg.py --checkpoint checkpoints/sam_vit_h_4b8939.pth --model-type vit_h --points-per-side 8 --output ./out --input images/cat.png
```

## Viewing the output

The amg.py script saves the masks to the ./out/{image_name}.p file. The ./view_output.ipynb notebook loads this pickle file to visualize the masks. Start the jupyter notebook with,

```
jupyter notebook ./view_output.ipynb
```

Then change the paths to the original image and mask pickle file in the notebook.
