# Running FC-CLIP

## Setup

First install fc-clip by following the **Example conda environment setup** instructions in the original repository. https://github.com/bytedance/fc-clip/blob/main/INSTALL.md

Clone our fork of the fc-clip repo.
```
https://github.com/shenkev/fc-clip.git
```

Download the model checkpoint from https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view and put it into the root folder ./fc-clip

Additionally, install the `imutils` package for `inference.py`.

Run inference using the `inference.py` file.

## Troubleshooting

* An incorrect version of `open-clip-torch` may cause a size mismatch with `attn_mask`. Ensure that `open-clip-torch==2.24.0` is installed (as of 21/09/24).
* During the installation of [detectron2](https://github.com/facebookresearch/detectron2), you might encounter issues with the existing `torch` installation ([related issue](https://github.com/facebookresearch/detectron2/pull/4234)). A workaround is to clone the repository locally, add a `pyproject.toml`, and then install the dependencies:
```cmd
git clone https://github.com/facebookresearch/detectron2.git
# Add the pyproject.toml
python -m pip install -e detectron2
Please add the following `pyproject.toml` file to the root folder of `detectron2`:
```
```toml
[build-system]
requires = ["setuptools", "torch"]
build-backend = "setuptools.build_meta"
```
