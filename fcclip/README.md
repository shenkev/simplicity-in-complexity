# Running FC-CLIP

## Setup

Clone our fork of the fc-clip repo.
```
https://github.com/shenkev/fc-clip.git
```

Install the required dependencies: `pip install -r requirements`.

Then install [detectron2](https://github.com/facebookresearch/detectron2): `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
```

Install the `imutils` package, which is required for running `inference.py`:  `pip install imutils`

Download the model checkpoint from https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view and put it into the root folder ./fc-clip

Run inference using the `inference.py` file.

## Troubleshooting

* An incorrect version of `open-clip-torch` may cause a size mismatch with `attn_mask`. Ensure that `open-clip-torch==2.24.0` is installed (as of 21/09/24).
* During the installation of [detectron2](https://github.com/facebookresearch/detectron2), you might encounter issues with the existing `torch` installation ([related issue](https://github.com/facebookresearch/detectron2/pull/4234)). A workaround is to clone the repository locally, add a `pyproject.toml`, and then install the dependencies:
```cmd
git clone https://github.com/facebookresearch/detectron2.git
# Add the pyproject.toml
python -m pip install -e detectron2
```
Please add the following `pyproject.toml` file to the root folder of `detectron2`:

```toml
[build-system]
requires = ["setuptools", "torch"]
build-backend = "setuptools.build_meta"
```
