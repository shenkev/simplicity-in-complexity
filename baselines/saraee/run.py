import os
import cv2
import glob
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms._presets import ImageClassification


device = torch.device("cuda")


class ImagesDataset(Dataset):

    def __init__(self, input_path, preprocess_transform):
        # self.preprocess_transform = preprocess_transform
        self.preprocess_transform = ImageClassification(resize_size=448, crop_size=448)
        self.input_path = input_path
        self.image_paths = glob.glob("{}/*".format(input_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # uint8, values 0-255
        im = convert_image_dtype(torch.from_numpy(im), torch.float32)  # still [H, W, C] format
        im = im.permute(2, 0, 1)

        im = self.preprocess_transform(im)

        fname = os.path.basename(image_path)

        return im, fname


def load_preprocess_and_models(layers=[2, 4, 7, 9, 12, 14, 16, 19, 21, 23, 26, 28, 30]):  # all possible relu outputs

    m = models.vgg16(weights='IMAGENET1K_V1', progress=True)
    t = models.VGG16_Weights.IMAGENET1K_V1.transforms()

    model_list = []
    m_list = list(m.children())

    for i in layers:
        model_list.append(m_list[0][:i])    

    assert len(model_list) == len(layers)

    for m in model_list:
        assert type(m[-1]) == torch.nn.modules.activation.ReLU

    model_list = [m.to(device) for m in model_list]

    return t, model_list


def run(loader, model_list, output_path, single_layer):

    filename_list = []
    predictions_list = []

    for batch_idx, (images, filenames) in enumerate(tqdm(loader)):
        images = images.to(device)

        output_list = [m(images) for m in model_list]
        output_list = [x.view(x.size(0), -1).mean(1).detach().cpu().numpy() for x in output_list]
        output_arr = np.stack(output_list, axis=1)

        filename_list.extend(filenames)
        predictions_list.append(output_arr)

    predictions = np.concatenate(predictions_list, axis=0)

    if predictions.shape[1] == 1:
        df = pd.DataFrame({
            "filename": filename_list,
            "vgg16_uae_layer{}".format(single_layer): predictions.squeeze()
        })
        df.to_pickle(output_path)
    else:
        pickle.dump({
            "filenames": filename_list,
            "predictions": predictions
        }, open(output_path, "wb"))


output_path = "/home/tshen/projects/object-centric-complexity/baselines/saraee/vgg16_rsivl.p"
input_path = "/ptmp/tshen/shared/RSIVL/images"
layers=[26]

preprocess, model_list = load_preprocess_and_models(layers=layers)
dataset = ImagesDataset(input_path, preprocess)
inference_loader = DataLoader(dataset, batch_size=32, shuffle=False)

run(inference_loader, model_list, output_path, single_layer=layers[0])