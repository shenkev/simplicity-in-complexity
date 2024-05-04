import os
import cv2
import glob
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd


def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	# print(image[:2, :2, :])
	# print(B[:2, :2], R[:2, :2], G[:2, :2])
	# print(B.shape, G.shape, R.shape, rg.shape, yb.shape, rbMean, ybStd, stdRoot, meanRoot)
	return stdRoot + (0.3 * meanRoot)


def entropy_gray(img, bins=256):
    marg = np.histogramdd(np.ravel(img), bins=bins)[0]/img.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy


def compute_feats(bins=32, threshold1=500/3, threshold2=500):

    out = {}
    dsets = ["rsivl", "visc", "sav_obj", "sav_sce", "sav_int", "sav_art", "sav_sup", "ic9600"]

    folder_paths = [
        "/ptmp/tshen/shared/RSIVL/images",
        "/ptmp/tshen/shared/VISC/VISC-C/images",
        "/ptmp/tshen/shared/Savoias/images/Objects",
        "/ptmp/tshen/shared/Savoias/images/Scenes",
        "/ptmp/tshen/shared/Savoias/images/IntDesign",
        "/ptmp/tshen/shared/Savoias/images/Art",
        "/ptmp/tshen/shared/Savoias/images/Suprematism",
        "/ptmp/tshen/shared/IC9600/images",
    ]

    for ds, fp in zip(dsets, folder_paths):
        file_paths = list(glob.glob("{}/*".format(fp)))

        edge_density = []
        entropy_arr = []
        colorfullness_arr = []

        for f in tqdm(file_paths):

            try:
                image = cv2.imread(f)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                edges = cv2.Canny(gray, threshold1=threshold1, threshold2=threshold2)
                edge_density.append(edges.mean())

                entropy_arr.append(entropy_gray(gray, bins=bins))

                colorfullness_arr.append(image_colorfulness(image))
            except:
                edge_density.append(None)
                entropy_arr.append(None)
                colorfullness_arr.append(None)

        df = pd.DataFrame({
             "filename": file_paths,
             "edge_density": edge_density,
             "entropy": entropy_arr,
             "colorfulness": colorfullness_arr
        })

        df["filename"] = df["filename"].apply(os.path.basename)

        if ds != "ic9600":
             assert len(df) == len(file_paths)
        else:
             print("WARN: only {}/{} images from ic9600 could compute features".format(
                  len(file_paths) - df['colorfulness'].isna().sum(), len(file_paths)))

        out[ds] = df
    
    pickle.dump(out, open("/ptmp/tshen/shared/Results/additional_features.p", "wb"))


compute_feats()