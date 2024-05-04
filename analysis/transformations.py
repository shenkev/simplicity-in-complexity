import numpy as np
from constants import DATASET_NAMES


def sqrt_xform(c):
    return np.sqrt(c)


def rank_xform(s):
    return s.rank(axis=0, method="average")

def get_sqrt_transforms(df):
    df["sqrt_seg_64points"] = sqrt_xform(df["num_seg_64points"])
    df["sqrt_num_classes"] = sqrt_xform(df["num_classes"])

def get_transforms(df):

    df["sqrt_seg_64points"] = sqrt_xform(df["num_seg_64points"])
    df["sqrt_num_classes"] = sqrt_xform(df["num_classes"])
    df["sqrt_num_unique_classes"] = sqrt_xform(df["num_unique_classes"])

    df["rank_seg_64points"] = rank_xform(df["num_seg_64points"])
    df["rank_num_classes"] = rank_xform(df["num_classes"])
    df["rank_num_unique_classes"] = rank_xform(df["num_unique_classes"])

    df["rank_seg_64points_x_num_classes"] = df["rank_seg_64points"] * df["num_classes"]
    df["rank_seg_64points_x_rank_num_classes"] = rank_xform(df["rank_seg_64points"] * df["rank_num_classes"])
    df["rank_seg_64points_x_num_unique_classes"] = df["rank_seg_64points"] * df["num_unique_classes"]
    df["rank_seg_64points_x_rank_num_unique_classes"] = rank_xform(df["rank_seg_64points"] * df["rank_num_unique_classes"])

    df["sqrt_seg_64points_x_num_classes"] = df["sqrt_seg_64points"] * df["num_classes"]
    df["sqrt_seg_64points_x_sqrt_num_classes"] = df["sqrt_seg_64points"] * df["sqrt_num_classes"]
    df["sqrt_seg_64points_x_num_unique_classes"] = df["sqrt_seg_64points"] * df["num_unique_classes"]
    df["sqrt_seg_64points_x_sqrt_num_unique_classes"] = df["sqrt_seg_64points"] * df["sqrt_num_unique_classes"]

    df["rank_entropy"] = sqrt_xform(df["entropy"])
    df["rank_colorfulness"] = sqrt_xform(df["colorfulness"])
    df["rank_visc_symmetry"] = sqrt_xform(df["visc_symmetry"])
    df["rank_edge_density"] = sqrt_xform(df["edge_density"])