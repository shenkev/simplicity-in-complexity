import os
import pickle
import pandas as pd
import numpy as np


def rename_datasets(df):
    for k, v in zip(['RSIVL', 'VISC', 'Savoias-IntDesign', 'Savoias-Objects', 'Savoias-Scenes', 'Savoias-Art', 'Savoias-Sup', 'IC9600'],
                ['rsivl', 'visc', 'sav_int', 'sav_obj', 'sav_sce', 'sav_art', 'sav_sup', 'ic9600']):

        df[v] = df[k]
        del df[k]

    return df                


def restore_file_exts(df):
    df['rsivl']['filename'] = df['rsivl']['filename'] + ".bmp"

    for k in ["visc", "sav_int", "sav_obj", "sav_sce", "sav_art", "sav_sup", "ic9600"]:
        df[k]['filename'] = df[k]['filename'] + ".jpg"

    return df


def load_rsivl_handcrafted():
    fpath = "/ptmp/tshen/shared/Results/rsivl_feats.p"
    data = pickle.load(open(fpath, "rb"))

    paper_file = "/ptmp/tshen/shared/RSIVL/rsivl_features_rsivl_dataset.csv"
    df = pd.read_csv(paper_file)
    df = df[["filename", "NumReg"]]
    df = df.rename(columns={"NumReg": "M8"})
    df['filename'] = df['filename'] + ".bmp"

    data['rsivl'] = data['rsivl'].drop(columns=["M8"])

    data['rsivl'] = pd.concat([data["rsivl"].set_index("filename"), df.set_index("filename")], 
            axis=1, join='inner').reset_index()

    return data


load_rsivl_handcrafted()

def load_visc_handcrafted():
    basefolder = "/ptmp/tshen/shared/complexity-project/ComplexityNN/"
    folders = ["out_RSIVL_metrics", "out_VISC_metrics", "out_Savoias_metrics/out_art", "out_Savoias_metrics/out_id", "out_Savoias_metrics/out_obj", "out_Savoias_metrics/out_sc", "out_Savoias_metrics/out_sup", "out_IC9600_metrics"]

    visc_metrics_baseline = {}

    mapping = {
                "out_RSIVL_metrics": "RSIVL",
                "out_VISC_metrics": "VISC",
                "out_Savoias_metrics/out_art": "Savoias-Art",
                "out_Savoias_metrics/out_id": "Savoias-IntDesign",
                "out_Savoias_metrics/out_obj": "Savoias-Objects",
                "out_Savoias_metrics/out_sc": "Savoias-Scenes",
                "out_Savoias_metrics/out_sup": "Savoias-Sup",
                "out_IC9600_metrics": "IC9600"
            }

    for f in folders:
        visc_metrics_baseline[mapping[f]] = pd.DataFrame(columns = ["image", "visc_metrics_baseline_symmetry", "visc_metrics_baseline_clutter"])
        preds = pd.read_csv(basefolder + f + "/metrics.csv")
        visc_metrics_baseline[mapping[f]]["image"] = preds["file"]
        visc_metrics_baseline[mapping[f]]["visc_metrics_baseline_symmetry"] = preds["symmetry"]
        visc_metrics_baseline[mapping[f]]["visc_metrics_baseline_clutter"] = preds["clutter"]

    visc_metrics_baseline = rename_datasets(visc_metrics_baseline)

    visc_metrics_baseline["rsivl"]["image"] = visc_metrics_baseline["rsivl"]["image"].astype(str).str.replace(r'(imm)(\d+)', r'\1(\2)', regex=True)

    for k, v in visc_metrics_baseline.items():
        visc_metrics_baseline[k] = v.rename(columns={
            "image": "filename", 
            "visc_metrics_baseline_symmetry": "visc_symmetry",
            "visc_metrics_baseline_clutter": "visc_clutter",
            })

    visc_metrics_baseline['rsivl']['filename'] = visc_metrics_baseline['rsivl']['filename'].str.replace('.png', '.bmp', regex=False)

    return visc_metrics_baseline


def load_visc_nn():
    basefolder = "/ptmp/tshen/shared/complexity-project/ComplexityNN/"
    folders = ["out_RSIVL", "out_VISC", "out_Savoias/out_art", "out_Savoias/out_id", "out_Savoias/out_obj", "out_Savoias/out_sc", "out_Savoias/out_sup", "out_IC9600"]

    visc_nn_baseline = {}

    mapping = {
                "out_RSIVL": "RSIVL",
                "out_VISC": "VISC",
                "out_Savoias/out_art": "Savoias-Art",
                "out_Savoias/out_id": "Savoias-IntDesign",
                "out_Savoias/out_obj": "Savoias-Objects",
                "out_Savoias/out_sc": "Savoias-Scenes",
                "out_Savoias/out_sup": "Savoias-Sup",
                "out_IC9600": "IC9600"
            }

    for f in folders:
        visc_nn_baseline[mapping[f]] = pd.DataFrame(columns = ["image", "visc_nn_baseline"])
        preds = pd.read_csv(basefolder + f + "/predictions.csv")
        visc_nn_baseline[mapping[f]]["image"] = preds["Images"].astype('str')
        visc_nn_baseline[mapping[f]]["visc_nn_baseline"] = preds["Scores"] * 100

    visc_nn_baseline = rename_datasets(visc_nn_baseline)

    visc_nn_baseline["rsivl"]["image"] = visc_nn_baseline["rsivl"]["image"].astype(str).str.replace(r'(imm)(\d+)', r'\1(\2)', regex=True)

    for k, v in visc_nn_baseline.items():
        visc_nn_baseline[k] = v.rename(columns={
            "image": "filename", 
            "visc_nn_baseline": "visc_nn",
            })

    visc_nn_baseline = restore_file_exts(visc_nn_baseline)

    return visc_nn_baseline


def load_ic9600_nn():
    basefolder = "/ptmp/tshen/shared/complexity-project/IC9600/"
    folders = ["out_RSIVL", "out_VISC", "out_Savoias/out_art", "out_Savoias/out_id", "out_Savoias/out_obj", "out_Savoias/out_sc", "out_Savoias/out_sup", "out_IC9600test"]

    ic9600_baseline = {}

    mapping = {
                "out_RSIVL": "RSIVL",
                "out_VISC": "VISC",
                "out_Savoias/out_art": "Savoias-Art",
                "out_Savoias/out_id": "Savoias-IntDesign",
                "out_Savoias/out_obj": "Savoias-Objects",
                "out_Savoias/out_sc": "Savoias-Scenes",
                "out_Savoias/out_sup": "Savoias-Sup",
                "out_IC9600test": "IC9600"
            }

    for f in folders:
        ic9600_baseline[mapping[f]] = pd.DataFrame(columns = ["image", "IC9600_baseline"])
        ind = 0
        for out in np.sort(os.listdir(basefolder + f)):
            if out[-3:] == "npy":
                l = out[:-4].split("_")
                imname = '_'. join(l[:-1])
                if f == "out_RSIVL":
                    imname = imname[:-2] + "(" + imname[-2:] + ")"
                ic9600_baseline[mapping[f]].loc[ind] = [imname, float(l[-1])]
                ind += 1


    ic9600_baseline = rename_datasets(ic9600_baseline)

    for k, v in ic9600_baseline.items():
            ic9600_baseline[k] = v.rename(columns={
                "image": "filename", 
                "IC9600_baseline": "ic9600_nn",
                })

    ic9600_baseline = restore_file_exts(ic9600_baseline)

    return ic9600_baseline


def load_savoias_nn():
    basefolder = "/ptmp/tshen/shared/complexity-project/Savoias/"
    files = ["vgg16_rsivl.p", "vgg16_visc.p", "vgg16_art.p", "vgg16_intdesign.p", "vgg16_objects.p", "vgg16_scenes.p", "vgg16_suprematism.p", "vgg16_ic9600.p"]

    Savoias_baseline = {}

    mapping = {
        "vgg16_rsivl.p": "RSIVL",
        "vgg16_visc.p": "VISC",
        "vgg16_art.p": "Savoias-Art",
        "vgg16_intdesign.p": "Savoias-IntDesign",
        "vgg16_objects.p": "Savoias-Objects",
        "vgg16_scenes.p": "Savoias-Scenes",
        "vgg16_suprematism.p": "Savoias-Sup",
        "vgg16_ic9600.p": "IC9600"
    }

    for p in files:
        Savoias_baseline[mapping[p]] = pickle.load(open(basefolder + p, "rb"))
        Savoias_baseline[mapping[p]]["filename"] = Savoias_baseline[mapping[p]]["filename"].str[:-4]
        Savoias_baseline[mapping[p]] = Savoias_baseline[mapping[p]].rename(columns = {"filename": "image", "vgg16_uae_layer26": "Savoias_baseline"})

    Savoias_baseline = rename_datasets(Savoias_baseline)

    for k, v in Savoias_baseline.items():
            Savoias_baseline[k] = v.rename(columns={
                "image": "filename", 
                "Savoias_baseline": "savoias_nn",
                })

    Savoias_baseline = restore_file_exts(Savoias_baseline)

    return Savoias_baseline