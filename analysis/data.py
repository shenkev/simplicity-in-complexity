import pickle
import pandas as pd


def load_complexity():
    rsivl = "/ptmp/tshen/shared/RSIVL/labels.xls"
    visc = "/ptmp/tshen/shared/VISC/VISC-C/labels.csv"
    sav_obj = "/ptmp/tshen/shared/Savoias/labels/xlsx/global_ranking_objects.xlsx"
    sav_sce = "/ptmp/tshen/shared/Savoias/labels/xlsx/global_ranking_scenes.xlsx"
    sav_art = "/ptmp/tshen/shared/Savoias/labels/xlsx/global_ranking_art.xlsx"
    sav_int = "/ptmp/tshen/shared/Savoias/labels/xlsx/global_ranking_interior_design.xlsx"
    sav_sup = "/ptmp/tshen/shared/Savoias/labels/xlsx/global_ranking_sup.xlsx"
    ic9600 = ["/ptmp/tshen/shared/IC9600/train.txt", "/ptmp/tshen/shared/IC9600/test.txt"]

    labels = {}
    
    labels['rsivl'] = pd.read_excel(rsivl)
    labels['rsivl'].rename(columns={'subjective scores': 'complexity', 'Unnamed: 0': 'filename'}, inplace=True)
    labels['rsivl']['filename'] = labels['rsivl']['filename'] + ".bmp"
    labels['rsivl'] = labels['rsivl'][['filename', 'complexity']]

    labels['visc'] = pd.read_csv(visc)
    labels['visc'].rename(columns={'score': 'complexity', 'image': 'filename'}, inplace=True)

    for v, n in zip([sav_int, sav_obj, sav_sce, sav_art, sav_sup], ["sav_int", "sav_obj", "sav_sce", "sav_art", "sav_sup"]):
        labels[n] = pd.read_excel(v)
        labels[n].rename(columns={'gt': 'complexity'}, inplace=True)
        labels[n]['filename'] = ["{}.jpg".format(i) for i in range(len(labels[n]))]

    labels['ic9600'] = pd.concat([pd.read_csv(fn, names=["filename", "complexity"], header=None, delimiter=r"  ") for fn in ic9600])
    labels['ic9600']["complexity"] = labels['ic9600']["complexity"]*100.0
    assert len(labels['ic9600']) == 9600

    for k in labels.keys():
        assert labels[k][['filename', 'complexity']].isnull().values.any() == False

    return labels


def load_sam_features():
    folder="/ptmp/tshen/shared/Results/2023July20" 
    filenames=['4points', '8points', '16points', '32points', '64points']

    rsivl = "{}/RSIVL".format(folder)
    visc = "{}/VISC".format(folder)
    sav_obj = "{}/Savoias-Objects".format(folder)
    sav_sce = "{}/Savoias-Scenes".format(folder)
    sav_art = "{}/Savoias-Art".format(folder)
    sav_int = "{}/Savoias-IntDesign".format(folder)
    sav_sup = "{}/Savoias-Suprematism".format(folder)
    ic9600 = "{}/IC9600".format(folder)

    preds = {}
    
    for v, n in zip([rsivl, visc, sav_int, sav_obj, sav_sce, sav_art, sav_sup, ic9600], ["rsivl", "visc", "sav_int", "sav_obj", "sav_sce", "sav_art", "sav_sup", "ic9600"]):
        preds_dataset = []
        single_file_len = 0

        for f in filenames:
            df = pd.read_csv("{}/{}.csv".format(v, f), names=["filename", f], header=None)
            preds_dataset.append(df)
            single_file_len = len(df)

        combined_df = pd.concat([df.set_index("filename") for df in preds_dataset], axis=1, join='inner').reset_index()  # join on filename
        assert len(combined_df) == single_file_len
        assert combined_df.isnull().values.any() == False

        combined_df = combined_df.rename(columns={
            "4points": "num_seg_4points", 
            "8points": "num_seg_8points",
            "16points": "num_seg_16points",
            "32points": "num_seg_32points",
            "64points": "num_seg_64points",
            })

        preds[n] = combined_df

    preds['rsivl']['filename'] = preds['rsivl']['filename'].str.replace('.json', '.bmp', regex=False)
    for k in ["visc", "sav_int", "sav_obj", "sav_sce", "sav_art", "sav_sup", "ic9600"]:
        preds[k]['filename'] = preds[k]['filename'].str.replace('.json', '.jpg', regex=False)

    return preds


def load_fcclip_features():
     fcclip_feature_path = "/ptmp/tshen/shared/Results/fcclip_labels.p"

     data = pickle.load(open(fcclip_feature_path, "rb"))

     data['sav_int'] = data['int']
     data['sav_obj'] = data['objects']
     data['sav_sce'] = data['scenes']
     data['sav_art'] = data['art']
     data['sav_sup'] = data['sup']

     del data['int']
     del data['objects']
     del data['scenes']
     del data['art']
     del data['sup']

     return data


def load_additional_features():
    feature_path = "/ptmp/tshen/shared/Results/additional_features.p"

    data = pickle.load(open(feature_path, "rb"))

    return data