import pandas as pd


def scenes_art_split(data):
    CATS = ["rsivl", "sav_obj_sce", "ic9600_sce", "sav_art", "sav_sup", "ic9600_paint", "visc", "sav_int"]

    data_new = {}
    data_new['rsivl'] = data['rsivl']
    data_new['visc'] = data['visc']
    data_new['sav_int'] = data['sav_int']
    data_new['sav_art'] = data['sav_art']
    data_new['sav_sup'] = data['sav_sup']
    data_new['sav_obj_sce'] =  pd.concat([data['sav_obj'], data['sav_sce']], axis=0, ignore_index=True)
    ic9600_cats = data["ic9600"]["filename"].str.split('_', expand=True)[0]
    data_new['ic9600_sce'] = data["ic9600"][ic9600_cats.isin(["person", "transport", "scenes", "architecture", "objects"])]
    data_new['ic9600_paint'] = data["ic9600"][ic9600_cats == "paintings"]
    
    return data_new, CATS