from linear_regression import line_regression
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut


RANDOM_SEED = 1

def _append(d, k, x):
    if k not in d:
        d[k] = [x]
    else:
        d[k].append(x)

def cross_validate(df, dset_name, N=3, M=1, ic9600_test=False):

    use_stratify = dset_name in ["visc", "ic9600_sce", "sav_obj_sce"]
    print("Running dataset {}, CV is stratified: {}".format(dset_name, use_stratify))

    # model --> metrics (one array for each)
    model_results = {}

    for m in range(M):

        if use_stratify:
            # seed needs to increment to get different splits
            kf = StratifiedKFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED+m)
            splits = list(kf.split(df, df['subcat']))
        else:
            kf = KFold(n_splits=N, shuffle=True, random_state=RANDOM_SEED+m)
            splits = list(kf.split(df))

        for train_idxs, test_idxs in splits:
            df_train, df_test = df.iloc[train_idxs], df.iloc[test_idxs]

            # iterate over models
            def fit_mod(mod_strs, df_tr, df_te):
                for s in mod_strs:
                    if s not in model_results:
                        model_results[s] = {}
                    results = line_regression("complexity", s, df_tr, df_te)
                    for k, v in results.items():
                        _append(model_results[s], k , v)

            # ours
            model_strs = [
                "sqrt_seg_64points",
                "sqrt_num_classes",                
                "sqrt_seg_64points + sqrt_num_classes",
                "sqrt_seg_64points_x_sqrt_num_classes",
                "sqrt_seg_64points_x_sqrt_num_classes + sqrt_seg_64points + sqrt_num_classes",                
            ]

            fit_mod(model_strs, df_train, df_test)

            # rsivl
            if dset_name == "rsivl":
                model_strs = [      
                    "M1 + M2 + M3 + M4 + M5 + M6 + M7 + M9 + M10 + M11",                
                    "M1 + M2 + M3 + M4 + M5 + M6 + M7 + M9 + M10 + M11 + M8",                
                    "M5 + M10",               
                    "M5 + M10 + M8",               
                ]
            else:
                model_strs = [      
                    "M1 + M2 + M3 + M4 + M5 + M6 + M7 + M9 + M10 + M11",                
                    "M5 + M10",               
                ]

            fit_mod(model_strs, df_train, df_test)

            # visc
            model_strs = [
                "visc_symmetry + visc_clutter",
                "visc_nn",
            ]
            
            df_train_nona = df_train[~df_train["visc_clutter"].isna()]
            df_test_nona = df_test[~df_test["visc_clutter"].isna()]

            fit_mod(model_strs, df_train_nona, df_test_nona)

            # savoias
            model_strs = [
                "savoias_nn"
            ]

            fit_mod(model_strs, df_train, df_test)

            # ic9600
            if ic9600_test:
                model_strs = [
                    "ic9600_nn"
                ]
                fit_mod(model_strs, df_train, df_test)

            # ablation
            model_strs = [
                "visc_symmetry",
                "sqrt_seg_64points + sqrt_num_classes + visc_symmetry",
            ]

            df_train_nona = df_train[~df_train["visc_symmetry"].isna()]
            df_test_nona = df_test[~df_test["visc_symmetry"].isna()]

            fit_mod(model_strs, df_train_nona, df_test_nona)

    return model_results