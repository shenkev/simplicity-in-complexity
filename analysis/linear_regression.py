from metrics import _pearson, _spearman, _loglikelihood, _aic, _bic, _r2, _rmse
import statsmodels.formula.api as smf


def line_regression(target, model_str, df_train, df_test, return_preds=False):

    modelFE = f"{target} ~ {model_str}"
    model = smf.ols(formula=modelFE, data=df_train).fit()

    # get metrics
    y_train = model.predict(df_train)
    y_test = model.predict(df_test)

    pearson_train = _pearson(y_train, df_train[target])
    pearson_test = _pearson(y_test, df_test[target])

    spearman_train = _spearman(y_train, df_train[target])
    spearman_test = _spearman(y_test, df_test[target])

    ll_train = _loglikelihood(model)

    aic_train = _aic(y_train, df_train[target], n=len(df_train), k=len(model.params))
    bic_train = _bic(y_train, df_train[target], n=len(df_train), k=len(model.params))

    r2_train = _r2(y_train, df_train[target])
    r2_test = _r2(y_test, df_test[target])

    rmse_train = _rmse(y_train, df_train[target])
    rmse_test = _rmse(y_test, df_test[target])

    out = {
        "pearson_train": pearson_train,
        "pearson_test": pearson_test,
        "spearman_train": spearman_train,
        "spearman_test": spearman_test,
        "ll_train": ll_train,
        "aic_train": aic_train,
        "bic_train": bic_train,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
    }

    if return_preds: out["predictions"] = y_test

    return out