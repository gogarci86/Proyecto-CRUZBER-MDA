"""
Evaluación del modelo: métricas globales, análisis de errores y diagnósticos.
"""
import numpy as np
import pandas as pd


def evaluate_global(test_R: pd.DataFrame, test_H: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas globales y por segmento ABC / Syntetos-Boylan.

    Returns df_eval concatenado con columnas real, pred, tipo_abc, sb_class.
    """
    df_eval = pd.concat([test_R, test_H], ignore_index=True)
    y_true = df_eval['real'].values
    y_pred = df_eval['pred'].values

    mae = np.mean(np.abs(y_true - y_pred))
    wmape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    bias_pct = np.mean(y_pred - y_true) / max(np.mean(y_true), 1) * 100

    print(f'\n    {"="*55}')
    print(f'    RENDIMIENTO GLOBAL (12 semanas)')
    print(f'    {"="*55}')
    print(f'    MAE:   {mae:.3f}')
    print(f'    WMAPE: {wmape*100:.1f}%')
    print(f'    R²:    {r2:.3f}')
    print(f'    Bias:  {bias_pct:+.1f}%')

    for abc_cls in ['A', 'B', 'C']:
        m = df_eval['tipo_abc'] == abc_cls
        if m.sum() == 0:
            continue
        yt = df_eval.loc[m, 'real']
        yp = df_eval.loc[m, 'pred']
        w = np.sum(np.abs(yt - yp)) / max(np.sum(np.abs(yt)), 1) * 100
        print(f'    Clase {abc_cls}: WMAPE = {w:.1f}%')

    return df_eval


def error_analysis(df_eval: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Identifica SKUs con mayor error absoluto y diagnóstica causa probable.
    Returns DataFrame sku_errors ordenado por error_abs_total.
    """
    df_err = df_eval.copy()
    df_err['error_abs'] = (df_err['real'] - df_err['pred']).abs()
    df_err['sesgo'] = df_err['pred'] - df_err['real']

    sku_errors = df_err.groupby('codigo_articulo').agg(
        error_abs_total=('error_abs', 'sum'),
        sesgo_total=('sesgo', 'sum'),
        real_total=('real', 'sum'),
        pred_total=('pred', 'sum'),
        n_semanas=('real', 'count'),
        sb_class=('sb_class', 'first'),
        tipo_abc=('tipo_abc', 'first'),
    ).reset_index()

    sku_errors['wmape_sku'] = sku_errors['error_abs_total'] / sku_errors['real_total'].clip(lower=1) * 100
    sku_errors['sesgo_pct'] = sku_errors['sesgo_total'] / sku_errors['real_total'].clip(lower=1) * 100
    sku_errors = sku_errors.sort_values('error_abs_total', ascending=False)

    top = sku_errors.head(top_n)
    total_e = sku_errors['error_abs_total'].sum()
    top10_pct = sku_errors.head(10)['error_abs_total'].sum() / max(total_e, 1) * 100
    top50_pct = sku_errors.head(50)['error_abs_total'].sum() / max(total_e, 1) * 100

    print(f'\n    Top 10 SKUs concentran {top10_pct:.1f}% del error total')
    print(f'    Top 50 SKUs concentran {top50_pct:.1f}% del error total')

    for _, row in top.iterrows():
        direction = 'SUBPREDICE' if row['sesgo_total'] < 0 else 'SOBREPREDICE'
        print(f'    {row["codigo_articulo"]:>12s}  {row["sb_class"]:>13s}  '
              f'ABC:{row["tipo_abc"]}  Real:{row["real_total"]:>6,.0f}  '
              f'Pred:{row["pred_total"]:>6,.0f}  {direction} {abs(row["sesgo_pct"]):.0f}%')

    return sku_errors


def walk_forward(df_test_full: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward backtest: métricas por semana ISO para simular S&OP semanal.
    Returns DataFrame df_wf con WMAPE, bias, ratios de rotura/sobrestock por semana.
    """
    semanas = sorted(df_test_full['semana_anio'].unique())
    rows = []
    for sem in semanas:
        m = df_test_full['semana_anio'] == sem
        df_sem = df_test_full[m]
        y_real = df_sem['real'].values
        y_pred = df_sem['pred'].values
        total_real = y_real.sum()
        total_pred = y_pred.sum()
        wmape_sem = np.sum(np.abs(y_real - y_pred)) / max(np.sum(np.abs(y_real)), 1) * 100
        bias_pct = (total_pred - total_real) / max(total_real, 1) * 100
        n = len(df_sem)
        rows.append({
            'semana': sem,
            'wmape': wmape_sem,
            'bias_pct': bias_pct,
            'uds_reales': total_real,
            'uds_predichas': total_pred,
            'pct_rotura': (y_pred < y_real).sum() / n * 100,
            'pct_sobrestock': (y_pred > y_real).sum() / n * 100,
            'n_skus': n,
        })

    df_wf = pd.DataFrame(rows)
    print(f'\n    Walk-forward ({len(semanas)} semanas):')
    print(f'    WMAPE medio: {df_wf["wmape"].mean():.1f}% ± {df_wf["wmape"].std():.1f}%')
    print(f'    Bias medio:  {df_wf["bias_pct"].mean():+.1f}%')
    worst = df_wf.loc[df_wf['wmape'].idxmax(), 'semana']
    print(f'    Peor semana: S{worst:.0f} ({df_wf["wmape"].max():.1f}%)')

    return df_wf


def overfitting_check(model_R, model_clf, model_reg_h,
                      X_tr_R, y_tr_R, pred_R, y_te_R,
                      X_tr_H, y_tr_H, pred_H, y_te_H,
                      cat_idx_R, cat_idx_H_clean) -> pd.DataFrame:
    """
    Compara WMAPE en train vs test para detectar memorización.
    """
    from catboost import Pool

    pred_tr_R = np.expm1(model_R.predict(Pool(X_tr_R, cat_features=cat_idx_R))).clip(0)

    prob_tr_h = model_clf.predict_proba(Pool(X_tr_H, cat_features=cat_idx_H_clean))[:, 1]
    vol_tr_h = np.expm1(model_reg_h.predict(Pool(X_tr_H, cat_features=cat_idx_H_clean))).clip(0)
    pred_tr_H = (prob_tr_h * vol_tr_h).clip(0)

    rows = []
    for name, y_tr, p_tr, y_te, p_te in [
        ('Smooth/Erratic', y_tr_R.values, pred_tr_R, y_te_R.values, pred_R),
        ('Intermittent/Lumpy', y_tr_H.values, pred_tr_H, y_te_H.values, pred_H),
    ]:
        wmape_tr = np.sum(np.abs(y_tr - p_tr)) / max(np.sum(np.abs(y_tr)), 1) * 100
        wmape_te = np.sum(np.abs(y_te - p_te)) / max(np.sum(np.abs(y_te)), 1) * 100
        gap = wmape_te - wmape_tr
        estado = (
            'OPTIMO' if gap < 6.5 else
            ('MODERADO' if gap < 15 else 'ALERTA OVERFITTING')
        )
        rows.append({'Modelo': name, 'WMAPE_train': wmape_tr, 'WMAPE_test': wmape_te,
                     'Gap_pp': gap, 'Estado': estado})
        print(f'    {name}: train={wmape_tr:.1f}%  test={wmape_te:.1f}%  Δ={gap:+.1f}pp  → {estado}')

    return pd.DataFrame(rows)
