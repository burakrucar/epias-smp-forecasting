"""
T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - MATEMATİK BÖLÜMÜ BİTİRME TEZİ
MASTER DENEYSEL DOĞRULAMA VE RAPORLAMA BORU HATTI (V5)

Bu betik, hakem/danışman revizyonundaki 14 maddenin tamamını tek merkezden,
tek bir master tahmin parquet dosyasından türeterek ve sıkı assert'lerle üretir:
- run_id: benzersiz çalışma kimliği
- test_predictions_master.parquet: tüm modellerin ve bazların saatlik tahminleri
- Tablo 1'den Tablo 11'e kadar tüm tablolar bu parquet'den türetilir
- Tüm assert'ler programatik olarak kontrol edilir
"""

import os
import sys
import hashlib
import time
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    brier_score_loss
)
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 0. YARDIMCI İSTATİSTİK FONKSİYONLARI
# -------------------------------------------------------------
def calculate_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def newey_west_daily_dm(y_true, y_pred_model, y_pred_bench, timestamps, loss="mae", max_lags=7):
    """
    Günlük ortalama kayıplar üzerinde Newey-West (HAC) standart hatalı Diebold-Mariano testi.
    d_t = L(Model) - L(Benchmark)
    Negatif d_t: Model daha düşük kayba sahip (Model daha iyi).
    """
    df_eval = pd.DataFrame({
        "timestamp": timestamps,
        "y_true": y_true,
        "pred_m": y_pred_model,
        "pred_b": y_pred_bench
    })
    df_eval["date"] = df_eval["timestamp"].dt.date
    
    if loss == "mae":
        df_eval["loss_m"] = np.abs(df_eval["y_true"] - df_eval["pred_m"])
        df_eval["loss_b"] = np.abs(df_eval["y_true"] - df_eval["pred_b"])
    else:
        df_eval["loss_m"] = (df_eval["y_true"] - df_eval["pred_m"])**2
        df_eval["loss_b"] = (df_eval["y_true"] - df_eval["pred_b"])**2
        
    daily = df_eval.groupby("date")[["loss_m", "loss_b"]].mean()
    # Yön: Model - Benchmark (Negatif = Model daha iyi)
    d = (daily["loss_m"] - daily["loss_b"]).values
    n = len(d)
    
    X = np.ones((n, 1))
    ols_res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    t_stat = float(ols_res.tvalues[0])
    p_two_sided = float(ols_res.pvalues[0])
    return t_stat, p_two_sided

def block_bootstrap_diff(metric_func, y_true, y_pred_model, y_pred_bench, timestamps, n_boot=1000, block_size=7, seed=42):
    """
    7 günlük blok bootstrap ile iki modelin başarı farkı için %95 güven aralığı.
    Diff = Metric(Model) - Metric(Benchmark)
    """
    rng = np.random.RandomState(seed)
    df_eval = pd.DataFrame({
        "timestamp": timestamps,
        "y_true": y_true,
        "pred_m": y_pred_model,
        "pred_b": y_pred_bench
    })
    df_eval["date"] = df_eval["timestamp"].dt.date
    unique_dates = df_eval["date"].unique()
    n_days = len(unique_dates)
    n_blocks = int(np.ceil(n_days / block_size))
    
    date_to_indices = df_eval.groupby("date").indices
    base_diff = metric_func(y_true, y_pred_model) - metric_func(y_true, y_pred_bench)
    diffs = []
    
    for _ in range(n_boot):
        start_indices = rng.randint(0, n_days - block_size + 1, size=n_blocks)
        sampled_dates = []
        for idx in start_indices:
            sampled_dates.extend(unique_dates[idx:idx + block_size])
        sampled_dates = sampled_dates[:n_days]
        
        sampled_row_indices = np.concatenate([date_to_indices[d] for d in sampled_dates])
        yt_s = y_true[sampled_row_indices]
        pm_s = y_pred_model[sampled_row_indices]
        pb_s = y_pred_bench[sampled_row_indices]
        
        diff = metric_func(yt_s, pm_s) - metric_func(yt_s, pb_s)
        diffs.append(diff)
        
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    return float(base_diff), ci_low, ci_high

def mcnemar_test(y_true, y_pred_model, y_pred_bench):
    corr_m = (y_pred_model == y_true)
    corr_b = (y_pred_bench == y_true)
    b = np.sum(corr_m & ~corr_b)
    c = np.sum(~corr_m & corr_b)
    if (b + c) == 0:
        return 0.0, 1.0
    chi2 = float(((abs(b - c) - 1.0)**2) / (b + c))
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))
    return chi2, p_val


# -------------------------------------------------------------
# 1. ÖZELLİK MATRİSİ ÇIKARIMI
# -------------------------------------------------------------
def build_feature_dataframe(df_raw, drop_initial_lags=True):
    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    
    # Causal forward fill
    df["load_forecast_mw"] = df["load_forecast_mw"].ffill()
    df["wind_forecast_mw"] = df["wind_forecast_mw"].ffill()

    hours = df["timestamp"].dt.hour.values
    dows = df["timestamp"].dt.dayofweek.values
    months = df["timestamp"].dt.month.values

    df["hour"] = hours
    df["dow"] = dows
    df["is_weekend"] = (dows >= 5).astype(int)

    # Saat one-hot kodlaması
    for h in range(24):
        df[f"hour_{h:02d}"] = (hours == h).astype(int)

    # Çeyreklik mevsimsellik
    df["quarter_sin"] = np.sin(2 * np.pi * months / 12.0)
    df["quarter_cos"] = np.cos(2 * np.pi * months / 12.0)

    # Kalan yük tahmini ve rüzgar oranı
    df["forecast_residual_load_mw"] = df["load_forecast_mw"] - df["wind_forecast_mw"]
    df["forecast_wind_ratio"] = df["wind_forecast_mw"] / np.maximum(df["load_forecast_mw"], 1.0)

    # Yön hedefleri
    df["target_direction_bin"] = (df["systemStatus"] == "Enerji Açığı").astype(int)
    status_3map = {"Enerji Açığı": 1, "Enerji Fazlası": -1, "Dengede": 0}
    df["target_direction_3c"] = df["systemStatus"].map(status_3map).values

    # PTF gecikmeleri
    df["ptf_lag_24"] = df["ptf"].shift(24)
    df["ptf_lag_48"] = df["ptf"].shift(48)
    df["ptf_lag_168"] = df["ptf"].shift(168)

    # SMF gecikmeleri
    latest_known_smf = np.full(n, np.nan)
    legit_same_hour_smf = np.full(n, np.nan)
    legit_same_hour_dir = np.full(n, np.nan)

    for i in range(n):
        h = hours[i]
        lag_latest = 15 + h
        if i >= lag_latest:
            latest_known_smf[i] = df["smf"].iloc[i - lag_latest]
        
        if h <= 9:
            if i >= 24:
                legit_same_hour_smf[i] = df["smf"].iloc[i - 24]
                legit_same_hour_dir[i] = df["target_direction_bin"].iloc[i - 24]
        else:
            if i >= 48:
                legit_same_hour_smf[i] = df["smf"].iloc[i - 48]
                legit_same_hour_dir[i] = df["target_direction_bin"].iloc[i - 48]

    df["latest_known_smf"] = latest_known_smf
    df["legit_same_hour_smf"] = legit_same_hour_smf
    df["legit_same_hour_dir"] = legit_same_hour_dir

    df["smf_lag_48"] = df["smf"].shift(48)
    df["smf_lag_72"] = df["smf"].shift(72)
    df["smf_lag_168"] = df["smf"].shift(168)
    df["smf_lag_336"] = df["smf"].shift(336)
    df["smf_lag_504"] = df["smf"].shift(504)
    df["smf_lag_672"] = df["smf"].shift(672)

    # Son 4 haftanın aynı saat medyanı (lags 168, 336, 504, 672)
    lags_4w = [df["smf_lag_168"], df["smf_lag_336"], df["smf_lag_504"], df["smf_lag_672"]]
    df["smf_median_4w"] = pd.concat(lags_4w, axis=1).median(axis=1)

    # 6 günün aynı saat ortalaması (D-2'den D-7'ye: 48, 72, 96, 120, 144, 168)
    clean_7d_lags = [df["smf"].shift(l) for l in [48, 72, 96, 120, 144, 168]]
    df["smf_clean_same_hour_7d_mean"] = pd.concat(clean_7d_lags, axis=1).mean(axis=1)

    # Son 4 haftanın aynı saat açık oranı baz modeli (Yön için)
    dir_lags_4w = [
        df["target_direction_bin"].shift(168),
        df["target_direction_bin"].shift(336),
        df["target_direction_bin"].shift(504),
        df["target_direction_bin"].shift(672)
    ]
    df["dir_open_ratio_4w"] = pd.concat(dir_lags_4w, axis=1).mean(axis=1)
    df["dir_pred_4w_maj"] = (df["dir_open_ratio_4w"] >= 0.5).astype(int)

    # Hedefler
    df["target_smf"] = df["smf"].values
    df["target_spread"] = df["smf"].values - df["ptf"].values

    if drop_initial_lags:
        # lag 672 sonrası temiz veri
        df_clean = df.dropna().reset_index(drop=True)
        return df_clean
    return df


def execute_master_pipeline():
    run_id = f"run_{int(time.time())}"
    print("=" * 110)
    print(f"  T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - MATEMATİK BÖLÜMÜ BİTİRME TEZİ")
    print(f"  MASTER DENEY VE DOĞRULAMA MOTORU (V5) | RUN ID: {run_id}")
    print("=" * 110)

    tables_dir = "reports/tables"
    figures_dir = "reports/figures"
    processed_dir = "data/processed"
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    data_path = "data/raw/epias_official_market_data_full.parquet"
    df_raw = pd.read_parquet(data_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    df_clean = build_feature_dataframe(df_raw, drop_initial_lags=True)
    
    # -------------------------------------------------------------
    # ZAMAN BÖLÜMLERİ
    # -------------------------------------------------------------
    train_mask = (df_clean["timestamp"] < "2024-10-01").values
    val_mask = ((df_clean["timestamp"] >= "2024-10-01") & (df_clean["timestamp"] < "2025-10-01")).values
    test_mask = (df_clean["timestamp"] >= "2025-10-01").values

    train_df = df_clean[train_mask].reset_index(drop=True)
    val_df = df_clean[val_mask].reset_index(drop=True)
    full_train_df = df_clean[train_mask | val_mask].reset_index(drop=True)
    test_df = df_clean[test_mask].reset_index(drop=True)

    hour_cols = [f"hour_{h:02d}" for h in range(24)]
    # Ridge için seçilen özellikler (VIF bağlantısını gidermek için residual load çıkarıldı)
    ridge_features = (
        hour_cols + ["is_weekend", "quarter_sin", "quarter_cos",
         "load_forecast_mw", "forecast_wind_ratio",
         "ptf_lag_24", "ptf_lag_48", "ptf_lag_168",
         "latest_known_smf", "legit_same_hour_smf",
         "smf_lag_48", "smf_lag_72", "smf_lag_168", "smf_median_4w", "smf_clean_same_hour_7d_mean"]
    )
    ridge_features_post = ridge_features + ["ptf"]
    ridge_ablation_features = [c for c in ridge_features if not c.startswith("load_") and not c.startswith("forecast_")]

    # -------------------------------------------------------------
    # VALİDASYON AYARLARI VE MODEL SEÇİMİ
    # -------------------------------------------------------------
    best_alpha = 1000.0
    best_lgb_params = {"max_depth": 4, "learning_rate": 0.05, "min_child_samples": 40}

    # -------------------------------------------------------------
    # NİHAİ TEST TAHMİNLERİNİN ÜRETİLMESİ (MASTER DATAFRAME)
    # -------------------------------------------------------------
    print("[1/5] Master Test Tahminleri Üretiliyor...")
    scaler_final = StandardScaler()
    X_full_train = scaler_final.fit_transform(full_train_df[ridge_features])
    X_test = scaler_final.transform(test_df[ridge_features])

    y_train_smf = full_train_df["target_smf"].values
    y_test_smf = test_df["target_smf"].values
    ptf_test = test_df["ptf"].values
    y_test_dir_bin = test_df["target_direction_bin"].values
    y_test_dir_3c = test_df["target_direction_3c"].values
    timestamps_test = test_df["timestamp"]

    # 1. Sabit Ridge
    ridge_fixed = Ridge(alpha=best_alpha, random_state=42).fit(X_full_train, y_train_smf)
    pred_ridge_fixed = ridge_fixed.predict(X_test)

    # 2. Genişleyen Pencereli Ridge (Aylık Refit)
    test_months = sorted(test_df["timestamp"].dt.to_period("M").unique())
    pred_ridge_exp = np.zeros(len(test_df))
    curr_train = full_train_df.copy()
    for m in test_months:
        idx_m = (test_df["timestamp"].dt.to_period("M") == m).values
        sub_te = test_df[idx_m]
        sc_m = StandardScaler()
        X_tr_m = sc_m.fit_transform(curr_train[ridge_features])
        X_te_m = sc_m.transform(sub_te[ridge_features])
        r_m = Ridge(alpha=best_alpha, random_state=42).fit(X_tr_m, curr_train["target_smf"])
        pred_ridge_exp[idx_m] = r_m.predict(X_te_m)
        curr_train = pd.concat([curr_train, sub_te], axis=0).reset_index(drop=True)

    # 3. Ridge Ablasyonu (Yük ve rüzgarsız)
    sc_abl = StandardScaler()
    X_tr_abl = sc_abl.fit_transform(full_train_df[ridge_ablation_features])
    X_te_abl = sc_abl.transform(test_df[ridge_ablation_features])
    pred_ridge_ablation = Ridge(alpha=best_alpha, random_state=42).fit(X_tr_abl, y_train_smf).predict(X_te_abl)

    # 4. Sabit LightGBM Seviye
    lgb_pre_fixed = lgb.LGBMRegressor(n_estimators=250, random_state=42, verbose=-1, **best_lgb_params)
    lgb_pre_fixed.fit(full_train_df[ridge_features], y_train_smf)
    pred_lgb_pre_fixed = lgb_pre_fixed.predict(test_df[ridge_features])

    # 5. Genişleyen Pencereli LightGBM Seviye
    pred_lgb_pre_exp = np.zeros(len(test_df))
    curr_train_lgb = full_train_df.copy()
    for m in test_months:
        idx_m = (test_df["timestamp"].dt.to_period("M") == m).values
        sub_te = test_df[idx_m]
        m_lgb = lgb.LGBMRegressor(n_estimators=250, random_state=42, verbose=-1, **best_lgb_params)
        m_lgb.fit(curr_train_lgb[ridge_features], curr_train_lgb["target_smf"])
        pred_lgb_pre_exp[idx_m] = m_lgb.predict(sub_te[ridge_features])
        curr_train_lgb = pd.concat([curr_train_lgb, sub_te], axis=0).reset_index(drop=True)

    # 6. GÖP Sonrası LightGBM Yayılım Modeli (PTF + Spread)
    lgb_spread_fixed = lgb.LGBMRegressor(n_estimators=250, random_state=42, verbose=-1, **best_lgb_params)
    lgb_spread_fixed.fit(full_train_df[ridge_features_post], full_train_df["target_spread"])
    pred_lgb_spread = lgb_spread_fixed.predict(test_df[ridge_features_post])
    pred_lgb_post_fixed = ptf_test + pred_lgb_spread

    # 7. Yön Modelleri ve Bazları
    # Eğitimden türetilen saatlik baz:
    train_hourly_open_prob = full_train_df.groupby("hour")["target_direction_bin"].mean().to_dict()
    pred_bench_hourly_maj = test_df["hour"].map(lambda h: int(train_hourly_open_prob[h] >= 0.5)).values
    
    # Eğitimden türetilen hafta sonu x saat bazı:
    train_dow_hour_open_prob = full_train_df.groupby(["is_weekend", "hour"])["target_direction_bin"].mean().to_dict()
    pred_bench_dow_hour_maj = test_df.apply(lambda r: int(train_dow_hour_open_prob.get((r["is_weekend"], r["hour"]), 0.5) >= 0.5), axis=1).values
    clim_probs_dow_hour = test_df.apply(lambda r: train_dow_hour_open_prob.get((r["is_weekend"], r["hour"]), 0.5), axis=1).values

    # Son 4 haftanın aynı saat açık oranı bazı (D-1 12:00'de bilinen):
    pred_bench_4w_maj = test_df["dir_pred_4w_maj"].values
    prob_bench_4w = test_df["dir_open_ratio_4w"].values

    # Lojistik Regresyon (Validasyon eşiği tau=0.45):
    logreg_final = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logreg_final.fit(X_full_train, full_train_df["target_direction_bin"])
    prob_logreg_test = logreg_final.predict_proba(X_test)[:, 1]
    pred_logreg_test = (prob_logreg_test >= 0.45).astype(int)

    # LightGBM Classifier (Validasyon eşiği tau=0.50):
    lgb_clf_final = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=40, class_weight="balanced", random_state=42, verbose=-1)
    lgb_clf_final.fit(full_train_df[ridge_features], full_train_df["target_direction_bin"])
    prob_lgb_test = lgb_clf_final.predict_proba(test_df[ridge_features])[:, 1]
    pred_lgb_test = (prob_lgb_test >= 0.50).astype(int)

    # 3 Sınıflı Model ve Bazları
    train_3c_hourly_mode = full_train_df.groupby("hour")["target_direction_3c"].apply(lambda s: s.mode()[0]).to_dict()
    train_3c_dow_hour_mode = full_train_df.groupby(["is_weekend", "hour"])["target_direction_3c"].apply(lambda s: s.mode()[0]).to_dict()
    pred_3c_hourly_maj = test_df["hour"].map(train_3c_hourly_mode).values
    pred_3c_dow_hour_maj = test_df.apply(lambda r: train_3c_dow_hour_mode.get((r["is_weekend"], r["hour"]), 1), axis=1).values
    
    map3 = {-1: 0, 0: 1, 1: 2}
    inv3 = {0: -1, 1: 0, 2: 1}
    lgb_3c = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=40, random_state=42, verbose=-1)
    lgb_3c.fit(full_train_df[ridge_features], full_train_df["target_direction_3c"].map(map3))
    preds_3c_raw = lgb_3c.predict(test_df[ridge_features])
    preds_3c_lgb = np.array([inv3[v] for v in preds_3c_raw])

    # -------------------------------------------------------------
    # MASTER TAHMİN DATAFRAME'İ (TEK KAYNAK)
    # -------------------------------------------------------------
    master_df = pd.DataFrame({
        "run_id": run_id,
        "timestamp": timestamps_test,
        "hour": test_df["hour"].values,
        "dow": test_df["dow"].values,
        "is_weekend": test_df["is_weekend"].values,
        "y_true_smf": y_test_smf,
        "y_true_dir_bin": y_test_dir_bin,
        "y_true_dir_3c": y_test_dir_3c,
        "ptf": ptf_test,
        "latest_known_smf": test_df["latest_known_smf"].values,
        "bench_smf_median_4w": test_df["smf_median_4w"].values,
        "bench_smf_lag_168": test_df["smf_lag_168"].values,
        "bench_ptf_lag_24": test_df["ptf_lag_24"].values,
        "bench_legit_same_hour_smf": test_df["legit_same_hour_smf"].values,
        "bench_smf_eq_ptf": ptf_test.copy(),
        "pred_ridge_fixed": pred_ridge_fixed,
        "pred_ridge_exp": pred_ridge_exp,
        "pred_ridge_ablation": pred_ridge_ablation,
        "pred_lgb_pre_fixed": pred_lgb_pre_fixed,
        "pred_lgb_pre_exp": pred_lgb_pre_exp,
        "pred_lgb_post_fixed": pred_lgb_post_fixed,
        "bench_dir_flat_maj": np.ones_like(y_test_dir_bin),
        "bench_dir_hourly_maj": pred_bench_hourly_maj,
        "bench_dir_dow_hour_maj": pred_bench_dow_hour_maj,
        "bench_dir_4w_maj": pred_bench_4w_maj,
        "prob_bench_dow_hour": clim_probs_dow_hour,
        "prob_bench_4w": prob_bench_4w,
        "pred_logreg": pred_logreg_test,
        "prob_logreg": prob_logreg_test,
        "pred_lgb_dir": pred_lgb_test,
        "prob_lgb_dir": prob_lgb_test,
        "bench_3c_hourly_maj": pred_3c_hourly_maj,
        "bench_3c_dow_hour_maj": pred_3c_dow_hour_maj,
        "pred_3c_lgb": preds_3c_lgb
    })
    
    master_parquet_path = os.path.join(processed_dir, "test_predictions_master.parquet")
    master_df.to_parquet(master_parquet_path)
    print(f"       Master Tahmin Dosyası Kaydedildi: {master_parquet_path} ({len(master_df)} satır)")

    # -------------------------------------------------------------
    # TÜM TABLOLAR MASTER DATAFRAME'DEN DİNAMİK OLARAK TÜRETİLİR
    # -------------------------------------------------------------
    print("\n[2/5] Tabloların Master Dosyadan Üretilmesi ve Assert Denetimi...")
    
    mean_test_smf = float(np.mean(master_df["y_true_smf"]))
    official_base_name = "Son 4 Haftanın Medyan SMF'si (smf_median_4w)"
    official_base_preds = master_df["bench_smf_median_4w"].values

    # TABLO 1: FİYAT TAHMİNİ KARŞILAŞTIRMA TABLOSU
    price_cols_map = [
        ("1. Resmi Referans Baz: Son 4 Haftanın Medyan SMF'si", "bench_smf_median_4w", "Meşru Baz Model"),
        ("2. Haftalık Aynı Saat Bazı (Lag 168)", "bench_smf_lag_168", "Uzak Baz Model"),
        ("3. Dünkü Aynı Saat PTF (ptf_lag_24)", "bench_ptf_lag_24", "Dünkü PTF Bazı"),
        ("4. Meşru Dinamik Aynı Saat Bazı", "bench_legit_same_hour_smf", "Dinamik Baz Model"),
        ("5. GÖP Sonrası Doğal Kural: SMF = PTF", "bench_smf_eq_ptf", "GÖP Sonrası Doğal Baz"),
        ("6. GÖP Öncesi: Ridge Regresyon (Sabit Model)", "pred_ridge_fixed", "İstatistiksel"),
        ("7. GÖP Öncesi: Ridge Regresyon (Genişleyen Pencere)", "pred_ridge_exp", "İstatistiksel (Aylık Refit)"),
        ("8. GÖP Öncesi: Ridge Ablasyonu (Yük ve Rüzgarsız)", "pred_ridge_ablation", "İstatistiksel Ablasyon"),
        ("9. GÖP Öncesi: LightGBM Seviye (Sabit Model)", "pred_lgb_pre_fixed", "Makine Öğrenmesi"),
        ("10. GÖP Öncesi: LightGBM Seviye (Genişleyen Pencere)", "pred_lgb_pre_exp", "Makine Öğrenmesi (Aylık Refit)"),
        ("11. GÖP Sonrası: LightGBM Yayılım Modeli (PTF + Spread)", "pred_lgb_post_fixed", "Makine Öğrenmesi")
    ]

    tablo1_rows = []
    for label, col, fam in price_cols_map:
        preds = master_df[col].values
        mae = float(mean_absolute_error(master_df["y_true_smf"], preds))
        rmse = float(root_mean_squared_error(master_df["y_true_smf"], preds))
        nmae = (mae / mean_test_smf) * 100.0
        smape = calculate_smape(master_df["y_true_smf"].values, preds)
        
        if col == "bench_smf_median_4w":
            diff_mae_val = 0.0
            ci_str = "-"
            t_stat = 0.0
            p_two_str = "-"
        else:
            # Model - Benchmark yönü (Negatif = Model daha iyi / daha düşük MAE)
            diff_mae_val = mae - mean_absolute_error(master_df["y_true_smf"], official_base_preds)
            _, ci_l, ci_h = block_bootstrap_diff(mean_absolute_error, master_df["y_true_smf"].values, preds, official_base_preds, timestamps_test)
            ci_str = f"[{ci_l:+.2f}, {ci_high_val:+.2f}]" if 'ci_high_val' in locals() else f"[{ci_l:+.2f}, {ci_h:+.2f}]"
            t_stat, p_two = newey_west_daily_dm(master_df["y_true_smf"].values, preds, official_base_preds, timestamps_test, loss="mae")
            p_two_str = "<0.0001" if p_two < 0.0001 else f"{p_two:.4f}"

        tablo1_rows.append({
            "run_id": run_id,
            "Model Adı": label,
            "Model Ailesi": fam,
            "MAE (TL/MWh)": round(mae, 2),
            "RMSE (TL/MWh)": round(rmse, 2),
            "NMAE (%)": round(nmae, 2),
            "sMAPE (%)": round(smape, 2),
            "Resmi Baza Göre MAE Farkı (TL)": round(diff_mae_val, 2),
            "95% Blok Bootstrap GA (TL)": ci_str,
            "Günlük HAC DM t-stat": round(t_stat, 3),
            "Günlük HAC DM İki Taraflı p": p_two_str
        })
    df_tablo1 = pd.DataFrame(tablo1_rows)
    df_tablo1.to_csv(os.path.join(tables_dir, "tablo1_fiyat_tahmini_karsilastirma.csv"), index=False)

    # TABLO 2: ÖZEL ÇİFTLER İÇİN İKİ YÖNLÜ HAC DM TESTLERİ
    special_pairs_map = [
        ("Ridge (Sabit) vs Resmi Baz (smf_median_4w)", "pred_ridge_fixed", "bench_smf_median_4w"),
        ("Ridge (Genişleyen) vs Resmi Baz (smf_median_4w)", "pred_ridge_exp", "bench_smf_median_4w"),
        ("Ridge (Sabit) vs Lag 168", "pred_ridge_fixed", "bench_smf_lag_168"),
        ("Ridge (Genişleyen) vs ptf_lag_24", "pred_ridge_exp", "bench_ptf_lag_24"),
        ("Ridge (Genişleyen) vs LightGBM (Genişleyen)", "pred_ridge_exp", "pred_lgb_pre_exp"),
        ("LightGBM Yayılım vs SMF = PTF", "pred_lgb_post_fixed", "bench_smf_eq_ptf")
    ]
    tablo2_rows = []
    for pair_name, c1, c2 in special_pairs_map:
        m1 = master_df[c1].values
        m2 = master_df[c2].values
        yt = master_df["y_true_smf"].values
        
        m1_mae = float(mean_absolute_error(yt, m1))
        m2_mae = float(mean_absolute_error(yt, m2))
        m1_rmse = float(root_mean_squared_error(yt, m1))
        m2_rmse = float(root_mean_squared_error(yt, m2))
        
        t_mae, p_mae = newey_west_daily_dm(yt, m1, m2, timestamps_test, loss="mae")
        t_rmse, p_rmse = newey_west_daily_dm(yt, m1, m2, timestamps_test, loss="mse")
        
        tablo2_rows.append({
            "run_id": run_id,
            "Karşılaştırma Çifti (Model 1 vs Model 2)": pair_name,
            "Model 1 MAE": round(m1_mae, 2),
            "Model 2 MAE": round(m2_mae, 2),
            "MAE Farkı (M1 - M2)": round(m1_mae - m2_mae, 2),
            "DM t-stat (MAE)": round(t_mae, 3),
            "DM İki Taraflı p (MAE)": "<0.0001" if p_mae < 0.0001 else f"{p_mae:.4f}",
            "Model 1 RMSE": round(m1_rmse, 2),
            "Model 2 RMSE": round(m2_rmse, 2),
            "RMSE Farkı (M1 - M2)": round(m1_rmse - m2_rmse, 2),
            "DM t-stat (RMSE)": round(t_rmse, 3),
            "DM İki Taraflı p (RMSE)": "<0.0001" if p_rmse < 0.0001 else f"{p_rmse:.4f}"
        })
    df_tablo2 = pd.DataFrame(tablo2_rows)
    df_tablo2.to_csv(os.path.join(tables_dir, "tablo2_ozel_ciftler_diebold_mariano.csv"), index=False)

    # TABLO 3: 12 AYLIK HATA VE BIAS DAĞILIMI (Dinamik olarak master_df'den türetildi)
    master_df["month_str"] = master_df["timestamp"].dt.strftime("%Y-%m")
    tablo3_rows = []
    
    ridge_beat_official_months = 0
    ridge_beat_ptf24_months = 0
    
    for m_str, g in master_df.groupby("month_str"):
        y_m = g["y_true_smf"].values
        p_r_fixed = g["pred_ridge_fixed"].values
        p_r_exp = g["pred_ridge_exp"].values
        p_b_off = g["bench_smf_median_4w"].values
        p_b_ptf = g["bench_ptf_lag_24"].values
        p_l_exp = g["pred_lgb_pre_exp"].values
        
        r_exp_mae = float(mean_absolute_error(y_m, p_r_exp))
        off_mae = float(mean_absolute_error(y_m, p_b_off))
        ptf_mae = float(mean_absolute_error(y_m, p_b_ptf))
        
        if r_exp_mae < off_mae:
            ridge_beat_official_months += 1
        if r_exp_mae < ptf_mae:
            ridge_beat_ptf24_months += 1
            
        tablo3_rows.append({
            "run_id": run_id,
            "Ay": m_str,
            "Gözlem": len(g),
            "Ortalama SMF (TL)": round(float(np.mean(y_m)), 2),
            "Ridge Sabit ME/Bias (TL)": round(float(np.mean(p_r_fixed - y_m)), 2),
            "Ridge Sabit MAE (TL)": round(float(mean_absolute_error(y_m, p_r_fixed)), 2),
            "Ridge Genişleyen MAE (TL)": round(r_exp_mae, 2),
            "Resmi Baz (Medyan 4w) MAE (TL)": round(off_mae, 2),
            "Dünkü PTF (ptf_lag_24) MAE (TL)": round(ptf_mae, 2),
            "LightGBM Genişleyen MAE (TL)": round(float(mean_absolute_error(y_m, p_l_exp)), 2),
            "Ridge(Genişleyen) - Resmi Baz Farkı (TL)": round(r_exp_mae - off_mae, 2)
        })
    df_tablo3 = pd.DataFrame(tablo3_rows)
    df_tablo3.to_csv(os.path.join(tables_dir, "tablo3_aylik_hata_ve_bias.csv"), index=False)

    # TABLO 4: 24 SAATİN TAMAMI İÇİN SAATLİK MAE DAĞILIMI
    tablo4_rows = []
    ridge_beat_official_hours = 0
    ridge_beat_ptf24_hours = 0
    
    for h in range(24):
        g = master_df[master_df["hour"] == h]
        y_h = g["y_true_smf"].values
        p_r_fixed = g["pred_ridge_fixed"].values
        p_r_exp = g["pred_ridge_exp"].values
        p_b_off = g["bench_smf_median_4w"].values
        p_b_ptf = g["bench_ptf_lag_24"].values
        
        r_exp_h_mae = float(mean_absolute_error(y_h, p_r_exp))
        off_h_mae = float(mean_absolute_error(y_h, p_b_off))
        ptf_h_mae = float(mean_absolute_error(y_h, p_b_ptf))
        
        if r_exp_h_mae < off_h_mae:
            ridge_beat_official_hours += 1
        if r_exp_h_mae < ptf_h_mae:
            ridge_beat_ptf24_hours += 1
            
        tablo4_rows.append({
            "run_id": run_id,
            "Saat": f"{h:02d}:00",
            "Ortalama SMF (TL)": round(float(np.mean(y_h)), 2),
            "Ridge Sabit MAE (TL)": round(float(mean_absolute_error(y_h, p_r_fixed)), 2),
            "Ridge Genişleyen MAE (TL)": round(r_exp_h_mae, 2),
            "Resmi Baz (Medyan 4w) MAE (TL)": round(off_h_mae, 2),
            "Dünkü PTF (ptf_lag_24) MAE (TL)": round(ptf_h_mae, 2),
            "Ridge(Genişleyen) - Resmi Baz Farkı (TL)": round(r_exp_h_mae - off_h_mae, 2)
        })
    df_tablo4 = pd.DataFrame(tablo4_rows)
    df_tablo4.to_csv(os.path.join(tables_dir, "tablo4_saatlik_24_saat_mae.csv"), index=False)

    # -------------------------------------------------------------
    # PROGRAMATİK ASSERT DENETİMLERİ (GEREKSİNİM 1, 3, 9, 7)
    # -------------------------------------------------------------
    # 1. Tablo 1 vs Tablo 3 vs Tablo 4 Ridge Sabit Tutarlılığı:
    tablo1_ridge_fixed_mae = float(df_tablo1.loc[df_tablo1["Model Adı"].str.contains("Sabit Model"), "MAE (TL/MWh)"].values[0])
    tablo3_weighted_fixed_mae = float(np.average(df_tablo3["Ridge Sabit MAE (TL)"], weights=df_tablo3["Gözlem"]))
    tablo4_mean_fixed_mae = float(np.mean(df_tablo4["Ridge Sabit MAE (TL)"]))
    
    assert abs(tablo1_ridge_fixed_mae - tablo3_weighted_fixed_mae) < 0.05, f"Tablo 1 ve Tablo 3 Ridge Sabit uyuşmazlığı: {tablo1_ridge_fixed_mae} vs {tablo3_weighted_fixed_mae}"
    assert abs(tablo1_ridge_fixed_mae - tablo4_mean_fixed_mae) < 0.05, f"Tablo 1 ve Tablo 4 Ridge Sabit uyuşmazlığı: {tablo1_ridge_fixed_mae} vs {tablo4_mean_fixed_mae}"

    # 2. Tablo 1 vs Tablo 3 vs Tablo 4 Ridge Genişleyen Tutarlılığı:
    tablo1_ridge_exp_mae = float(df_tablo1.loc[df_tablo1["Model Adı"].str.contains("Genişleyen Pencere"), "MAE (TL/MWh)"].values[0])
    tablo3_weighted_exp_mae = float(np.average(df_tablo3["Ridge Genişleyen MAE (TL)"], weights=df_tablo3["Gözlem"]))
    tablo4_mean_exp_mae = float(np.mean(df_tablo4["Ridge Genişleyen MAE (TL)"]))
    assert abs(tablo1_ridge_exp_mae - tablo3_weighted_exp_mae) < 0.05, f"Tablo 1 ve Tablo 3 Ridge Genişleyen uyuşmazlığı: {tablo1_ridge_exp_mae} vs {tablo3_weighted_exp_mae}"
    assert abs(tablo1_ridge_exp_mae - tablo4_mean_exp_mae) < 0.05, f"Tablo 1 ve Tablo 4 Ridge Genişleyen uyuşmazlığı: {tablo1_ridge_exp_mae} vs {tablo4_mean_exp_mae}"

    # 3. Tablo 1 vs Tablo 3 vs Tablo 4 Resmi Baz Tutarlılığı:
    tablo1_base_mae = float(df_tablo1.loc[df_tablo1["Model Adı"].str.contains("Resmi Referans Baz"), "MAE (TL/MWh)"].values[0])
    tablo3_weighted_base_mae = float(np.average(df_tablo3["Resmi Baz (Medyan 4w) MAE (TL)"], weights=df_tablo3["Gözlem"]))
    tablo4_mean_base_mae = float(np.mean(df_tablo4["Resmi Baz (Medyan 4w) MAE (TL)"]))
    assert abs(tablo1_base_mae - tablo3_weighted_base_mae) < 0.05, f"Tablo 1 ve Tablo 3 Baz uyuşmazlığı: {tablo1_base_mae} vs {tablo3_weighted_base_mae}"
    assert abs(tablo1_base_mae - tablo4_mean_base_mae) < 0.05, f"Tablo 1 ve Tablo 4 Baz uyuşmazlığı: {tablo1_base_mae} vs {tablo4_mean_base_mae}"
    print("       Assert 1 Başarılı: Tablo 1, Tablo 3 ve Tablo 4 fiyat tahmin metrikleri birebir tutarlıdır.")

    # TABLO 5: YÖN SINIFLANDIRMA KARŞILAŞTIRMA TABLOSU
    dir_official_preds = master_df["bench_dir_dow_hour_maj"].values
    brier_ref_dow = float(brier_score_loss(master_df["y_true_dir_bin"], master_df["prob_bench_dow_hour"]))
    brier_ref_4w = float(brier_score_loss(master_df["y_true_dir_bin"], master_df["prob_bench_4w"]))

    dir_models_map = [
        ("1. Resmi Referans Baz: Hafta İçi/Sonu x Saat Çoğunluk", "bench_dir_dow_hour_maj", "prob_bench_dow_hour"),
        ("2. Saat Bazlı Çoğunluk Bazı (Eğitimden)", "bench_dir_hourly_maj", None),
        ("3. Son 4 Haftanın Aynı Saat Açık Oranı Bazı", "bench_dir_4w_maj", "prob_bench_4w"),
        ("4. Genel Çoğunluk Sınıfı (Her Zaman Açık)", "bench_dir_flat_maj", None),
        ("5. Lojistik Regresyon (Validasyon Eşikli)", "pred_logreg", "prob_logreg"),
        ("6. LightGBM Sınıflandırıcı (Val Eşikli)", "pred_lgb_dir", "prob_lgb_dir")
    ]
    tablo5_rows = []
    for label, col_p, col_prob in dir_models_map:
        p_bin = master_df[col_p].values
        prob = master_df[col_prob].values if col_prob is not None else None
        
        acc = float(accuracy_score(master_df["y_true_dir_bin"], p_bin))
        bal_acc = float(balanced_accuracy_score(master_df["y_true_dir_bin"], p_bin))
        f1 = float(f1_score(master_df["y_true_dir_bin"], p_bin))
        mcc = float(matthews_corrcoef(master_df["y_true_dir_bin"], p_bin))
        auc_str = f"{roc_auc_score(master_df['y_true_dir_bin'], prob):.4f}" if prob is not None else "-"
        
        if prob is not None:
            bs = float(brier_score_loss(master_df["y_true_dir_bin"], prob))
            bss_dow = (1.0 - (bs / brier_ref_dow)) * 100.0
            bss_4w = (1.0 - (bs / brier_ref_4w)) * 100.0
            bs_str = f"{bs:.4f}"
            bss_dow_str = f"%{bss_dow:.2f}"
            bss_4w_str = f"%{bss_4w:.2f}"
        else:
            bs_str = "-"
            bss_dow_str = "-"
            bss_4w_str = "-"
            
        if col_p == "bench_dir_dow_hour_maj":
            acc_ci_str = "-"
            mcc_ci_str = "-"
            mcn_str = "Referans Baz"
        else:
            diff_acc, a_l, a_h = block_bootstrap_diff(accuracy_score, master_df["y_true_dir_bin"].values, p_bin, dir_official_preds, timestamps_test)
            diff_mcc, m_l, m_h = block_bootstrap_diff(matthews_corrcoef, master_df["y_true_dir_bin"].values, p_bin, dir_official_preds, timestamps_test)
            acc_ci_str = f"[{a_l*100:+.2f}, {a_h*100:+.2f}]%"
            mcc_ci_str = f"[{m_l:+.3f}, {m_h:+.3f}]"
            chi2, p_mcn = mcnemar_test(master_df["y_true_dir_bin"].values, p_bin, dir_official_preds)
            mcn_str = f"chi2={chi2:.1f} (p<0.0001)" if p_mcn < 0.0001 else f"chi2={chi2:.1f} (p={p_mcn:.4f})"

        tablo5_rows.append({
            "run_id": run_id,
            "Model Adı": label,
            "Doğruluk (Acc)": round(acc, 4),
            "Baza Göre Acc GA": acc_ci_str,
            "Dengeli Doğruluk": round(bal_acc, 4),
            "MCC": round(mcc, 4),
            "Baza Göre MCC GA": mcc_ci_str,
            "F1 Skoru": round(f1, 4),
            "ROC AUC": auc_str,
            "Brier Skoru": bs_str,
            "BSS vs Hafta/Saat (%)": bss_dow_str,
            "BSS vs 4w Açık (%)": bss_4w_str,
            "McNemar Testi": mcn_str
        })
    df_tablo5 = pd.DataFrame(tablo5_rows)
    df_tablo5.to_csv(os.path.join(tables_dir, "tablo5_yon_siniflandirma_karsilastirma.csv"), index=False)

    # TABLO 6: 3 SINIFLI MODEL VE BAZLARI
    tablo6_rows = [
        {"run_id": run_id, "Model": "1. Saat Bazlı Çoğunluk Bazı (3 Sınıflı)", "Doğruluk (Acc)": round(float(accuracy_score(master_df["y_true_dir_3c"], master_df["bench_3c_hourly_maj"])), 4), "Dengeli Doğruluk": round(float(balanced_accuracy_score(master_df["y_true_dir_3c"], master_df["bench_3c_hourly_maj"])), 4), "Macro F1": round(float(f1_score(master_df["y_true_dir_3c"], master_df["bench_3c_hourly_maj"], average="macro")), 4), "MCC": round(float(matthews_corrcoef(master_df["y_true_dir_3c"], master_df["bench_3c_hourly_maj"])), 4)},
        {"run_id": run_id, "Model": "2. Hafta İçi/Sonu x Saat Bazı (3 Sınıflı)", "Doğruluk (Acc)": round(float(accuracy_score(master_df["y_true_dir_3c"], master_df["bench_3c_dow_hour_maj"])), 4), "Dengeli Doğruluk": round(float(balanced_accuracy_score(master_df["y_true_dir_3c"], master_df["bench_3c_dow_hour_maj"])), 4), "Macro F1": round(float(f1_score(master_df["y_true_dir_3c"], master_df["bench_3c_dow_hour_maj"], average="macro")), 4), "MCC": round(float(matthews_corrcoef(master_df["y_true_dir_3c"], master_df["bench_3c_dow_hour_maj"])), 4)},
        {"run_id": run_id, "Model": "3. LightGBM 3 Sınıflı Model", "Doğruluk (Acc)": round(float(accuracy_score(master_df["y_true_dir_3c"], master_df["pred_3c_lgb"])), 4), "Dengeli Doğruluk": round(float(balanced_accuracy_score(master_df["y_true_dir_3c"], master_df["pred_3c_lgb"])), 4), "Macro F1": round(float(f1_score(master_df["y_true_dir_3c"], master_df["pred_3c_lgb"], average="macro")), 4), "MCC": round(float(matthews_corrcoef(master_df["y_true_dir_3c"], master_df["pred_3c_lgb"])), 4)}
    ]
    df_tablo6 = pd.DataFrame(tablo6_rows)
    df_tablo6.to_csv(os.path.join(tables_dir, "tablo6_uc_sinifli_yon_karsilastirma.csv"), index=False)

    # TABLO 7: 24 SAATİN TAMAMI İÇİN MODEL DOĞRULUĞU VS EĞİTİMDEN TÜRETİLEN SAAT BAZI
    tablo7_rows = []
    for h in range(24):
        g = master_df[master_df["hour"] == h]
        y_h = g["y_true_dir_bin"].values
        p_m = g["pred_lgb_dir"].values
        p_b = g["bench_dir_hourly_maj"].values
        
        acc_m = float(accuracy_score(y_h, p_m))
        acc_b = float(accuracy_score(y_h, p_b))
        test_open_ratio = float(np.mean(y_h) * 100.0)
        oracle_acc = max(test_open_ratio, 100.0 - test_open_ratio)
        
        tablo7_rows.append({
            "run_id": run_id,
            "Saat": f"{h:02d}:00",
            "Test Açık Oranı (%)": round(test_open_ratio, 2),
            "Eğitimden Türetilen Saat Bazı Doğruluğu (%)": round(acc_b * 100.0, 2),
            "LightGBM Doğruluğu (%)": round(acc_m * 100.0, 2),
            "Model - Eğitim Bazı Farkı (% Puan)": round((acc_m - acc_b) * 100.0, 2),
            "Oracle (Test Verisi Çoğunluk Oranı) (%)": round(oracle_acc, 2)
        })
    df_tablo7 = pd.DataFrame(tablo7_rows)
    df_tablo7.to_csv(os.path.join(tables_dir, "tablo7_saatlik_yon_dogruluk.csv"), index=False)

    # ASSERT KONTROLÜ (Tablo 5 ve Tablo 7 Tutarlılığı):
    tablo7_model_acc_mean = float(np.mean(df_tablo7["LightGBM Doğruluğu (%)"]))
    tablo5_model_acc = float(df_tablo5.loc[df_tablo5["Model Adı"].str.contains("LightGBM"), "Doğruluk (Acc)"].values[0] * 100.0)
    tablo7_base_acc_mean = float(np.mean(df_tablo7["Eğitimden Türetilen Saat Bazı Doğruluğu (%)"]))
    tablo5_base_acc = float(df_tablo5.loc[df_tablo5["Model Adı"].str.contains("2. Saat Bazlı"), "Doğruluk (Acc)"].values[0] * 100.0)
    
    assert abs(tablo7_model_acc_mean - tablo5_model_acc) < 0.05, f"Tablo 7 ve Tablo 5 LightGBM uyuşmazlığı: {tablo7_model_acc_mean} vs {tablo5_model_acc}"
    assert abs(tablo7_base_acc_mean - tablo5_base_acc) < 0.05, f"Tablo 7 ve Tablo 5 Saat Bazı uyuşmazlığı: {tablo7_base_acc_mean} vs {tablo5_base_acc}"
    print("       Assert 2 Başarılı: Tablo 5 ve Tablo 7 yön doğrulukları birebir tutarlıdır.")

    # TABLO 10: ÇOKLU KESME ANI GERÇEK UFUK ANALİZİ (18:00 - 21:00 Puant)
    puant_idx = master_df["hour"].between(18, 21).values
    sub_puant = master_df[puant_idx].copy().reset_index(drop=True)
    y_puant = sub_puant["y_true_smf"].values
    ts_puant = sub_puant["timestamp"]

    # Cutoff A: D-1 12:00 (Ufuk: 30-33h). Ridge Sabit Tahmini
    pred_c_12 = sub_puant["pred_ridge_fixed"].values
    mae_c_12 = float(mean_absolute_error(y_puant, pred_c_12))

    # Cutoff B (06:00) ve Cutoff C (15:00) için özellik çıkarımı
    puant_test_indices = np.where(puant_idx)[0]
    train_val_len = len(full_train_df)

    sub_06 = test_df.iloc[puant_test_indices].copy().reset_index(drop=True)
    sub_15 = test_df.iloc[puant_test_indices].copy().reset_index(drop=True)

    sub_06_latest = []
    sub_06_same = []
    sub_15_latest = []
    sub_15_same = []

    for k, idx in enumerate(puant_test_indices):
        g_idx = train_val_len + idx
        h = int(test_df.iloc[idx]["hour"])
        
        # 06:00 cutoff: son bilinen saat 04:00 (gecikme = h - 4)
        # aynı saat baz: dünkü aynı saat (lag 24), dün akşam yayımlanmıştır
        sub_06_latest.append(df_clean.iloc[g_idx - (h - 4)]["smf"])
        sub_06_same.append(df_clean.iloc[g_idx - 24]["smf"])
        
        # 15:00 cutoff: son bilinen saat 13:00 (gecikme = h - 13)
        # aynı saat baz: dünkü aynı saat (lag 24)
        sub_15_latest.append(df_clean.iloc[g_idx - (h - 13)]["smf"])
        sub_15_same.append(df_clean.iloc[g_idx - 24]["smf"])

    sub_06["latest_known_smf"] = np.array(sub_06_latest)
    sub_06["legit_same_hour_smf"] = np.array(sub_06_same)

    sub_15["latest_known_smf"] = np.array(sub_15_latest)
    sub_15["legit_same_hour_smf"] = np.array(sub_15_same)

    pred_c_06 = ridge_fixed.predict(scaler_final.transform(sub_06[ridge_features]))
    mae_c_06 = float(mean_absolute_error(y_puant, pred_c_06))

    pred_c_15 = ridge_fixed.predict(scaler_final.transform(sub_15[ridge_features]))
    mae_c_15 = float(mean_absolute_error(y_puant, pred_c_15))

    # Bazlar (18-21 puant için):
    base_puant_official = sub_puant["bench_smf_median_4w"].values
    base_puant_ptf24 = sub_puant["bench_ptf_lag_24"].values
    mae_b_off = float(mean_absolute_error(y_puant, base_puant_official))
    mae_b_ptf = float(mean_absolute_error(y_puant, base_puant_ptf24))

    # Cutoff'a özgü anlık SMF bazları:
    mae_last_12 = float(mean_absolute_error(y_puant, sub_puant["latest_known_smf"].values))
    mae_last_06 = float(mean_absolute_error(y_puant, sub_06["latest_known_smf"].values))
    mae_last_15 = float(mean_absolute_error(y_puant, sub_15["latest_known_smf"].values))

    # ASSERT KONTROLÜ (Gereksinim 7):
    # Tablo 4'teki 18, 19, 20, 21 saatlerinin ortalama Ridge Sabit MAE'si ile Tablo 10'daki 12:00 satırı tutmalı:
    tablo4_puant_ridge_mae = float(df_tablo4.loc[df_tablo4["Saat"].isin(["18:00", "19:00", "20:00", "21:00"]), "Ridge Sabit MAE (TL)"].mean())
    assert abs(mae_c_12 - tablo4_puant_ridge_mae) < 0.05, f"Tablo 10 Cutoff 12:00 MAE ile Tablo 4 saat 18-21 ortalaması uyuşmuyor: {mae_c_12} vs {tablo4_puant_ridge_mae}"
    print("       Assert 3 Başarılı: Tablo 10 Cutoff 12:00 puant MAE'si ile Tablo 4 saat 18-21 ortalaması birebir tutarlıdır.")

    tablo10_rows = [
        {"run_id": run_id, "Kesme Anı (Cutoff)": "D-1 saat 12:00 (GÖP Öncesi)", "Gerçek Ufuk": "30 - 33 Saat Öncesi", "Kullanılan Model": "Ridge Regresyon (Sabit)", "Ridge MAE (TL)": round(mae_c_12, 2), "Resmi Baz (Medyan 4w) MAE (TL)": round(mae_b_off, 2), "Dünkü PTF (ptf_lag_24) MAE (TL)": round(mae_b_ptf, 2), "Son Bilinen SMF Baz MAE (TL)": round(mae_last_12, 2)},
        {"run_id": run_id, "Kesme Anı (Cutoff)": "D günü saat 06:00 (Gün İçi Sabah)", "Gerçek Ufuk": "12 - 15 Saat Öncesi", "Kullanılan Model": "Ridge Regresyon (Sabit)", "Ridge MAE (TL)": round(mae_c_06, 2), "Resmi Baz (Medyan 4w) MAE (TL)": round(mae_b_off, 2), "Dünkü PTF (ptf_lag_24) MAE (TL)": round(mae_b_ptf, 2), "Son Bilinen SMF Baz MAE (TL)": round(mae_last_06, 2)},
        {"run_id": run_id, "Kesme Anı (Cutoff)": "D günü saat 15:00 (Teslimat Öncesi)", "Gerçek Ufuk": "3 - 6 Saat Öncesi", "Kullanılan Model": "Ridge Regresyon (Sabit)", "Ridge MAE (TL)": round(mae_c_15, 2), "Resmi Baz (Medyan 4w) MAE (TL)": round(mae_b_off, 2), "Dünkü PTF (ptf_lag_24) MAE (TL)": round(mae_b_ptf, 2), "Son Bilinen SMF Baz MAE (TL)": round(mae_last_15, 2)}
    ]
    df_tablo10 = pd.DataFrame(tablo10_rows)
    df_tablo10.to_csv(os.path.join(tables_dir, "tablo10_gercek_ufuk_analizi.csv"), index=False)

    # MINCER-ZARNOWITZ REGRESYONU VE WALD TESTİ (Gereksinim 10)
    # y = alpha + beta * y_hat
    X_mz_ridge = sm.add_constant(master_df["pred_ridge_fixed"].values)
    res_mz_ridge = sm.OLS(master_df["y_true_smf"].values, X_mz_ridge).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    wald_r = res_mz_ridge.wald_test("const = 0, x1 = 1")
    stat_r = float(np.squeeze(wald_r.statistic))
    pval_r = float(np.squeeze(wald_r.pvalue))
    
    X_mz_base = sm.add_constant(master_df["bench_smf_median_4w"].values)
    res_mz_base = sm.OLS(master_df["y_true_smf"].values, X_mz_base).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    wald_b = res_mz_base.wald_test("const = 0, x1 = 1")
    stat_b = float(np.squeeze(wald_b.statistic))
    pval_b = float(np.squeeze(wald_b.pvalue))

    mz_summary_df = pd.DataFrame([
        {"run_id": run_id, "Model": "Ridge Regresyon (Sabit)", "Gözlem Sayısı (N)": len(master_df), "alpha (HAC se)": f"{res_mz_ridge.params[0]:.2f} ({res_mz_ridge.bse[0]:.2f})", "beta (HAC se)": f"{res_mz_ridge.params[1]:.4f} ({res_mz_ridge.bse[1]:.4f})", "R2": round(float(res_mz_ridge.rsquared), 4), "Wald Testi H0: (alpha=0, beta=1)": f"F={stat_r:.2f} (p={pval_r:.4e})"},
        {"run_id": run_id, "Model": "Resmi Baz (smf_median_4w)", "Gözlem Sayısı (N)": len(master_df), "alpha (HAC se)": f"{res_mz_base.params[0]:.2f} ({res_mz_base.bse[0]:.2f})", "beta (HAC se)": f"{res_mz_base.params[1]:.4f} ({res_mz_base.bse[1]:.4f})", "R2": round(float(res_mz_base.rsquared), 4), "Wald Testi H0: (alpha=0, beta=1)": f"F={stat_b:.2f} (p={pval_b:.4e})"}
    ])
    mz_summary_df.to_csv(os.path.join(tables_dir, "mincer_zarnowitz_analizi.csv"), index=False)

    # AYLIK ME İLE SMF KORELASYONU (Gereksinim 10)
    monthly_smf_means = df_tablo3["Ortalama SMF (TL)"].values
    monthly_ridge_bias = df_tablo3["Ridge Sabit ME/Bias (TL)"].values
    r_corr, r_p = stats.pearsonr(monthly_smf_means, monthly_ridge_bias)
    ci_corr = stats.pearsonr(monthly_smf_means, monthly_ridge_bias).confidence_interval(0.95)
    corr_summary_df = pd.DataFrame([{
        "run_id": run_id,
        "n_ay": len(monthly_smf_means),
        "Pearson Korelasyon (r)": round(float(r_corr), 4),
        "p-değeri": f"{r_p:.4f}",
        "95% GA": f"[{ci_corr[0]:.2f}, {ci_corr[1]:.2f}]"
    }])
    corr_summary_df.to_csv(os.path.join(tables_dir, "aylik_me_smf_korelasyon.csv"), index=False)

    # EĞİTİM / DOĞRULAMA / TEST HATA VE NMAE TABLOSU (Gereksinim 11)
    mean_tr_smf = float(np.mean(train_df["target_smf"]))
    mean_val_smf = float(np.mean(val_df["target_smf"]))
    
    sc_tr_eval = StandardScaler()
    X_tr_e = sc_tr_eval.fit_transform(train_df[ridge_features])
    X_val_e = sc_tr_eval.transform(val_df[ridge_features])
    r_eval = Ridge(alpha=best_alpha, random_state=42).fit(X_tr_e, train_df["target_smf"])
    
    r_tr_mae = float(mean_absolute_error(train_df["target_smf"], r_eval.predict(X_tr_e)))
    r_val_mae = float(mean_absolute_error(val_df["target_smf"], r_eval.predict(X_val_e)))
    r_te_mae = float(df_tablo1.loc[df_tablo1["Model Adı"].str.contains("Sabit Model"), "MAE (TL/MWh)"].values[0])

    m_lgb_tr = lgb.LGBMRegressor(n_estimators=250, random_state=42, verbose=-1, **best_lgb_params)
    m_lgb_tr.fit(train_df[ridge_features], train_df["target_smf"])
    lgb_tr_mae = float(mean_absolute_error(train_df["target_smf"], m_lgb_tr.predict(train_df[ridge_features])))
    lgb_val_mae = float(mean_absolute_error(val_df["target_smf"], m_lgb_tr.predict(val_df[ridge_features])))
    lgb_te_mae = float(df_tablo1.loc[df_tablo1["Model Adı"].str.contains(r"LightGBM Seviye \(Sabit"), "MAE (TL/MWh)"].values[0])

    split_comp_df = pd.DataFrame([
        {"run_id": run_id, "Dönem": "Eğitim (Train)", "Saat Sayısı": len(train_df), "Ortalama SMF (TL)": round(mean_tr_smf, 2), "Ridge MAE (TL)": round(r_tr_mae, 2), "Ridge NMAE (%)": round((r_tr_mae / mean_tr_smf)*100, 2), "LightGBM MAE (TL)": round(lgb_tr_mae, 2), "LightGBM NMAE (%)": round((lgb_tr_mae / mean_tr_smf)*100, 2)},
        {"run_id": run_id, "Dönem": "Doğrulama (Val)", "Saat Sayısı": len(val_df), "Ortalama SMF (TL)": round(mean_val_smf, 2), "Ridge MAE (TL)": round(r_val_mae, 2), "Ridge NMAE (%)": round((r_val_mae / mean_val_smf)*100, 2), "LightGBM MAE (TL)": round(lgb_val_mae, 2), "LightGBM NMAE (%)": round((lgb_val_mae / mean_val_smf)*100, 2)},
        {"run_id": run_id, "Dönem": "Dış Test (Test)", "Saat Sayısı": len(test_df), "Ortalama SMF (TL)": round(mean_test_smf, 2), "Ridge MAE (TL)": round(r_te_mae, 2), "Ridge NMAE (%)": round((r_te_mae / mean_test_smf)*100, 2), "LightGBM MAE (TL)": round(lgb_te_mae, 2), "LightGBM NMAE (%)": round((lgb_te_mae / mean_test_smf)*100, 2)}
    ])
    split_comp_df.to_csv(os.path.join(tables_dir, "donemler_arasi_hata_ve_nmae_tablosu.csv"), index=False)

    # -------------------------------------------------------------
    # 3. VIF ANALİZİ VE RIDGE KATSAYI İŞARET KARARLILIĞI (GEREKSİNİM 9)
    # -------------------------------------------------------------
    print("\n[3/5] VIF Analizi ve Ridge Katsayı İşaret Kararlılığı Bootstrap'i...")
    continuous_feats_full = [
        "load_forecast_mw", "forecast_residual_load_mw", "forecast_wind_ratio",
        "ptf_lag_24", "ptf_lag_48", "ptf_lag_168",
        "latest_known_smf", "legit_same_hour_smf",
        "smf_lag_48", "smf_lag_72", "smf_lag_168", "smf_median_4w", "smf_clean_same_hour_7d_mean"
    ]
    continuous_feats_selected = [c for c in continuous_feats_full if c != "forecast_residual_load_mw"]

    # 1. Residual Load Dahil VIF
    X_vif_full = full_train_df[continuous_feats_full].values
    vif_full = [float(variance_inflation_factor(X_vif_full, i)) for i in range(len(continuous_feats_full))]

    # 2. Residual Load Çıkarılmış VIF (Modelde Kullanılan)
    X_vif_sel = full_train_df[continuous_feats_selected].values
    vif_sel = [float(variance_inflation_factor(X_vif_sel, i)) for i in range(len(continuous_feats_selected))]

    vif_rows = []
    for col, vf in zip(continuous_feats_full, vif_full):
        vs = vif_sel[continuous_feats_selected.index(col)] if col in continuous_feats_selected else np.nan
        vif_rows.append({
            "run_id": run_id,
            "Özellik": col,
            "VIF (Tüm Değişkenler Dahil)": round(vf, 2),
            "VIF (Residual Load Çıkarılmış - Model Sürümü)": round(vs, 2) if not np.isnan(vs) else "Çıkarıldı (Lineer Bağımlı)"
        })
    df_vif = pd.DataFrame(vif_rows)
    df_vif.to_csv(os.path.join(tables_dir, "vif_analiz_tablosu.csv"), index=False)

    # 3. Ridge Katsayı İşaret Kararlılığı (1000 Blok Bootstrap)
    rng_coef = np.random.RandomState(42)
    dates_full_train = full_train_df["timestamp"].dt.date.unique()
    n_days_tr = len(dates_full_train)
    n_blocks_tr = int(np.ceil(n_days_tr / 7))
    date_to_idx_tr = full_train_df.groupby(full_train_df["timestamp"].dt.date).indices

    coef_bootstraps = []
    for _ in range(1000):
        start_ids = rng_coef.randint(0, n_days_tr - 7 + 1, size=n_blocks_tr)
        s_dates = []
        for s in start_ids:
            s_dates.extend(dates_full_train[s:s + 7])
        s_dates = s_dates[:n_days_tr]
        s_rows = np.concatenate([date_to_idx_tr[d] for d in s_dates])
        
        sub_X = full_train_df.iloc[s_rows][ridge_features]
        sub_y = full_train_df.iloc[s_rows]["target_smf"]
        
        sc_b = StandardScaler()
        X_scaled_b = sc_b.fit_transform(sub_X)
        r_b = Ridge(alpha=best_alpha, random_state=42).fit(X_scaled_b, sub_y)
        coef_bootstraps.append(r_b.coef_)
        
    coef_bootstraps = np.array(coef_bootstraps) # shape (1000, n_features)

    coef_stability_rows = []
    base_coefs = ridge_fixed.coef_
    for j, f_name in enumerate(ridge_features):
        b_j = coef_bootstraps[:, j]
        c_mean = float(np.mean(b_j))
        c_std = float(np.std(b_j))
        c_low = float(np.percentile(b_j, 2.5))
        c_high = float(np.percentile(b_j, 97.5))
        pos_pct = float(np.mean(b_j > 0) * 100.0)
        
        coef_stability_rows.append({
            "run_id": run_id,
            "Özellik": f_name,
            "Ridge Standartlaştırılmış Katsayı (Beta)": round(base_coefs[j], 2),
            "Bootstrap Ortalama": round(c_mean, 2),
            "Bootstrap 95% GA": f"[{c_low:+.2f}, {c_high:+.2f}]",
            "Pozitif Katsayı Oranı (%)": round(pos_pct, 1),
            "İşaret Kararlılığı": "Kararlı (Pozitif)" if pos_pct >= 95 else ("Kararlı (Negatif)" if pos_pct <= 5 else "Kararsız / Nötr")
        })
    df_coef_stab = pd.DataFrame(coef_stability_rows)
    df_coef_stab.to_csv(os.path.join(tables_dir, "ridge_katsayi_ve_isaret_kararliligi.csv"), index=False)

    # -------------------------------------------------------------
    # 4. ZEHİR TESTİ (POISON TEST - GEREKSİNİM 1 & DOĞRULAMA)
    # -------------------------------------------------------------
    print("\n[4/5] Zehir Testi (Point-in-Time Gelecek Sızıntı Denetimi)...")
    test_dates = test_df["timestamp"].dt.date.unique()
    rng_pois = np.random.RandomState(42)
    sample_poison_days = rng_pois.choice(test_dates, size=5, replace=False)
    
    tz_info = df_clean["timestamp"].dt.tz
    for p_day in sample_poison_days:
        td = pd.Timestamp(p_day, tz=tz_info)
        cutoff = td - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
        df_poisoned = df_raw.copy()
        mask_future = (df_poisoned["timestamp"] > cutoff)
        df_poisoned.loc[mask_future, ["smf", "systemStatus"]] = np.nan
        df_poisoned.loc[df_poisoned["timestamp"] >= td, "ptf"] = np.nan
        
        df_p_feat = build_feature_dataframe(df_poisoned, drop_initial_lags=False)
        
        day_mask_clean = (df_clean["timestamp"].dt.date == p_day).values
        day_mask_pois = (df_p_feat["timestamp"].dt.date == p_day).values
        orig_day_feats = df_clean.loc[day_mask_clean, ridge_features].values
        pois_day_feats = df_p_feat.loc[day_mask_pois, ridge_features].values
        
        diff_pois = np.nanmax(np.abs(orig_day_feats - pois_day_feats))
        has_nan = np.isnan(pois_day_feats).any()
        
        assert not has_nan, f"Zehir testi başarısız: {p_day} tarihinde özelliklerde NaN oluştu!"
        assert diff_pois < 1e-5, f"Zehir testi başarısız: {p_day} tarihinde fark={diff_pois}"
        
    print("       Zehir Testi Başarılı: 5 rastgele test gününde gelecek tamamen zehirlendiğinde bile özellikler 0 hata ile korundu.")

    # -------------------------------------------------------------
    # 5. TABLO EŞLEŞTİRME VE SHA-256 HASH TABLOSU (GEREKSİNİM 1 & 14)
    # -------------------------------------------------------------
    print("\n[5/5] Tablo Dosya Eşleştirmesi ve SHA-256 Sağlama Toplamı Tablosu...")
    table_files = [
        ("Tablo 1: Fiyat Tahmini Karşılaştırma Tablosu", "tablo1_fiyat_tahmini_karsilastirma.csv"),
        ("Tablo 2: Özel Çiftler İçin İki Yönlü HAC DM Testleri", "tablo2_ozel_ciftler_diebold_mariano.csv"),
        ("Tablo 3: 12 Aylık Hata ve ME/Bias Dağılımı", "tablo3_aylik_hata_ve_bias.csv"),
        ("Tablo 4: 24 Saatlik MAE ve Baz Karşılaştırması", "tablo4_saatlik_24_saat_mae.csv"),
        ("Tablo 5: Yön Sınıflandırma Karşılaştırma Tablosu", "tablo5_yon_siniflandirma_karsilastirma.csv"),
        ("Tablo 6: 3 Sınıflı Yön Modeli ve Bazları", "tablo6_uc_sinifli_yon_karsilastirma.csv"),
        ("Tablo 7: 24 Saat İçin Model Doğruluğu vs Eğitim Saat Bazı", "tablo7_saatlik_yon_dogruluk.csv"),
        ("Tablo 8: Dönemler Arası Hata ve NMAE Dağılımı", "donemler_arasi_hata_ve_nmae_tablosu.csv"),
        ("Tablo 9: Çoklu Doğrusal Bağlantı ve VIF Analizi", "vif_analiz_tablosu.csv"),
        ("Tablo 10: Çoklu Kesme Anı Gerçek Ufuk Analizi (18-21 Puant)", "tablo10_gercek_ufuk_analizi.csv"),
        ("Tablo 11: Mincer-Zarnowitz Regresyonu ve Wald Testi", "mincer_zarnowitz_analizi.csv"),
        ("Ek Tablo: Aylık ME ile SMF Korelasyonu", "aylik_me_smf_korelasyon.csv"),
        ("Ek Tablo: Ridge Katsayı ve İşaret Kararlılığı", "ridge_katsayi_ve_isaret_kararliligi.csv"),
        ("Kaynak Veri: EPİAŞ Resmi Piyasa Verisi (Parquet)", "../../data/raw/epias_official_market_data_full.parquet"),
        ("Master Tahmin: Tüm Modellerin Test Tahmin Parquet'si", "../../data/processed/test_predictions_master.parquet")
    ]
    
    manifest_rows = []
    for t_desc, f_rel in table_files:
        if f_rel.startswith("../"):
            f_path = os.path.normpath(os.path.join(tables_dir, f_rel))
        else:
            f_path = os.path.join(tables_dir, f_rel)
            
        if os.path.exists(f_path):
            f_size = os.path.getsize(f_path)
            with open(f_path, "rb") as f:
                f_hash = hashlib.sha256(f.read()).hexdigest()
            manifest_rows.append({
                "run_id": run_id,
                "Rapor Tablosu": t_desc,
                "Kaynak Dosya": f_rel.replace("../../", ""),
                "Boyut (Bayt)": f_size,
                "SHA-256 Hash": f_hash
            })
            
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(tables_dir, "tablo_eslestirme_ve_dogrulama.csv"), index=False)

    print("=" * 110)
    print(f"  TÜM TABLOLAR MASTER DATAFRAME ÜZERİNDEN BAŞARIYLA ÜRETİLDİ | RUN ID: {run_id}")
    print("=" * 110)
    return run_id


if __name__ == "__main__":
    execute_master_pipeline()

