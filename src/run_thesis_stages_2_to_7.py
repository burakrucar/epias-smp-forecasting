"""
T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - MATEMATİK BÖLÜMÜ BİTİRME TEZİ
SMF VE DENGELEME YÖNÜ TAHMİNİ: AŞAMA 2 - AŞAMA 7 UÇTAN UCA ÇALIŞTIRMA VE DOĞRULAMA MOTORU

Bu betik, çalışma planındaki Aşama 2'den Aşama 7'ye kadar olan tüm adımları tek seferde,
sıfır veri sızıntısıyla (leakage-free) ve katı matematiksel kurallarla icra eder:
- Aşama 2: Veri Yükleme ve Yayın Takvimi (Publication Lag) Tablosu
- Aşama 3: Veri Temizleme, Dağılım Analizi ve Ekonometrik Durağanlık Testleri (ADF & KPSS)
- Aşama 4: Sızıntısız Deney Tasarımı (D-1 12:00 Cutoff, 12-36h Horizon, Temporal Split)
- Aşama 5: Baz Modeller (Seasonal Naive 24, Seasonal Naive 168, SMF=PTF, Çoğunluk Yönü)
- Aşama 6: İstatistiksel (Ridge, Lojistik) ve Makine Öğrenmesi (LightGBM) Modelleri
- Aşama 7: Karşılaştırmalı Metrikler (MAE, RMSE, sMAPE, Acc, BalAcc, MCC), Diebold-Mariano Testi,
           Ufuk Bazlı (12h-35h) Hata Ayrıştırması ve SHAP Açıklanabilirlik Analizi.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef
)
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 0. YARDIMCI METRİK VE İSTATİSTİK FONKSİYONLARI
# -------------------------------------------------------------
def calculate_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)

def diebold_mariano_test(y_true, y_pred_model, y_pred_bench, h=1, loss="mae"):
    """
    Harvey, Leybourne and Newbold (1997) düzeltmeli Diebold-Mariano testi.
    H0: Model ve benchmark eşit tahmin doğruluğuna sahiptir.
    H1: Model, benchmark'tan istatistiki olarak üstündür (tek taraflı test).
    """
    if loss == "mae":
        d = np.abs(y_true - y_pred_bench) - np.abs(y_true - y_pred_model)
    else:
        d = (y_true - y_pred_bench)**2 - (y_true - y_pred_model)**2
    
    n = len(d)
    mean_d = np.mean(d)
    
    # Otokorelasyon düzeltmesi (Newey-West / Bartlett çekirdeği)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        cov = np.cov(d[:-lag], d[lag:])[0, 1]
        gamma_sum += 2.0 * weight * cov
    
    var_d = (gamma0 + gamma_sum) / n
    if var_d <= 0:
        return 0.0, 0.5
    
    dm_stat = mean_d / np.sqrt(var_d)
    # HLN küçük örneklem düzeltmesi
    hln_stat = dm_stat * np.sqrt((n + 1 - 2*h + (h * (h - 1) / n)) / n)
    p_value = 1.0 - stats.norm.cdf(hln_stat)  # tek taraflı p-değeri (üstünlük testi)
    return float(hln_stat), float(p_value)


def run_full_thesis_pipeline():
    print("=" * 95)
    print("  T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - MATEMATİK BÖLÜMÜ BİTİRME TEZİ")
    print("  SMF VE SİSTEM YÖNÜ TAHMİN MOTORU (AŞAMA 2 - AŞAMA 7 DOĞRULAMA PROTOKOLÜ)")
    print("=" * 95)

    reports_dir = "reports"
    tables_dir = os.path.join(reports_dir, "tables")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # -------------------------------------------------------------
    # AŞAMA 2: VERİ TOPLAMA VE YAYIN ZAMANI (PUBLICATION LAG) TABLOSU
    # -------------------------------------------------------------
    print("\n[AŞAMA 2] Veri Yükleme ve Yayın Zamanı Sızıntı Tablosu Oluşturuluyor...")
    data_path = "data/raw/epias_official_market_data_full.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Veri bulunamadı: {data_path}")

    df_raw = pd.read_parquet(data_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    pub_table = [
        {"Veri": "SMF (Sistem Marjinal Fiyatı)", "Yayınlayan": "TEİAŞ / EPİAŞ", "Yayın Zamanı": "D+1 veya saatlik gecikmeyle", "D-1 12:00 Durumu": "Yalnızca t-24 ve önceki saatler kesin bilinir"},
        {"Veri": "PTF (Piyasa Takas Fiyatı)", "Yayınlayan": "EPİAŞ GÖP", "Yayın Zamanı": "D-1 13:00 - 14:00", "D-1 12:00 Durumu": "D-1 12:00'de HENÜZ YOK (Aşama 1'de girdi olamaz)"},
        {"Veri": "Yük Tahmin Planı (Load Forecast)", "Yayınlayan": "TEİAŞ", "Yayın Zamanı": "D-1 saat 10:30 - 11:30", "D-1 12:00 Durumu": "D gününün 24 saati için kamuya açıklanmış (KULLANILABİLİR)"},
        {"Veri": "Rüzgar Üretim Tahmini (Wind Forecast)", "Yayınlayan": "TEİAŞ RİTM", "Yayın Zamanı": "D-1 saat 10:30 - 11:30", "D-1 12:00 Durumu": "D gününün 24 saati için kamuya açıklanmış (KULLANILABİLİR)"},
        {"Veri": "Sistem Denge Yönü (systemStatus)", "Yayınlayan": "TEİAŞ", "Yayın Zamanı": "D günü gerçek zaman", "D-1 12:00 Durumu": "D günü gerçekleşen yön bilinmez; geçmiş gecikmeli yön bilinir"}
    ]
    df_pub = pd.DataFrame(pub_table)
    df_pub.to_csv(os.path.join(tables_dir, "yayin_zamani_tablosu.csv"), index=False)
    print("  -> Yayın Zamanı Tablosu kaydedildi: reports/tables/yayin_zamani_tablosu.csv")

    # -------------------------------------------------------------
    # AŞAMA 3: VERİ TEMİZLEME VE EKONOMETRİK DURAĞANLIK ANALİZİ
    # -------------------------------------------------------------
    print("\n[AŞAMA 3] Keşifsel Veri Analizi ve Ekonometrik Durağanlık Testleri (ADF / KPSS)...")
    
    # Eksik değerleri causal (ffill) doldur
    df_raw["load_forecast_mw"] = df_raw["load_forecast_mw"].ffill()
    df_raw["wind_forecast_mw"] = df_raw["wind_forecast_mw"].ffill()
    
    smf_series = df_raw["smf"].values
    ptf_series = df_raw["ptf"].values
    spread_series = smf_series - ptf_series
    log_smf_series = np.log(np.maximum(smf_series, 1.0))
    diff_smf_series = np.diff(smf_series)

    # İstatistiki Dağılım Özellikleri
    eda_stats = {
        "Metrik": ["Gözlem Sayısı (Saat)", "Ortalama (TL/MWh)", "Standart Sapma", "Medyan", "Minimum", "Maksimum", "Çarpıklık (Skewness)", "Basıklık (Kurtosis)"],
        "SMF Fiyatı": [len(smf_series), np.mean(smf_series), np.std(smf_series), np.median(smf_series), np.min(smf_series), np.max(smf_series), stats.skew(smf_series), stats.kurtosis(smf_series)],
        "PTF Fiyatı": [len(ptf_series), np.mean(ptf_series), np.std(ptf_series), np.median(ptf_series), np.min(ptf_series), np.max(ptf_series), stats.skew(ptf_series), stats.kurtosis(ptf_series)],
        "Yayılım (SMF - PTF)": [len(spread_series), np.mean(spread_series), np.std(spread_series), np.median(spread_series), np.min(spread_series), np.max(spread_series), stats.skew(spread_series), stats.kurtosis(spread_series)]
    }
    df_eda = pd.DataFrame(eda_stats)
    df_eda.to_csv(os.path.join(tables_dir, "eda_ozet_istatistikleri.csv"), index=False)

    # ADF ve KPSS Testleri
    def perform_stationarity_tests(series, name):
        # ADF Testi (H0: Birim kök vardır - Seri durağan DEĞİLDİR)
        adf_res = adfuller(series, autolag="AIC")
        adf_stat, adf_p = adf_res[0], adf_res[1]
        
        # KPSS Testi (H0: Seri durağandır)
        kpss_res = kpss(series, regression="c", nlags="auto")
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
        
        adf_conclusion = "Durağan (Birim Kök Reddedildi)" if adf_p < 0.05 else "Durağan Değil (Birim Kök Var)"
        kpss_conclusion = "Durağan (H0 Reddedilemedi)" if kpss_p >= 0.05 else "Durağan Değil (H0 Reddedildi)"
        
        return {
            "Seri Adı": name,
            "ADF Test İstatistiği": round(adf_stat, 4),
            "ADF p-değeri": f"{adf_p:.4e}",
            "ADF Kararı (p<0.05)": adf_conclusion,
            "KPSS Test İstatistiği": round(kpss_stat, 4),
            "KPSS p-değeri": f"{kpss_p:.4e}",
            "KPSS Kararı (p>=0.05)": kpss_conclusion
        }

    stat_results = [
        perform_stationarity_tests(smf_series, "Ham SMF (TL/MWh)"),
        perform_stationarity_tests(log_smf_series, "Log(SMF)"),
        perform_stationarity_tests(diff_smf_series, "Fark Serisi (Delta SMF)"),
        perform_stationarity_tests(spread_series, "Yayılım Serisi (SMF - PTF)")
    ]
    df_stationarity = pd.DataFrame(stat_results)
    df_stationarity.to_csv(os.path.join(tables_dir, "duraganlik_test_sonuclari.csv"), index=False)
    print("  -> Durağanlık (ADF/KPSS) Tablosu kaydedildi: reports/tables/duraganlik_test_sonuclari.csv")

    # Sınıf Dağılımı
    status_counts = df_raw["systemStatus"].value_counts()
    status_ratios = df_raw["systemStatus"].value_counts(normalize=True) * 100
    df_class_dist = pd.DataFrame({
        "Sistem Yönü": status_counts.index,
        "Saat Sayısı": status_counts.values,
        "Oran (%)": status_ratios.round(2).values
    })
    df_class_dist.to_csv(os.path.join(tables_dir, "sinif_dagilimi_tablosu.csv"), index=False)

    # -------------------------------------------------------------
    # AŞAMA 4: SIZINTISIZ ÖZELLİK MÜHENDİSLİĞİ VE ZAMAN BÖLÜMLEMESİ
    # -------------------------------------------------------------
    print("\n[AŞAMA 4] Sızıntısız Zaman Serisi Özellik Seti Oluşturuluyor (D-1 12:00 Cutoff)...")
    
    n = len(df_raw)
    hours = df_raw["timestamp"].dt.hour.values
    dows = df_raw["timestamp"].dt.dayofweek.values
    months = df_raw["timestamp"].dt.month.values

    df = df_raw.copy()
    df["hour"] = hours
    df["dow"] = dows
    df["month"] = months
    df["is_weekend"] = (dows >= 5).astype(int)

    # Döngüsel Trigonometrik Kodlama
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * months / 12.0)

    # Kalan Yük ve Yenilenebilir Oranları
    df["forecast_residual_load_mw"] = df["load_forecast_mw"] - df["wind_forecast_mw"]
    df["forecast_wind_ratio"] = df["wind_forecast_mw"] / np.maximum(df["load_forecast_mw"], 1.0)

    # Yön haritalaması: Açık = 1, Fazla = -1, Dengede = 0
    status_map = {"Enerji Açığı": 1, "Enerji Fazlası": -1, "Dengede": 0}
    num_status = df["systemStatus"].map(status_map).values

    # D-1 12:00 cutoff için en son bilinen saatlik gecikmeler (lag_offset = 14)
    # Teslimat saati h için D-1 10:00 saatinin uzaklığı = 14 + h saat
    lag_offset = 14
    latest_smf = np.full(n, np.nan)
    latest_ptf = np.full(n, np.nan)
    latest_dir = np.full(n, np.nan)

    for i in range(n):
        lag = lag_offset + hours[i]
        if i >= lag:
            latest_smf[i] = df["smf"].iloc[i - lag]
            latest_ptf[i] = df["ptf"].iloc[i - lag]
            latest_dir[i] = num_status[i - lag]

    df["latest_known_smf"] = latest_smf
    df["latest_known_ptf"] = latest_ptf
    df["latest_known_status"] = latest_dir

    # Sabit Sızıntısız Gecikmeler
    for l in [24, 48, 72, 168]:
        df[f"smf_lag_{l}"] = df["smf"].shift(l)
        df[f"ptf_lag_{l}"] = df["ptf"].shift(l)
        df[f"status_lag_{l}"] = pd.Series(num_status).shift(l).values

    # Haftalık aynı saat ortalaması (lag 48 - 168 arası 6 günün ortalaması)
    same_hour_lags = [df["smf"].shift(l) for l in [48, 72, 96, 120, 144, 168]]
    df["smf_same_hour_7d_mean"] = pd.concat(same_hour_lags, axis=1).mean(axis=1)

    # D-2 rolling istatistikler
    df["smf_mean_d2"] = df["smf"].shift(48).rolling(24).mean()
    df["smf_std_d2"] = df["smf"].shift(48).rolling(24).std()

    # Tahmin Ufku (12 saat ile 35 saat)
    df["horizon_hours"] = 12 + hours

    # Hedefler
    df["target_direction"] = num_status
    df["target_smf"] = df["smf"].values
    df["target_spread"] = df["smf"].values - df["ptf"].values

    df_clean = df.dropna().reset_index(drop=True)

    # Özellik sütunları
    feature_cols_gop_oncesi = [
        "hour", "dow", "month", "is_weekend", "horizon_hours",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "load_forecast_mw", "wind_forecast_mw", "forecast_residual_load_mw", "forecast_wind_ratio",
        "latest_known_smf", "latest_known_ptf", "latest_known_status",
        "smf_lag_48", "ptf_lag_48", "status_lag_48",
        "smf_lag_72", "ptf_lag_72", "status_lag_72",
        "smf_lag_168", "ptf_lag_168", "status_lag_168",
        "smf_same_hour_7d_mean", "smf_mean_d2", "smf_std_d2"
    ]
    feature_cols_gop_sonrasi = feature_cols_gop_oncesi + ["ptf"]

    # Zaman Bölmesi (Train: 2024, Val: 2025 ilk 9 ay, Test: Son 12 ay - 365 gün)
    train_mask = (df_clean["timestamp"] < "2024-10-01").values
    val_mask = ((df_clean["timestamp"] >= "2024-10-01") & (df_clean["timestamp"] < "2025-10-01")).values
    test_mask = (df_clean["timestamp"] >= "2025-10-01").values

    train_df = df_clean[train_mask | val_mask].reset_index(drop=True)  # Final model Train+Val
    test_df = df_clean[test_mask].reset_index(drop=True)

    print(f"  -> Toplam Temiz Veri: {len(df_clean)} saat")
    print(f"  -> Eğitim Kümesi:    {len(train_df)} saat ({train_df['timestamp'].min().strftime('%Y-%m-%d')} -> {train_df['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  -> Test Kümesi:      {len(test_df)} saat ({test_df['timestamp'].min().strftime('%Y-%m-%d')} -> {test_df['timestamp'].max().strftime('%Y-%m-%d')})")

    # -------------------------------------------------------------
    # AŞAMA 5: BAZ MODELLER (BENCHMARKS)
    # -------------------------------------------------------------
    print("\n[AŞAMA 5] Baz (Benchmark) Modeller Hesaplanıyor...")

    y_test_smf = test_df["target_smf"].values
    y_test_dir = test_df["target_direction"].values
    ptf_test = test_df["ptf"].values

    # Naive Fiyat Referansları
    pred_naive_lag24_smf = test_df["smf_lag_24"].values
    pred_naive_lag168_smf = test_df["smf_lag_168"].values
    pred_naive_ptf_rule = ptf_test.copy()  # "SMF = PTF" kuralı (GÖP Sonrası referansı)

    # Naive Yön Referansları
    pred_naive_majority_dir = np.ones_like(y_test_dir)  # Her Zaman Açık
    pred_naive_lag24_dir = test_df["status_lag_24"].values
    pred_naive_lag168_dir = test_df["status_lag_168"].values

    # -------------------------------------------------------------
    # AŞAMA 6: MODELLERİN EĞİTİLMESİ (RIDGE, LOJİSTİK, LIGHTGBM)
    # -------------------------------------------------------------
    print("\n[AŞAMA 6] İstatistiksel ve Makine Öğrenmesi Modelleri Eğitiliyor...")

    X_train_pre = train_df[feature_cols_gop_oncesi].values
    X_test_pre = test_df[feature_cols_gop_oncesi].values
    y_train_smf = train_df["target_smf"].values
    y_train_dir = train_df["target_direction"].values
    y_train_spread = train_df["target_spread"].values

    X_train_post = train_df[feature_cols_gop_sonrasi].values
    X_test_post = test_df[feature_cols_gop_sonrasi].values

    # Standartlaştırma (Doğrusal Modeller için)
    scaler_pre = StandardScaler()
    X_train_pre_scaled = scaler_pre.fit_transform(X_train_pre)
    X_test_pre_scaled = scaler_pre.transform(X_test_pre)

    # 1. Doğrusal İstatistiki Model: Ridge Regresyon
    print("  -> 1. Ridge Regresyon (L2 Düzenlileştirilmiş ARX Vekili)...")
    ridge_model = Ridge(alpha=100.0, random_state=42)
    ridge_model.fit(X_train_pre_scaled, y_train_smf)
    pred_ridge_smf = ridge_model.predict(X_test_pre_scaled)

    # 2. İstatistiki Sınıflandırma: Lojistik Regresyon
    print("  -> 2. Lojistik Regresyon Sınıflandırıcı...")
    # Sınıf etiketlerini 0, 1, 2'ye eşle (-1:0, 0:1, 1:2)
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map = {0: -1, 1: 0, 2: 1}
    y_train_mapped = np.array([label_map[v] for v in y_train_dir])
    
    logreg_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logreg_model.fit(X_train_pre_scaled, y_train_mapped)
    pred_logreg_dir = np.array([inv_map[v] for v in logreg_model.predict(X_test_pre_scaled)])

    # 3. Gelişmiş Ağaç Tabanlı Model: LightGBM Regresyon (GÖP Öncesi SMF)
    print("  -> 3. LightGBM Doğrudan SMF Regresörü (GÖP Öncesi - D-1 12:00)...")
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        min_child_samples=30,
        random_state=42,
        verbose=-1
    )
    lgb_reg.fit(X_train_pre, y_train_smf)
    pred_lgb_pre_smf = lgb_reg.predict(X_test_pre)

    # 4. Gelişmiş Ağaç Tabanlı Sınıflandırıcı: LightGBM Multiclass
    print("  -> 4. LightGBM Sistem Yönü Sınıflandırıcısı...")
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_samples=40,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    lgb_clf.fit(X_train_pre, y_train_mapped)
    pred_lgb_dir = np.array([inv_map[v] for v in lgb_clf.predict(X_test_pre)])

    # 5. GÖP Sonrası Yayılım Modeli: LightGBM Spread (SMF = PTF + Spread)
    print("  -> 5. GÖP Sonrası Yayılım Modeli (SMF = PTF + Pred_Spread)...")
    lgb_spread = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        min_child_samples=30,
        random_state=42,
        verbose=-1
    )
    lgb_spread.fit(X_train_post, y_train_spread)
    pred_lgb_spread = lgb_spread.predict(X_test_post)
    pred_lgb_post_smf = ptf_test + pred_lgb_spread

    # -------------------------------------------------------------
    # AŞAMA 7: DEĞERLENDİRME, METRİKLER VE DİEBOLD-MARİANO TESTİ
    # -------------------------------------------------------------
    print("\n[AŞAMA 7] Karşılaştırmalı Değerlendirme ve İstatistiki Testler...")

    # A) FİYAT TAHMİNİ KARŞILAŞTIRMA TABLOSU
    benchmark_smf = pred_naive_lag168_smf  # Birincil Naif Baz Model

    models_price = [
        ("1. Naif Model 1 (Dünkü Aynı Saat - Lag 24)", pred_naive_lag24_smf, "Baz Model"),
        ("2. Naif Model 2 (Geçen Haftanın Aynı Saati - Lag 168)", pred_naive_lag168_smf, "Baz Model"),
        ("3. Naif Kural (SMF = PTF Kuralı)", pred_naive_ptf_rule, "GÖP Sonrası Baz"),
        ("4. İstatistiksel Model: Ridge Regresyon (ARX)", pred_ridge_smf, "İstatistiksel"),
        ("5. Gelişmiş Model: LightGBM Regresyon (GÖP Öncesi)", pred_lgb_pre_smf, "Makine Öğrenmesi"),
        ("6. Gelişmiş Model: LightGBM Yayılım Modeli (GÖP Sonrası)", pred_lgb_post_smf, "Makine Öğrenmesi")
    ]

    price_results = []
    for name, preds, family in models_price:
        mae = mean_absolute_error(y_test_smf, preds)
        rmse = root_mean_squared_error(y_test_smf, preds)
        smape = calculate_smape(y_test_smf, preds)
        
        # Diebold-Mariano testi (Lag 168 Naif Baz Modeline Karşı)
        if name == "2. Naif Model 2 (Geçen Haftanın Aynı Saati - Lag 168)":
            dm_stat, p_val = 0.0, 1.0
            sig = "-"
        else:
            dm_stat, p_val = diebold_mariano_test(y_test_smf, preds, benchmark_smf, h=24, loss="mae")
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "Anlamsız"))

        price_results.append({
            "Model Adı": name,
            "Model Ailesi": family,
            "MAE (TL/MWh)": round(mae, 2),
            "RMSE (TL/MWh)": round(rmse, 2),
            "sMAPE (%)": round(smape, 2),
            "DM Test İstatistiği": round(dm_stat, 3),
            "DM p-değeri": f"{p_val:.4f}",
            "İstatistiki Üstünlük": sig
        })

    df_price_eval = pd.DataFrame(price_results)
    df_price_eval.to_csv(os.path.join(tables_dir, "smf_fiyat_tahmini_karsilastirma.csv"), index=False)
    print("  -> Fiyat Tahmin Tablosu kaydedildi: reports/tables/smf_fiyat_tahmini_karsilastirma.csv")

    # B) SİSTEM YÖNÜ SINIFLANDIRMA KARŞILAŞTIRMA TABLOSU
    models_dir = [
        ("1. Çoğunluk Sınıfı (Her Zaman Açık)", pred_naive_majority_dir, "Baz Model"),
        ("2. Dünkü Yön Takipçisi (Lag 24)", pred_naive_lag24_dir, "Baz Model"),
        ("3. Geçen Haftanın Yön Takipçisi (Lag 168)", pred_naive_lag168_dir, "Baz Model"),
        ("4. İstatistiksel Model: Lojistik Regresyon", pred_logreg_dir, "İstatistiksel"),
        ("5. Gelişmiş Model: LightGBM Sınıflandırıcı", pred_lgb_dir, "Makine Öğrenmesi")
    ]

    dir_results = []
    for name, preds, family in models_dir:
        acc = accuracy_score(y_test_dir, preds)
        bal_acc = balanced_accuracy_score(y_test_dir, preds)
        macro_f1 = f1_score(y_test_dir, preds, average="macro")
        mcc = matthews_corrcoef(y_test_dir, preds)

        dir_results.append({
            "Model Adı": name,
            "Model Ailesi": family,
            "Doğruluk (Accuracy)": round(acc, 4),
            "Dengeli Doğruluk (Bal. Acc)": round(bal_acc, 4),
            "Macro F1": round(macro_f1, 4),
            "MCC": round(mcc, 4)
        })

    df_dir_eval = pd.DataFrame(dir_results)
    df_dir_eval.to_csv(os.path.join(tables_dir, "sistem_yonu_siniflandirma_karsilastirma.csv"), index=False)
    print("  -> Yön Sınıflandırma Tablosu kaydedildi: reports/tables/sistem_yonu_siniflandirma_karsilastirma.csv")

    # C) UFIK BAZLI (HORIZON EFFECT: 12h - 35h) HATA ANALİZİ
    print("\n  -> Tahmin Ufku Analizi (12. Saatten 35. Saate Kadar Hata Ayrıştırması)...")
    horizon_results = []
    test_df_copy = test_df.copy()
    test_df_copy["pred_lgb_pre"] = pred_lgb_pre_smf
    test_df_copy["pred_naive_168"] = pred_naive_lag168_smf
    test_df_copy["pred_lgb_dir"] = pred_lgb_dir

    for h in sorted(test_df_copy["horizon_hours"].unique()):
        sub = test_df_copy[test_df_copy["horizon_hours"] == h]
        h_mae_lgb = mean_absolute_error(sub["target_smf"], sub["pred_lgb_pre"])
        h_mae_naive = mean_absolute_error(sub["target_smf"], sub["pred_naive_168"])
        h_acc_dir = accuracy_score(sub["target_direction"], sub["pred_lgb_dir"])
        
        horizon_results.append({
            "Ufuk (Saat)": int(h),
            "Teslimat Saati": f"{int(h-12):02d}:00",
            "LightGBM SMF MAE (TL/MWh)": round(h_mae_lgb, 2),
            "Naive 168 SMF MAE (TL/MWh)": round(h_mae_naive, 2),
            "Model İyileşmesi (TL/MWh)": round(h_mae_naive - h_mae_lgb, 2),
            "Yön Tahmin Doğruluğu (%)": round(h_acc_dir * 100, 2)
        })

    df_horizon = pd.DataFrame(horizon_results)
    df_horizon.to_csv(os.path.join(tables_dir, "tahmin_ufku_hata_analizi.csv"), index=False)
    print("  -> Ufuk Hata Analizi Tablosu kaydedildi: reports/tables/tahmin_ufku_hata_analizi.csv")

    # D) SHAP AÇIKLANABİLİRLİK ANALİZİ ("NİYE ÖYLE OLUYOR?")
    print("\n  -> SHAP Açıklanabilirlik Analizi Hesaplanıyor...")
    explainer = shap.TreeExplainer(lgb_reg)
    shap_sample = X_test_pre[:1000]  # İlk 1000 örnek ile hesaplama hızı
    shap_values = explainer.shap_values(shap_sample)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_feature_idx = np.argsort(mean_abs_shap)[::-1]

    shap_table = []
    for rank, idx in enumerate(top_feature_idx[:10], start=1):
        feat_name = feature_cols_gop_oncesi[idx]
        imp = mean_abs_shap[idx]
        
        # Piyasa gerekçelendirmesi
        if "residual" in feat_name:
            reason = "Kalan yük arttıkça pahalı yakıtlı santraller devreye girer, SMF yükselir."
        elif "load" in feat_name:
            reason = "Yüksek sistem tüketimi marjinal maliyeti doğrudan yukarı çeker."
        elif "wind" in feat_name:
            reason = "Rüzgar üretimi arttıkça linyit/gaz santralleri kısılır, SMF düşer."
        elif "latest_known" in feat_name or "lag" in feat_name:
            reason = "Piyasanın otoregresif fiyat hafızası ve son bilinen rejim etkisi."
        elif "hour" in feat_name:
            reason = "Puant saatler (17:00-21:00) ile gece saatleri arasındaki talep farkı."
        else:
            reason = "Takvim ve döngüsel mevsimsellik bileşeni."

        shap_table.append({
            "Sıra": rank,
            "Özellik Adı": feat_name,
            "Ortalama Mutlak SHAP (TL)": round(imp, 2),
            "Piyasa Mantığı ve İktisadi Açıklama": reason
        })

    df_shap = pd.DataFrame(shap_table)
    df_shap.to_csv(os.path.join(tables_dir, "shap_onem_duzeyleri_ve_gerekceleri.csv"), index=False)
    print("  -> SHAP Analizi Tablosu kaydedildi: reports/tables/shap_onem_duzeyleri_ve_gerekceleri.csv")

    print("\n" + "=" * 95)
    print("  TÜM AŞAMALAR (2-7) BAŞARIYLA TAMAMLANDI VE TABLOLAR ÜRETİLDİ!")
    print("=" * 95)

if __name__ == "__main__":
    run_full_thesis_pipeline()
