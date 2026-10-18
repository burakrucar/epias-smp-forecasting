"""
Resmi EPİAŞ Verisi ile Özellik Mühendisliği (Feature Engineering)
Seçenek C Kurgusu:
- Aşama 1: GÖP Öncesi (Saat 11:30/12:00 - PTF Bilinmiyor)
- Aşama 2: GÖP Sonrası (Saat 14:00 - PTF Açıklandı ve Biliniyor)
"""

import pandas as pd
import numpy as np


def build_thesis_feature_matrix(df: pd.DataFrame, lag_offset: int = 14) -> tuple[pd.DataFrame, list, list]:
    """
    Resmi EPİAŞ veri setinden hem GÖP Öncesi (PTF'siz) hem GÖP Sonrası (PTF'li)
    sızıntısız özellik setlerini türetir.
    
    lag_offset: D-1 saat 11:30'da bilinen en son piyasa saatinin teslimat saatine uzaklığı.
      - 14: D-1 saat 10:00 (varsayılan baz kurgu)
      - 17: D-1 saat 07:00 (muhafazakar gecikme)
      - 24: D-1 saat 00:00 (gece yarısı kapanışı)
      - 38: D-2 saat 10:00 (tam kesinleşmiş D-2 kurgusu)
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    hours = df["timestamp"].dt.hour.values
    dows = df["timestamp"].dt.dayofweek.values
    months = df["timestamp"].dt.month.values

    # 1. Deterministik / Takvim Özellikleri
    df["hour"] = hours
    df["dow"] = dows
    df["month"] = months
    df["is_weekend"] = (dows >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * months / 12.0)

    # TAMAMEN NEDENSEL (STRICTLY CAUSAL) DOLDURMA:
    # Sadece ve sadece geçmişten gelen son bilinen değeri ileri taşır (ffill).
    # Gelecekten geriye bilgi taşıyan enterpolasyon ve bfill kesinlikle kullanılmaz.
    if "load_forecast_mw" in df.columns:
        df["load_forecast_mw"] = df["load_forecast_mw"].ffill()
    if "wind_forecast_mw" in df.columns:
        df["wind_forecast_mw"] = df["wind_forecast_mw"].ffill()

    # 2. TEİAŞ Resmi Gün Öncesi Tahminleri (Tahmin anında kesin bilinenler)
    # Artık Yük Tahmini (Residual Load Forecast = Yük - Rüzgar)
    df["forecast_residual_load_mw"] = df["load_forecast_mw"] - df["wind_forecast_mw"]
    df["forecast_wind_ratio"] = df["wind_forecast_mw"] / np.maximum(df["load_forecast_mw"], 1.0)

    # 3. Dinamik Gate-Closure Gecikmeleri (lag = lag_offset + h)
    latest_smf = np.full(n, np.nan)
    latest_ptf = np.full(n, np.nan)
    latest_status = np.full(n, np.nan)

    # 3 Sınıflı Haritalama: Enerji Fazlası = -1, Dengede = 0, Enerji Açığı = 1
    status_map = {"Enerji Fazlası": -1, "Dengede": 0, "Enerji Açığı": 1}
    numeric_status = df["systemStatus"].map(status_map).values

    for i in range(n):
        lag = lag_offset + hours[i]
        if i >= lag:
            latest_smf[i] = df["smf"].iloc[i - lag]
            latest_ptf[i] = df["ptf"].iloc[i - lag]
            latest_status[i] = numeric_status[i - lag]

    df["latest_known_smf"] = latest_smf
    df["latest_known_ptf"] = latest_ptf
    df["latest_known_status"] = latest_status

    # 4. Tarihsel Çok Adımlı Sabit Gecikmeler
    # lag 48 (2 gün önce), lag 72 (3 gün önce), lag 168 (1 hafta önceki aynı saat)
    for lag in [48, 72, 168]:
        df[f"smf_lag_{lag}"] = df["smf"].shift(lag)
        df[f"ptf_lag_{lag}"] = df["ptf"].shift(lag)
        df[f"status_lag_{lag}"] = pd.Series(numeric_status).shift(lag).values
        df[f"spread_lag_{lag}"] = (df["smf"] - df["ptf"]).shift(lag)
        df[f"load_lag_{lag}"] = df["load_forecast_mw"].shift(lag)

    # 5. Son 7 Günün Aynı Saat Ortalaması (Rolling 7-Day Same-Hour Persistence)
    # lag 48 ve üstü haftalık ortalama için sızıntısız vekil: (lag_48 + lag_72 + lag_96 + lag_120 + lag_144 + lag_168) / 6
    same_hour_lags = [df["smf"].shift(l) for l in [48, 72, 96, 120, 144, 168]]
    df["smf_same_hour_7d_mean"] = pd.concat(same_hour_lags, axis=1).mean(axis=1)

    # 6. D-2 Hareketli İstatistikler
    df["smf_mean_d2"] = df["smf"].shift(48).rolling(24).mean()
    df["smf_std_d2"] = df["smf"].shift(48).rolling(24).std()
    df["ptf_mean_d2"] = df["ptf"].shift(48).rolling(24).mean()

    # 7. Hedef Değişkenler
    df["target_direction"] = numeric_status  # -1, 0, 1
    df["target_smf"] = df["smf"].values      # TL/MWh

    # NaN satırları (ilk 168 saat / 7 gün) filtrele
    df_clean = df.dropna().reset_index(drop=True)

    # -------------------------------------------------------------
    # ÖZELLİK LİSTELERİ:
    # -------------------------------------------------------------
    # A) GÖP Öncesi (Saat 11:30 - PTF Bilinmiyor)
    cols_gop_oncesi = [
        "hour", "dow", "month", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "load_forecast_mw", "wind_forecast_mw", "forecast_residual_load_mw", "forecast_wind_ratio",
        "latest_known_smf", "latest_known_ptf", "latest_known_status",
        "smf_lag_48", "ptf_lag_48", "status_lag_48", "spread_lag_48",
        "smf_lag_72", "ptf_lag_72", "status_lag_72",
        "smf_lag_168", "ptf_lag_168", "status_lag_168",
        "smf_same_hour_7d_mean", "smf_mean_d2", "smf_std_d2", "ptf_mean_d2"
    ]

    # B) GÖP Sonrası (Saat 14:00 - PTF Açıklandı)
    # GÖP Öncesi tüm özelliklere ek olarak teslimat saatinin kesinleşmiş PTF'si ve PTF bazlı göstergeler eklenir:
    cols_gop_sonrasi = cols_gop_oncesi + ["ptf"]

    return df_clean, cols_gop_oncesi, cols_gop_sonrasi
