"""
EPİAŞ Şeffaflık Platformu Resmi Veri Çekici Modülü
Doğrulanabilir, İzlenebilir ve Resmi API (eptr2) Üzerinden İndirme

İndirilen Resmi Seriler:
1. 'mcp-smp-imb': 
   - PTF (Piyasa Takas Fiyatı - TL/MWh)
   - SMF (Sistem Marjinal Fiyatı - TL/MWh)
   - Pozitif Dengesizlik Fiyatı (TL/MWh)
   - Negatif Dengesizlik Fiyatı (TL/MWh)
   - Resmi Sistem Durumu (Enerji Açığı, Enerji Fazlası, Dengede)
2. 'load-plan': 
   - TEİAŞ Gün Öncesi Yük Tahmin Planı (lep - MW)
3. 'wind-forecast': 
   - TEİAŞ Gün Öncesi Rüzgar Üretim Tahmini (forecast - MW)
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import eptr2
from dotenv import load_dotenv


def fetch_month_data(ep, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Belirli bir ay aralığı için EPİAŞ'tan 3 temel resmi seriyi çeker ve saatlik birleştirir.
    """
    # 1. Fiyat ve Dengesizlik Verisi
    df_imb = ep.call("mcp-smp-imb", start_date=start_str, end_date=end_str)
    df_imb["timestamp"] = pd.to_datetime(df_imb["date"])
    df_imb = df_imb[["timestamp", "ptf", "smf", "positiveImbalance", "negativeImbalance", "systemStatus"]].copy()

    # 2. TEİAŞ Resmi Yük Tahmini (Load Plan)
    try:
        df_load = ep.call("load-plan", start_date=start_str, end_date=end_str)
        df_load["timestamp"] = pd.to_datetime(df_load["date"])
        df_load = df_load[["timestamp", "lep"]].rename(columns={"lep": "load_forecast_mw"})
        df_merged = pd.merge(df_imb, df_load, on="timestamp", how="left")
    except Exception as e:
        print(f"      [Uyarı] load-plan çekilemedi ({start_str}): {e}")
        df_merged = df_imb.copy()
        df_merged["load_forecast_mw"] = np.nan

    # 3. TEİAŞ Resmi Rüzgar Tahmini (Wind Forecast - 10 dakikalıktan saatlik ortalamaya)
    try:
        df_wind = ep.call("wind-forecast", start_date=start_str, end_date=end_str)
        df_wind["timestamp"] = pd.to_datetime(df_wind["date"]).dt.floor("h")
        wind_hourly = df_wind.groupby("timestamp")["forecast"].mean().reset_index()
        wind_hourly.rename(columns={"forecast": "wind_forecast_mw"}, inplace=True)
        df_merged = pd.merge(df_merged, wind_hourly, on="timestamp", how="left")
    except Exception as e:
        print(f"      [Uyarı] wind-forecast çekilemedi ({start_str}): {e}")
        df_merged["wind_forecast_mw"] = np.nan

    return df_merged


def fetch_all_epias_official(start_year: int = 2024, end_year: int = 2025, end_month: int = 9, output_path: str = "data/raw/epias_official_market_data.parquet"):
    load_dotenv(".env")
    username = os.environ.get("EPTR_USERNAME")
    if not username:
        raise ValueError(".env dosyasında EPTR_USERNAME bulunamadı!")

    print("=" * 70)
    print("  EPİAŞ ŞEFFAFLIK PLATFORMU RESMİ VERİ İNDİRME SÜRECİ")
    print(f"  Hesap: {username[:4]}***")
    print("=" * 70)

    ep = eptr2.EPTR2(use_dotenv=True, dotenv_path=".env")
    all_chunks = []

    # Aylık döngü oluştur
    curr = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, end_month, 28)

    while curr < end_dt:
        # Ayın son gününü bul
        next_month = curr.replace(day=28) + timedelta(days=4)
        month_end = next_month - timedelta(days=next_month.day)
        
        start_str = curr.strftime("%Y-%m-%d")
        end_str = month_end.strftime("%Y-%m-%d")
        
        print(f"  -> İndiriliyor: {start_str} ile {end_str} arası...")
        try:
            chunk = fetch_month_data(ep, start_str, end_str)
            all_chunks.append(chunk)
            print(f"     Başarılı: {len(chunk)} saatlik satır çekildi.")
        except Exception as ex:
            print(f"     [Hata] {start_str} ayı çekilemedi: {ex}")

        curr = month_end + timedelta(days=1)
        time.sleep(0.3)  # API nezaketi / rate limiting koruması

    full_df = pd.concat(all_chunks, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_parquet(output_path)
    
    print("\n" + "=" * 70)
    print(f"  RESMİ VERİ BAŞARIYLA KAYDEDİLDİ: {output_path}")
    print(f"  Toplam Saatlik Satır: {len(full_df)}")
    print(f"  Tarih Aralığı: {full_df['timestamp'].min()} -> {full_df['timestamp'].max()}")
    print("  Sütunlar:", full_df.columns.tolist())
    print("=" * 70)

    return full_df


if __name__ == "__main__":
    fetch_all_epias_official()
