"""
Resmi EPİAŞ Veri Seti Doğrulama ve İnceleme Modülü (Data Provenance & Audit)
"""

import pandas as pd
import numpy as np


def verify_official_dataset(path: str = "data/raw/epias_official_market_data.parquet"):
    print("=" * 70)
    print("  RESMİ EPİAŞ VERİ SETİ DOĞRULAMA VE KALİTE KONTROL RAPORU")
    print("=" * 70)

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"\n1. GENEL BİLGİLER:")
    print(f"   Toplam Saatlik Gözlem: {len(df):,}")
    print(f"   Başlangıç Tarihi:      {df['timestamp'].min()}")
    print(f"   Bitiş Tarihi:          {df['timestamp'].max()}")
    print(f"   Eksik Değerler (NaN):")
    for col in df.columns:
        print(f"     - {col:20s}: {df[col].isna().sum()} eksik (%{df[col].isna().mean()*100:.2f})")

    # 2. Resmi Sistem Durumu (Yön) Dağılımı
    print(f"\n2. RESMİ SİSTEM YÖNÜ DAĞILIMI (systemStatus):")
    vc = df["systemStatus"].value_counts(dropna=False)
    vc_pct = df["systemStatus"].value_counts(normalize=True, dropna=False) * 100
    for status, count in vc.items():
        print(f"   - {status:16s}: {count:6d} saat (%{vc_pct[status]:.2f})")

    # Çoğunluk Sınıfı Baseline'ı:
    majority_class = vc.index[0]
    majority_acc = vc_pct.iloc[0]
    print(f"   ==> Çoğunluk Sınıfı (Majority Class Baseline): '{majority_class}' (%{majority_acc:.2f})")

    # 3. Fiyat İstatistikleri (PTF vs SMF)
    print(f"\n3. FİYAT DAĞILIMLARI (TL/MWh):")
    print(f"   PTF Ortalama: {df['ptf'].mean():.2f} TL | Medyan: {df['ptf'].median():.2f} TL | Min: {df['ptf'].min():.2f} | Max: {df['ptf'].max():.2f}")
    print(f"   SMF Ortalama: {df['smf'].mean():.2f} TL | Medyan: {df['smf'].median():.2f} TL | Min: {df['smf'].min():.2f} | Max: {df['smf'].max():.2f}")
    spread = df["smf"] - df["ptf"]
    print(f"   Yayılım (Spread = SMF - PTF): Ortalama = {spread.mean():.2f} TL | Std = {spread.std():.2f} TL")

    # 4. Rastgele Örnek Günlerin Doğrulaması (Kullanıcının Web Ekranıyla Kıyaslaması İçin)
    sample_dates = ["2024-01-15", "2024-07-10", "2025-02-14"]
    print(f"\n4. PLATFORMLA ELLE DOĞRULAMA KESİTLERİ (3 Farklı Gün - Saat 12:00 Değerleri):")
    print(f"   {'Tarih':12s} {'Saat':6s} {'PTF (TL)':10s} {'SMF (TL)':10s} {'Durum':16s} {'Yük Tahmini (MW)':18s} {'RES Tahmini (MW)':18s}")
    print("   " + "-" * 90)

    for d in sample_dates:
        row = df[(df["timestamp"].dt.strftime("%Y-%m-%d") == d) & (df["timestamp"].dt.hour == 12)]
        if not row.empty:
            r = row.iloc[0]
            print(f"   {d:12s} 12:00  {r['ptf']:10.2f} {r['smf']:10.2f} {str(r['systemStatus']):16s} {r['load_forecast_mw']:18.1f} {r['wind_forecast_mw']:18.1f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    verify_official_dataset()
