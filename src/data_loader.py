"""
EPİAŞ ve Piyasa Veri Yükleyici Modülü
- Gerçek EPİAŞ veri setini (data/raw/epias_real_data.parquet) yükleme
- eptr2 ile EPİAŞ Şeffaflık Platformu API'sinden canlı veri çekme
- Geliştirme için sentetik veri üretme
"""

import os
import pandas as pd
import numpy as np


def load_real_epias_data(filepath: str = "data/raw/epias_real_data.parquet") -> pd.DataFrame:
    """
    Yerel EPİAŞ gerçek piyasa veri setini yükler.
    Tarih Aralığı: 2024-01-03 - 2025-09-09 (~14.783 saat)
    İçerik: PTF, SMF, Tüketim, Üretim Dağılımı (Rüzgar, Güneş, Hidro, Termik),
            Hava Durumu Değişkenleri (Sıcaklık, Rüzgar, Güneş Radyasyonu).
    """
    if not os.path.exists(filepath):
        # Üst dizin veya alternatif yol kontrolü
        alt_path = os.path.join(os.path.dirname(__file__), "..", filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"EPİAŞ veri dosyası bulunamadı: {filepath}")

    df = pd.read_parquet(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Sistem Dengeleme Yönünü EPİAŞ kurallarına göre türet:
    # Spread = SMF - PTF
    # > 1 TL ise Sistem Açığı (+1, YAL talimatı)
    # < -1 TL ise Sistem Fazlası (-1, YAT talimatı)
    # Aksi halde Dengede (0)
    spread = df["smf"] - df["ptf"]
    df["direction"] = np.where(spread > 1.0, 1, np.where(spread < -1.0, -1, 0))
    df["spread"] = spread

    return df


def load_from_eptr2(start_date: str, end_date: str, username: str = None, password: str = None) -> pd.DataFrame:
    """
    eptr2 kütüphanesini kullanarak EPİAŞ Şeffaflık Platformundan canlı veri çeker.
    Ücretsiz EPİAŞ hesabı gerektirir.
    """
    import eptr2
    if username and password:
        os.environ["EPTR_USERNAME"] = username
        os.environ["EPTR_PASSWORD"] = password
        
    ep = eptr2.EPTR2()
    mcp_res = ep.call("mcp", start_date=start_date, end_date=end_date)
    df_mcp = pd.DataFrame(mcp_res["items"])
    
    smp_res = ep.call("smp", start_date=start_date, end_date=end_date)
    df_smp = pd.DataFrame(smp_res["items"])
    
    df_merged = pd.merge(df_mcp, df_smp, on=["date"], suffixes=("_mcp", "_smp"))
    return df_merged
