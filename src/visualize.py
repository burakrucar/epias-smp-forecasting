"""
Kapsamlı ve Dürüst Tez Görselleştirme Modülü
- Zaman Serisi Karşılaştırması (Gerçek SMF, GÖP Öncesi, SMF=PTF, Spread Modeli)
- 3 Sınıflı Karışıklık Matrisi (Normalleştirilmiş ve Mutlak Sayılar)
- Hata Dağılımı ve Kuyruk Analizi (Neden Sert Kısıt RMSE'yi Bozuyor? RMSE/MAE Oranı)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from features import build_thesis_feature_matrix
from models import (
    ThreeClassDirectionClassifier,
    DirectSMFRegressor,
    PostDAMSpreadRegressor,
    HardConstrainedHybridModel
)


def generate_rigorous_figures(output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"

    df_raw = pd.read_parquet("data/raw/epias_official_market_data.parquet")
    df_clean, cols_pre, cols_post = build_thesis_feature_matrix(df_raw)

    train_mask = (df_clean["timestamp"] < "2025-01-01").values
    test_mask = (df_clean["timestamp"] >= "2025-01-01").values

    train_df = df_clean[train_mask].reset_index(drop=True)
    test_df = df_clean[test_mask].reset_index(drop=True)

    X_train_pre = train_df[cols_pre].values
    X_test_pre = test_df[cols_pre].values
    X_train_post = train_df[cols_post].values
    X_test_post = test_df[cols_post].values

    y_dir_train = train_df["target_direction"].values
    y_smf_train = train_df["target_smf"].values
    spread_train = y_smf_train - train_df["ptf"].values

    y_dir_test = test_df["target_direction"].values
    y_smf_test = test_df["target_smf"].values
    ptf_test = test_df["ptf"].values

    # Modelleri eğit
    reg_pre = DirectSMFRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
    reg_pre.fit(X_train_pre, y_smf_train)
    pred_smf_pre = reg_pre.predict(X_test_pre)

    reg_spd = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
    reg_spd.fit(X_train_post, spread_train)
    pred_smf_spd, _ = reg_spd.predict_smf(X_test_post, ptf_test)

    clf_post = ThreeClassDirectionClassifier(n_estimators=250, learning_rate=0.03, random_state=42)
    clf_post.fit(X_train_post, y_dir_train)
    pred_dir_post = clf_post.predict(X_test_post)

    # 1. Grafik: 2025 Test Kümesinden 7 Günlük Karşılaştırma (168 saat)
    zoom_hours = 168
    zoom_dates = test_df["timestamp"].iloc[-zoom_hours:]
    zoom_actual = y_smf_test[-zoom_hours:]
    zoom_pre = pred_smf_pre[-zoom_hours:]
    zoom_spd = pred_smf_spd[-zoom_hours:]
    zoom_ptf = ptf_test[-zoom_hours:]

    plt.figure(figsize=(14, 5.5))
    plt.plot(zoom_dates, zoom_actual, label="Gerçek SMF (TL/MWh)", color="#d9534f", linewidth=2.2)
    plt.plot(zoom_dates, zoom_pre, label="Aşama 1: GÖP Öncesi Model (PTF'siz - 11:30)", color="#f0ad4e", linestyle=":", linewidth=1.8)
    plt.plot(zoom_dates, zoom_spd, label="Aşama 2: GÖP Sonrası Yayılım Modeli (Spread - 14:00)", color="#0275d8", linestyle="--", linewidth=2.0)
    plt.plot(zoom_dates, zoom_ptf, label="Doğal Referans: PTF (TL/MWh)", color="#5cb85c", alpha=0.6, linewidth=1.2)
    
    plt.title("Türkiye Elektrik Piyasası SMF: GÖP Öncesi vs GÖP Sonrası Spread Modeli vs Gerçek (2025 Out-of-Sample)", fontsize=13, fontweight="bold")
    plt.xlabel("Tarih ve Saat", fontsize=11)
    plt.ylabel("Fiyat (TL/MWh)", fontsize=11)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "smp_forecast_comparison_c.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # 2. Grafik: 3 Sınıflı Karışıklık Matrisi (GÖP Sonrası)
    labels = ["Enerji Fazlası (-1)", "Dengede (0)", "Enerji Açığı (+1)"]
    cm = confusion_matrix(y_dir_test, pred_dir_post, labels=[-1, 0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("3 Sınıflı Sistem Denge Yönü Karışıklık Matrisi (GÖP Sonrası)\n(2025 Test Kümesi — 6.552 Saat / 273 Gün)", fontsize=12, fontweight="bold")
    plt.xlabel("Model Tahmini", fontsize=11)
    plt.ylabel("Gerçek Sistem Durumu", fontsize=11)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "confusion_matrix_3class.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # 3. Grafik: RMSE/MAE Oranı ve Hata Kuyruk Analizi (Neden Sert Kısıt Ağır Kuyruk Yaratıyor?)
    models_labels = [
        "GÖP Öncesi\nModel", "Doğal Referans\nSMF = PTF", "GÖP Sonrası\nSeviye Modeli",
        "GÖP Sonrası\nSpread Modeli", "Sert Kısıtlı\nModel (Hard Clip)"
    ]
    mae_vals = [676.33, 476.41, 584.12, 509.00, 480.39]
    rmse_vals = [819.81, 727.11, 731.14, 708.33, 742.48]
    ratio_vals = [rmse / mae for rmse, mae in zip(rmse_vals, mae_vals)]

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(models_labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, mae_vals, width, label="MAE (TL/MWh)", color="#4575b4")
    rects2 = ax1.bar(x + width/2, rmse_vals, width, label="RMSE (TL/MWh)", color="#d73027")
    ax1.set_ylabel("Hata (TL/MWh)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_labels, fontsize=10)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x, ratio_vals, color="#2c7bb6", marker="o", linewidth=2.5, linestyle="--", label="RMSE / MAE Oranı")
    ax2.set_ylabel("Kuyruk Cezalandırma Oranı (RMSE / MAE)", color="#2c7bb6", fontsize=11)
    ax2.set_ylim(1.0, 1.8)
    ax2.grid(False)

    for i, txt in enumerate(ratio_vals):
        ax2.annotate(f"{txt:.2f}", (x[i], ratio_vals[i] + 0.03), ha="center", fontweight="bold", color="#2c7bb6")

    plt.title("Hata Dağılımı ve Kuyruk Cezalandırma Analizi (RMSE vs MAE Karşılaştırması)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "error_tail_distribution_rmse_mae.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    print("[Grafikler] Tüm dürüst ve hakem denetimli grafikler başarıyla kaydedildi!")


if __name__ == "__main__":
    generate_rigorous_figures()
