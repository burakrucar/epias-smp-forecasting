"""
Kapsamlı ve Katı Ekonometrik Değerlendirme Modülü (Claude Hakem Raporu Düzeltmeleri)
- 3 Aşamalı Bölünme (Train 9 Ay / Val 3 Ay / Test 9 Ay - 273 Gün)
- Kontrollü Ablasyon Deneyi (Controlled Ablation: Yalnızca PTF farkı)
- GÖP Sonrası Doğal Referanslar (SMF = PTF ve SMF = PTF + MeanSpread)
- Günlük Düzeyde Diebold-Mariano Testi (N = 273 Gün, Hem MAE Hem MSE Kayıpları ile)
- Günlük Düzeyde Wilcoxon Signed-Rank Testi (Sınıflandırma Karşılaştırması)
- Sınıf Bazında Detaylı Metrikler (Precision, Recall, F1)
- Sert Kısıtın RMSE Bozucu Etkisinin Şeffaf Raporlanması
"""

import os
import sys
import numpy as np
import pandas as pd

from features import build_thesis_feature_matrix
from models import (
    ThreeClassDirectionClassifier,
    DirectSMFRegressor,
    PostDAMSpreadRegressor,
    HardConstrainedHybridModel
)
from metrics import calculate_regression_metrics, calculate_classification_metrics
from significance_tests import (
    daily_diebold_mariano_test,
    daily_classification_wilcoxon_test,
    get_detailed_class_metrics
)


def run_rigorous_evaluation():
    print("=" * 85)
    print("  KAPSAMLI VE KATI EKONOMETRİK TEZ BENCHMARK'I")
    print("  Veri: EPİAŞ Resmi Şeffaflık Platformu (15.336 Saat / 21 Ay)")
    print("=" * 85)

    # 1. Veri Yükleme ve Özellikler
    data_path = "data/raw/epias_official_market_data.parquet"
    df_raw = pd.read_parquet(data_path)
    df_clean, cols_pre, cols_post = build_thesis_feature_matrix(df_raw)

    # 2. Üçlü Zaman Serisi Bölümlemesi
    train_mask = (df_clean["timestamp"] < "2024-10-01").values
    val_mask = ((df_clean["timestamp"] >= "2024-10-01") & (df_clean["timestamp"] < "2025-01-01")).values
    test_mask = (df_clean["timestamp"] >= "2025-01-01").values

    train_df = df_clean[train_mask].reset_index(drop=True)
    val_df = df_clean[val_mask].reset_index(drop=True)
    test_df = df_clean[test_mask].reset_index(drop=True)

    # Final model için Train+Val (2024 Yılının Tamamı)
    full_train_df = df_clean[train_mask | val_mask].reset_index(drop=True)

    print(f"\n[1/5] Veri Bölümleme Mimarisi:")
    print(f"      1. Aşama Eğitim (Train):        {len(train_df):5d} saat  ({train_df['timestamp'].min().strftime('%Y-%m-%d')} - {train_df['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"      2. Aşama Doğrulama (Val):       {len(val_df):5d} saat  ({val_df['timestamp'].min().strftime('%Y-%m-%d')} - {val_df['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"      Final Eğitim (Train + Val):     {len(full_train_df):5d} saat  (2024 Yılı Tamamı: 8.616 saat)")
    print(f"      3. Aşama Test (Out-of-Sample):  {len(test_df):5d} saat  (2025 Yılı: 273 Gün / 6.552 saat)")

    # Hedefler ve Değişkenler
    y_dir_full = full_train_df["target_direction"].values
    y_smf_full = full_train_df["target_smf"].values
    spread_full = y_smf_full - full_train_df["ptf"].values

    y_dir_test = test_df["target_direction"].values
    y_smf_test = test_df["target_smf"].values
    ptf_test = test_df["ptf"].values
    spread_test = y_smf_test - ptf_test

    mean_spread_train = np.mean(spread_full)

    # -------------------------------------------------------------
    # 3. REFERANS (BASELINE) MODELLERİ
    # -------------------------------------------------------------
    print("\n[2/5] Referans (Benchmark) Modelleri Hesaplanıyor...")
    
    # Sınıflandırma Baseline'ları
    majority_pred = np.ones_like(y_dir_test)  # Enerji Açığı = 1
    lag13_dir = test_df["latest_known_status"].values
    lag48_dir = test_df["status_lag_48"].values
    lag168_dir = test_df["status_lag_168"].values

    # Regresyon Baseline'ları (GÖP Öncesi)
    lag13_smf = test_df["latest_known_smf"].values
    lag48_smf = test_df["smf_lag_48"].values
    lag168_smf = test_df["smf_lag_168"].values
    r7d_smf = test_df["smf_same_hour_7d_mean"].values

    # GÖP Sonrası Doğal Referanslar (PTF Bilinirken)
    base_smf_eq_ptf = ptf_test.copy()                                    # SMF = PTF (Spread = 0)
    base_smf_ptf_plus_mean = ptf_test + mean_spread_train               # SMF = PTF + MeanSpread

    # -------------------------------------------------------------
    # 4. AŞAMA 1: GÖP ÖNCESİ (Saat 11:30 - PTF Bilinmiyor)
    # -------------------------------------------------------------
    print("\n[3/5] Aşama 1 Modelleri Eğitiliyor (GÖP Öncesi — PTF Kesinlikle Yok)...")
    X_train_pre = full_train_df[cols_pre].values
    X_test_pre = test_df[cols_pre].values

    clf_pre = ThreeClassDirectionClassifier(n_estimators=250, learning_rate=0.03, random_state=42)
    clf_pre.fit(X_train_pre, y_dir_full)
    pred_dir_pre = clf_pre.predict(X_test_pre)

    reg_pre = DirectSMFRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
    reg_pre.fit(X_train_pre, y_smf_full)
    pred_smf_pre = reg_pre.predict(X_test_pre)

    # -------------------------------------------------------------
    # 5. AŞAMA 2: GÖP SONRASI (Saat 14:00 - PTF Biliniyor)
    # -------------------------------------------------------------
    print("\n[4/5] Aşama 2 Modelleri Eğitiliyor (GÖP Sonrası — Kontrollü Ablasyon ve Spread Modeli)...")
    X_train_post = full_train_df[cols_post].values
    X_test_post = test_df[cols_post].values

    # Ablasyon Yön Sınıflandırıcısı (Aynı özellik seti + PTF)
    clf_post = ThreeClassDirectionClassifier(n_estimators=250, learning_rate=0.03, random_state=42)
    clf_post.fit(X_train_post, y_dir_full)
    pred_dir_post = clf_post.predict(X_test_post)

    # Doğrudan Seviye Regresörü (Ablasyon: Aynı mimari + PTF)
    reg_post_direct = DirectSMFRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
    reg_post_direct.fit(X_train_post, y_smf_full)
    pred_smf_post_direct = reg_post_direct.predict(X_test_post)

    # Yayılım (Spread: SMF - PTF) Regresörü (Enflasyon ve Fiyat Seviyesi Kaymasını Çözen Model)
    reg_post_spread = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
    reg_post_spread.fit(X_train_post, spread_full)
    pred_smf_post_spread, pred_spread_only = reg_post_spread.predict_smf(X_test_post, ptf_test)

    # Sert Kısıtlı Model (Hata Patlaması Analizi İçin)
    # Modelin kendi tahmin ettiği pred_dir_post kullanılır (test gerçek yönü ASLA kullanılmaz!)
    probs_post = clf_post.predict_proba(X_train_post)
    reg_aug = DirectSMFRegressor(n_estimators=250, learning_rate=0.03, random_state=42)
    reg_aug.fit(np.hstack([X_train_post, probs_post]), y_smf_full)

    hard_model = HardConstrainedHybridModel(clf_post, reg_aug)
    _, pred_smf_hard, _ = hard_model.predict(X_test_post, ptf_test)

    # -------------------------------------------------------------
    # 6. KAPSAMLI METRİK HESAPLAMALARI
    # -------------------------------------------------------------
    print("\n[5/5] Sonuçlar Hesaplanıyor ve Hipotez Testleri Koşturuluyor...\n")

    # TABLO 1: HEDEF 1 - SİSTEM YÖNÜ SINIFLANDIRMA (3 Sınıflı)
    res_maj_c = calculate_classification_metrics(y_dir_test, majority_pred)
    res_l13_c = calculate_classification_metrics(y_dir_test, lag13_dir)
    res_l48_c = calculate_classification_metrics(y_dir_test, lag48_dir)
    res_168_c = calculate_classification_metrics(y_dir_test, lag168_dir)
    res_pre_c = calculate_classification_metrics(y_dir_test, pred_dir_pre)
    res_pst_c = calculate_classification_metrics(y_dir_test, pred_dir_post)

    df_clf_table = pd.DataFrame([
        {"Model / Aşama": "1. Majority Class (Her Saate Enerji Açığı)", "Accuracy": f"%{res_maj_c['Accuracy']*100:.2f}", "Macro-F1": f"{res_maj_c['Macro_F1']:.3f}", "Weighted-F1": f"{res_maj_c['Weighted_F1']:.3f}"},
        {"Model / Aşama": "2. Dinamik Persistence (Lag 13+h)", "Accuracy": f"%{res_l13_c['Accuracy']*100:.2f}", "Macro-F1": f"{res_l13_c['Macro_F1']:.3f}", "Weighted-F1": f"{res_l13_c['Weighted_F1']:.3f}"},
        {"Model / Aşama": "3. Lag-48 (2 Gün Önceki Aynı Saat)", "Accuracy": f"%{res_l48_c['Accuracy']*100:.2f}", "Macro-F1": f"{res_l48_c['Macro_F1']:.3f}", "Weighted-F1": f"{res_l48_c['Weighted_F1']:.3f}"},
        {"Model / Aşama": "4. Lag-168 (Geçen Hafta Aynı Saat)", "Accuracy": f"%{res_168_c['Accuracy']*100:.2f}", "Macro-F1": f"{res_168_c['Macro_F1']:.3f}", "Weighted-F1": f"{res_168_c['Weighted_F1']:.3f}"},
        {"Model / Aşama": "5. AŞAMA 1: GÖP Öncesi Model (PTF'siz - 11:30)", "Accuracy": f"%{res_pre_c['Accuracy']*100:.2f}", "Macro-F1": f"{res_pre_c['Macro_F1']:.3f}", "Weighted-F1": f"{res_pre_c['Weighted_F1']:.3f}"},
        {"Model / Aşama": "6. AŞAMA 2: GÖP Sonrası Model (PTF'li - 14:00)", "Accuracy": f"%{res_pst_c['Accuracy']*100:.2f}", "Macro-F1": f"{res_pst_c['Macro_F1']:.3f}", "Weighted-F1": f"{res_pst_c['Weighted_F1']:.3f}"}
    ])

    print("=" * 85)
    print("   TABLO 1: 3 SINIFLI SİSTEM DENGELENME YÖNÜ TAHMİNİ (2025 TEST — 273 GÜN)")
    print("=" * 85)
    print(df_clf_table.to_string(index=False))

    # SINIF BAZINDA DETAYLI DAĞILIM (Aşama 1 vs Aşama 2)
    print("\n   --- AŞAMA 1 (GÖP ÖNCESİ) SINIF BAZINDA PERFORMANS ---")
    print(get_detailed_class_metrics(y_dir_test, pred_dir_pre).to_string(index=False))

    print("\n   --- AŞAMA 2 (GÖP SONRASI) SINIF BAZINDA PERFORMANS ---")
    print(get_detailed_class_metrics(y_dir_test, pred_dir_post).to_string(index=False))

    # TABLO 2: HEDEF 2 - SMF FİYAT TAHMİNİ (REGRESYON - TL/MWh)
    res_l13_r = calculate_regression_metrics(y_smf_test, lag13_smf)
    res_l48_r = calculate_regression_metrics(y_smf_test, lag48_smf)
    res_168_r = calculate_regression_metrics(y_smf_test, lag168_smf)
    res_r7d_r = calculate_regression_metrics(y_smf_test, r7d_smf)

    res_b2a_r = calculate_regression_metrics(y_smf_test, base_smf_eq_ptf)
    res_b2b_r = calculate_regression_metrics(y_smf_test, base_smf_ptf_plus_mean)

    res_pre_r = calculate_regression_metrics(y_smf_test, pred_smf_pre)
    res_pst_dir_r = calculate_regression_metrics(y_smf_test, pred_smf_post_direct)
    res_pst_spd_r = calculate_regression_metrics(y_smf_test, pred_smf_post_spread)
    res_pst_hrd_r = calculate_regression_metrics(y_smf_test, pred_smf_hard)

    df_reg_table = pd.DataFrame([
        # Naif Referanslar (PTF Bilinmezken)
        {"Model / Kurgu": "1. Dinamik Persistence (Lag 13+h)", "MAE (TL)": f"{res_l13_r['MAE']:.2f}", "RMSE (TL)": f"{res_l13_r['RMSE']:.2f}", "RMSE/MAE": f"{res_l13_r['RMSE']/res_l13_r['MAE']:.2f}", "WAPE (%)": f"%{res_l13_r['WAPE']*100:.2f}", "R2": f"{res_l13_r['R2']:.3f}"},
        {"Model / Kurgu": "2. Lag-48 (2 Gün Önce)", "MAE (TL)": f"{res_l48_r['MAE']:.2f}", "RMSE (TL)": f"{res_l48_r['RMSE']:.2f}", "RMSE/MAE": f"{res_l48_r['RMSE']/res_l48_r['MAE']:.2f}", "WAPE (%)": f"%{res_l48_r['WAPE']*100:.2f}", "R2": f"{res_l48_r['R2']:.3f}"},
        {"Model / Kurgu": "3. Lag-168 (Geçen Hafta)", "MAE (TL)": f"{res_168_r['MAE']:.2f}", "RMSE (TL)": f"{res_168_r['RMSE']:.2f}", "RMSE/MAE": f"{res_168_r['RMSE']/res_168_r['MAE']:.2f}", "WAPE (%)": f"%{res_168_r['WAPE']*100:.2f}", "R2": f"{res_168_r['R2']:.3f}"},
        {"Model / Kurgu": "4. Rolling 7-Gün Aynı Saat Ort.", "MAE (TL)": f"{res_r7d_r['MAE']:.2f}", "RMSE (TL)": f"{res_r7d_r['RMSE']:.2f}", "RMSE/MAE": f"{res_r7d_r['RMSE']/res_r7d_r['MAE']:.2f}", "WAPE (%)": f"%{res_r7d_r['WAPE']*100:.2f}", "R2": f"{res_r7d_r['R2']:.3f}"},
        
        # AŞAMA 1: GÖP Öncesi Model
        {"Model / Kurgu": "5. AŞAMA 1: GÖP Öncesi (PTF'siz Model)", "MAE (TL)": f"{res_pre_r['MAE']:.2f}", "RMSE (TL)": f"{res_pre_r['RMSE']:.2f}", "RMSE/MAE": f"{res_pre_r['RMSE']/res_pre_r['MAE']:.2f}", "WAPE (%)": f"%{res_pre_r['WAPE']*100:.2f}", "R2": f"{res_pre_r['R2']:.3f}"},

        # GÖP Sonrası Doğal Referanslar
        {"Model / Kurgu": "6. Doğal Referans: SMF = PTF", "MAE (TL)": f"{res_b2a_r['MAE']:.2f}", "RMSE (TL)": f"{res_b2a_r['RMSE']:.2f}", "RMSE/MAE": f"{res_b2a_r['RMSE']/res_b2a_r['MAE']:.2f}", "WAPE (%)": f"%{res_b2a_r['WAPE']*100:.2f}", "R2": f"{res_b2a_r['R2']:.3f}"},
        {"Model / Kurgu": "7. Doğal Referans: SMF = PTF + Ortalama Spread", "MAE (TL)": f"{res_b2b_r['MAE']:.2f}", "RMSE (TL)": f"{res_b2b_r['RMSE']:.2f}", "RMSE/MAE": f"{res_b2b_r['RMSE']/res_b2b_r['MAE']:.2f}", "WAPE (%)": f"%{res_b2b_r['WAPE']*100:.2f}", "R2": f"{res_b2b_r['R2']:.3f}"},

        # AŞAMA 2: GÖP Sonrası Modeller
        {"Model / Kurgu": "8. AŞAMA 2: Seviye Modeli (Ablasyon + PTF)", "MAE (TL)": f"{res_pst_dir_r['MAE']:.2f}", "RMSE (TL)": f"{res_pst_dir_r['RMSE']:.2f}", "RMSE/MAE": f"{res_pst_dir_r['RMSE']/res_pst_dir_r['MAE']:.2f}", "WAPE (%)": f"%{res_pst_dir_r['WAPE']*100:.2f}", "R2": f"{res_pst_dir_r['R2']:.3f}"},
        {"Model / Kurgu": "9. AŞAMA 2: Yayılım Modeli (Spread Regressor)", "MAE (TL)": f"{res_pst_spd_r['MAE']:.2f}", "RMSE (TL)": f"{res_pst_spd_r['RMSE']:.2f}", "RMSE/MAE": f"{res_pst_spd_r['RMSE']/res_pst_spd_r['MAE']:.2f}", "WAPE (%)": f"%{res_pst_spd_r['WAPE']*100:.2f}", "R2": f"{res_pst_spd_r['R2']:.3f}"},
        {"Model / Kurgu": "10. AŞAMA 2: Sert Kısıtlı Model (Hard Constraint)", "MAE (TL)": f"{res_pst_hrd_r['MAE']:.2f}", "RMSE (TL)": f"{res_pst_hrd_r['RMSE']:.2f}", "RMSE/MAE": f"{res_pst_hrd_r['RMSE']/res_pst_hrd_r['MAE']:.2f}", "WAPE (%)": f"%{res_pst_hrd_r['WAPE']*100:.2f}", "R2": f"{res_pst_hrd_r['R2']:.3f}"}
    ])

    print("\n" + "=" * 85)
    print("   TABLO 2: SMF FİYAT TAHMİNİ (REGRESYON - TL/MWh - 2025 TEST — 273 GÜN)")
    print("=" * 85)
    print(df_reg_table.to_string(index=False))

    # GÜNLÜK DÜZEYDE DIEBOLD-MARIANO VE WILCOXON TESTLERİ
    print("\n" + "=" * 85)
    print("   İSTATİSTİKSEL HİPOTEZ TESTLERİ (GÜNLÜK AGREGASYON — N = 273 GÜN)")
    print("=" * 85)

    # 1. GÖP Öncesi vs GÖP Sonrası (PTF'nin Marjinal Katkısı)
    dm_mae_pre_post = daily_diebold_mariano_test(y_smf_test, pred_smf_pre, pred_smf_post_direct, power=1)
    dm_mse_pre_post = daily_diebold_mariano_test(y_smf_test, pred_smf_pre, pred_smf_post_direct, power=2)

    print(f"  1. PTF'nin Marjinal Bilgi Katkısı (GÖP Öncesi vs GÖP Sonrası Seviye Modeli):")
    print(f"     - Günlük MAE Kaybı ile DM Testi: DM = {dm_mae_pre_post['DM_Stat']:.3f} | p = {dm_mae_pre_post['P_Value']:.4e} {'[Anlamlı p < 0.05]' if dm_mae_pre_post['Significant_at_05'] else '[Anlamsız]'}")
    print(f"     - Günlük MSE Kaybı ile DM Testi: DM = {dm_mse_pre_post['DM_Stat']:.3f} | p = {dm_mse_pre_post['P_Value']:.4e} {'[Anlamlı p < 0.05]' if dm_mse_pre_post['Significant_at_05'] else '[Anlamsız]'}")

    # 2. Doğal Referans (SMF = PTF) vs GÖP Sonrası Spread Modeli
    dm_mae_ptf_spd = daily_diebold_mariano_test(y_smf_test, base_smf_eq_ptf, pred_smf_post_spread, power=1)
    dm_mse_ptf_spd = daily_diebold_mariano_test(y_smf_test, base_smf_eq_ptf, pred_smf_post_spread, power=2)
    print(f"\n  2. Doğal Referans (SMF = PTF) vs GÖP Sonrası Spread Modeli:")
    print(f"     - Günlük MAE Kaybı ile DM Testi: DM = {dm_mae_ptf_spd['DM_Stat']:.3f} | p = {dm_mae_ptf_spd['P_Value']:.4e} {'[Anlamlı p < 0.05]' if dm_mae_ptf_spd['Significant_at_05'] else '[Anlamsız]'}")
    print(f"     - Günlük MSE Kaybı ile DM Testi: DM = {dm_mse_ptf_spd['DM_Stat']:.3f} | p = {dm_mse_ptf_spd['P_Value']:.4e} {'[Anlamlı p < 0.05]' if dm_mse_ptf_spd['Significant_at_05'] else '[Anlamsız]'}")

    # 3. Sert Kısıtın Hata Davranışı Analizi (Seviye Modeli vs Sert Kısıtlı Model)
    dm_mae_dir_hrd = daily_diebold_mariano_test(y_smf_test, pred_smf_post_direct, pred_smf_hard, power=1)
    dm_mse_dir_hrd = daily_diebold_mariano_test(y_smf_test, pred_smf_post_direct, pred_smf_hard, power=2)
    print(f"\n  3. Sert Kısıtın (Hard Constraint) Davranış Analizi (Seviye Modeli vs Sert Kısıt):")
    print(f"     - Günlük MAE Kaybı (Sert kısıt MAE'yi düşürüyor mu?): DM = {dm_mae_dir_hrd['DM_Stat']:.3f} | p = {dm_mae_dir_hrd['P_Value']:.4e} {'[MAE Azaldı]' if dm_mae_dir_hrd['DM_Stat'] > 0 else '[MAE Arttı]'}")
    print(f"     - Günlük MSE Kaybı (Sert kısıt RMSE'yi bozuyor mu?):  DM = {dm_mse_dir_hrd['DM_Stat']:.3f} | p = {dm_mse_dir_hrd['P_Value']:.4e} {'[MSE Azaldı]' if dm_mse_dir_hrd['DM_Stat'] > 0 else '[MSE Bozuldu/Arttı]'}")

    # 4. Sınıflandırma: Günlük Wilcoxon Signed-Rank Testi
    wilc_pre_post = daily_classification_wilcoxon_test(y_dir_test, pred_dir_pre, pred_dir_post)
    print(f"\n  4. Sınıflandırma Günlük Doğruluk Karşılaştırması (GÖP Öncesi vs GÖP Sonrası):")
    print(f"     - Günlük Ortalama Doğruluk: GÖP Öncesi = %{wilc_pre_post['Mean_Acc1']*100:.2f} | GÖP Sonrası = %{wilc_pre_post['Mean_Acc2']*100:.2f}")
    print(f"     - Wilcoxon İstatistiği: {wilc_pre_post['Wilcoxon_Stat']:.1f} | p = {wilc_pre_post['P_Value']:.4e} {'[İstatistiksel Olarak Anlamlı p < 0.05]' if wilc_pre_post['Significant_at_05'] else '[Anlamsız]'}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_rigorous_evaluation()
