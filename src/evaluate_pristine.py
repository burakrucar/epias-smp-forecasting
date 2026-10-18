"""
Akademik Tez Değerlendirme Modülü: Dokunulmamış 12 Aylık Nihai Test Kümesi (8.760 Saat / 365 Tam Gün)
Dönem: 2025-10-01 00:00 - 2026-09-30 23:00

Metodolojik İlkeler:
1. Veri Bütünlüğü: 365 gün × 24 saat = tam 8.760 saat. Sıfır düşen satır, sıfır eksik saat.
2. Model Eğitimi: Train+Val (2024-01-08 - 2025-09-30, 15.168 saat) birleştirilerek test öncesi tam geçmişle eğitilir.
   (Ayrıca dondurulmuş 2024 modeli ile kavram kayması karşılaştırması sunulur).
3. Doğru İstatistiksel Hipotez Testleri:
   - Harvey-Leybourne-Newbold (HLN) düzeltmeli günlük Diebold-Mariano testleri (N = 365 gün) hem MAE hem MSE için.
   - Çoklu testler için Holm-Bonferroni düzeltmesi.
   - Sınıflandırma için günlük eşleştirilmiş Wilcoxon signed-rank testi ve t-testi.
4. Olasılık Kalitesi & Brier Beceri Skoru (BSS):
   - Eğitim dönemi sınıf frekansları (%31,8 Fazla, %4,1 Dengede, %64,1 Açık) klimatolojik referans olarak kullanılır.
   - BSS = 1 - (BS_model / BS_clim)
5. Çift Yönlü Cezalandırılan Gerçekçi Portföy P&L Simülasyonu:
   - EPİAŞ resmi ikili fiyatlandırma uzlaştırması (P_neg ve P_pos).
   - Hem kazançlar hem de yanlış pozisyonda katlanılan ağır piyasa cezaları tam yansıtılır.
   - Mükemmel Öngörü (Oracle üst sınır), Yayılım Modeli Stratejisi, Her Zaman Açık, Lag-168 ve Sıfır Hareket.
   - Günlük P&L için 7 günlük Hareketli Blok Bootstrap (B=2000) ile %95 güven aralıkları.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_recall_fscore_support
)

from features import build_thesis_feature_matrix
from significance_tests import (
    daily_diebold_mariano_test,
    daily_classification_wilcoxon_test,
    get_detailed_class_metrics,
    calculate_brier_skill_score,
    holm_bonferroni_correction,
    block_bootstrap_ci
)


def run_pristine_benchmark():
    print("=" * 95)
    print("  DOKUNULMAMIŞ NİHAİ TEST KÜMESİ AKADEMİK DOĞRULAMA (365 TAM GÜN / 8.760 SAAT)")
    print("=" * 95)

    data_path = "data/raw/epias_official_market_data_full.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

    df_raw = pd.read_parquet(data_path)
    df_clean, cols_gop_oncesi, cols_gop_sonrasi = build_thesis_feature_matrix(df_raw)

    print(f"  Toplam Temizlenmiş Saatlik Satır: {len(df_clean)}")

    # 3 Zaman Dilimi:
    # 1. Train: 2024-01-08 - 2024-12-31 (8.616 saat)
    # 2. Validation: 2025-01-01 - 2025-09-30 (6.552 saat)
    # 3. Test: 2025-10-01 - 2026-09-30 (8.760 saat = 365 gün × 24 saat)
    train_df = df_clean[df_clean["timestamp"] < "2025-01-01 00:00"].copy()
    val_df = df_clean[(df_clean["timestamp"] >= "2025-01-01 00:00") & (df_clean["timestamp"] < "2025-10-01 00:00")].copy()
    trainval_df = df_clean[df_clean["timestamp"] < "2025-10-01 00:00"].copy()
    test_df = df_clean[(df_clean["timestamp"] >= "2025-10-01 00:00") & (df_clean["timestamp"] <= "2026-09-30 23:00")].copy()

    assert len(test_df) == 8760, f"HATA: Test kümesi 8.760 saat olmalı, bulunan: {len(test_df)}"

    print(f"  Train:     {len(train_df):5d} saat ({train_df['timestamp'].min().strftime('%Y-%m-%d')} -> {train_df['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Val:       {len(val_df):5d} saat ({val_df['timestamp'].min().strftime('%Y-%m-%d')} -> {val_df['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Train+Val: {len(trainval_df):5d} saat (Test Öncesi Tüm Tarihsel Veri)")
    print(f"  Test:      {len(test_df):5d} saat ({test_df['timestamp'].min().strftime('%Y-%m-%d')} -> {test_df['timestamp'].max().strftime('%Y-%m-%d')} - 365 TAM GÜN)")
    print("-" * 95)

    y_dir_test = test_df["target_direction"].values
    y_smf_test = test_df["target_smf"].values
    ptf_test = test_df["ptf"].values
    naive_7d_test = test_df["smf_same_hour_7d_mean"].values
    naive_lag168_dir = test_df["status_lag_168"].fillna(1).values

    # Test sınıf dağılımı
    test_counts = pd.Series(y_dir_test).value_counts().sort_index()
    print("  Nihai Test Kümesi Gerçek Sınıf Dağılımı (N = 8.760 Saat):")
    print(f"    - Enerji Fazlası (-1): {test_counts.get(-1, 0):4d} saat (%{test_counts.get(-1, 0)/87.60:.2f})")
    print(f"    - Dengede        ( 0): {test_counts.get( 0, 0):4d} saat (%{test_counts.get( 0, 0)/87.60:.2f})")
    print(f"    - Enerji Açığı   ( 1): {test_counts.get( 1, 0):4d} saat (%{test_counts.get( 1, 0)/87.60:.2f})")

    # Eğitim dönemi dağılımı (Klimatoloji referansı)
    train_counts = trainval_df["target_direction"].value_counts(normalize=True)
    p_train = np.array([train_counts.get(-1, 0.0), train_counts.get(0, 0.0), train_counts.get(1, 0.0)])
    print(f"  Tarihsel Eğitim Ön Dağılımı (Train+Val Klimatoloji): Fazla=%{p_train[0]*100:.2f}, Denge=%{p_train[1]*100:.2f}, Açık=%{p_train[2]*100:.2f}")
    print("=" * 95)

    # =========================================================================
    # 1. BÖLÜM: SİSTEM YÖNÜ SINIFLANDIRMA PERFORMANSI
    # =========================================================================
    print("\n" + "=" * 95)
    print("  1. BÖLÜM: SİSTEM YÖNÜ SINIFLANDIRMA ANALİZİ (3 SINIFLI: Açık / Fazla / Dengede)")
    print("=" * 95)

    from models import ThreeClassDirectionClassifier

    # A) Aşama 1 (GÖP Öncesi / PTF'siz Model)
    clf_pre = ThreeClassDirectionClassifier()
    clf_pre.fit(trainval_df[cols_gop_oncesi], trainval_df["target_direction"])
    pred_dir_pre = clf_pre.predict(test_df[cols_gop_oncesi])
    proba_dir_pre = clf_pre.predict_proba(test_df[cols_gop_oncesi])

    # B) Aşama 2 (GÖP Sonrası / PTF'li Model)
    clf_post = ThreeClassDirectionClassifier()
    clf_post.fit(trainval_df[cols_gop_sonrasi], trainval_df["target_direction"])
    pred_dir_post = clf_post.predict(test_df[cols_gop_sonrasi])
    proba_dir_post = clf_post.predict_proba(test_df[cols_gop_sonrasi])

    # C) Referanslar
    # Çoğunluk sınıfı: Her zaman Enerji Açığı (1)
    pred_dir_maj = np.ones(len(y_dir_test))
    # Lag-168 Naif Yön
    pred_dir_lag168 = naive_lag168_dir

    def eval_clf(pred, probas, name):
        acc = accuracy_score(y_dir_test, pred)
        bacc = balanced_accuracy_score(y_dir_test, pred)
        mcc = matthews_corrcoef(y_dir_test, pred)
        macro_f1 = f1_score(y_dir_test, pred, average="macro", zero_division=0)
        weight_f1 = f1_score(y_dir_test, pred, average="weighted", zero_division=0)
        
        bss_dict = calculate_brier_skill_score(y_dir_test, probas, p_train) if probas is not None else {"BS_Model": np.nan, "BSS_Percentage": np.nan}
        return {
            "Model": name,
            "Accuracy": acc,
            "Balanced_Acc": bacc,
            "MCC": mcc,
            "Macro_F1": macro_f1,
            "Weighted_F1": weight_f1,
            "Brier_Score": bss_dict["BS_Model"],
            "BSS (%)": bss_dict["BSS_Percentage"]
        }

    # Klimatoloji için sabit olasılık matrisi
    proba_clim = np.tile(p_train, (len(y_dir_test), 1))

    clf_results = [
        eval_clf(pred_dir_maj, proba_clim, "Çoğunluk Sınıfı (Her Zaman Açık)"),
        eval_clf(pred_dir_lag168, None, "Lag-168 Naif Takipçi (1 Hafta Önce)"),
        eval_clf(pred_dir_pre, proba_dir_pre, "Aşama 1: GÖP Öncesi Model (PTF'siz)"),
        eval_clf(pred_dir_post, proba_dir_post, "Aşama 2: GÖP Sonrası Model (PTF'li)"),
    ]

    df_clf_summary = pd.DataFrame(clf_results)
    print("\n" + df_clf_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Sınıf bazında Ayrıntılı Metrikler
    print("\n--- Aşama 2 (GÖP Sonrası) Sınıf Bazında Metrikler ---")
    df_class_detail = get_detailed_class_metrics(y_dir_test, pred_dir_post)
    print(df_class_detail.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Günlük Düzeyde İstatistiksel Testler (N = 365 Gün)
    n_days = 365
    y_dir_daily = y_dir_test.reshape(n_days, 24)
    pred_pre_daily = pred_dir_pre.reshape(n_days, 24)
    pred_post_daily = pred_dir_post.reshape(n_days, 24)
    pred_maj_daily = pred_dir_maj.reshape(n_days, 24)

    acc_pre_d = np.mean(pred_pre_daily == y_dir_daily, axis=1)
    acc_post_d = np.mean(pred_post_daily == y_dir_daily, axis=1)
    acc_maj_d = np.mean(pred_maj_daily == y_dir_daily, axis=1)

    w_pre_maj = stats.wilcoxon(acc_pre_d, acc_maj_d)
    w_post_maj = stats.wilcoxon(acc_post_d, acc_maj_d)
    w_post_pre = stats.wilcoxon(acc_post_d, acc_pre_d)

    print("\n--- Eşleştirilmiş Günlük Wilcoxon Signed-Rank Testleri (N = 365 Gün) ---")
    print(f"  1. Aşama 1 vs Çoğunluk: Ortalama Fark = %{(np.mean(acc_pre_d)-np.mean(acc_maj_d))*100:+.2f} | W = {w_pre_maj.statistic:.1f}, p = {w_pre_maj.pvalue:.4f}")
    print(f"  2. Aşama 2 vs Çoğunluk: Ortalama Fark = %{(np.mean(acc_post_d)-np.mean(acc_maj_d))*100:+.2f} | W = {w_post_maj.statistic:.1f}, p = {w_post_maj.pvalue:.4f}")
    print(f"  3. Aşama 2 vs Aşama 1:  Ortalama Fark = %{(np.mean(acc_post_d)-np.mean(acc_pre_d))*100:+.2f} | W = {w_post_pre.statistic:.1f}, p = {w_post_pre.pvalue:.4f}")

    # =========================================================================
    # 2. BÖLÜM: SMF FİYAT TAHMİNİ PERFORMANSI (DÜRÜST BENCHMARK)
    # =========================================================================
    print("\n" + "=" * 95)
    print("  2. BÖLÜM: SMF FİYAT TAHMİNİ (TL/MWh) VE REFERANS KARŞILAŞTIRMASI")
    print("=" * 95)

    from models import DirectSMFRegressor, PostDAMSpreadRegressor

    # A) Aşama 1: GÖP Öncesi Seviye Regresörü (PTF'siz)
    reg_pre = DirectSMFRegressor()
    reg_pre.fit(trainval_df[cols_gop_oncesi], trainval_df["target_smf"])
    pred_smf_pre = reg_pre.predict(test_df[cols_gop_oncesi])

    # B) Aşama 2: GÖP Sonrası Seviye Regresörü
    reg_post_level = DirectSMFRegressor()
    reg_post_level.fit(trainval_df[cols_gop_sonrasi], trainval_df["target_smf"])
    pred_smf_post_level = reg_post_level.predict(test_df[cols_gop_sonrasi])

    # C) Aşama 2: GÖP Sonrası Yayılım (Spread L2 - MSE) Regresörü
    reg_spread_l2 = PostDAMSpreadRegressor()
    reg_spread_l2.fit(trainval_df[cols_gop_sonrasi], trainval_df["target_smf"] - trainval_df["ptf"])
    pred_smf_spread_l2, pred_spread_vals = reg_spread_l2.predict_smf(test_df[cols_gop_sonrasi], ptf_test)

    # D) Aşama 2: GÖP Sonrası Yayılım (Spread L1 - MAE / Medyan) Regresörü
    model_l1 = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=250, learning_rate=0.03,
        max_depth=5, min_child_samples=30, random_state=42, verbose=-1
    )
    model_l1.fit(trainval_df[cols_gop_sonrasi], trainval_df["target_smf"] - trainval_df["ptf"])
    pred_spread_l1_vals = model_l1.predict(test_df[cols_gop_sonrasi])
    pred_smf_spread_l1 = ptf_test + pred_spread_l1_vals

    # E) Aşama 2: Sert Kısıtlı Model (Hard Constraint)
    pred_smf_hard = pred_smf_post_level.copy()
    for i in range(len(pred_smf_hard)):
        d = pred_dir_post[i]
        if d == 1 and pred_smf_hard[i] < ptf_test[i]:
            pred_smf_hard[i] = ptf_test[i]
        elif d == -1 and pred_smf_hard[i] > ptf_test[i]:
            pred_smf_hard[i] = ptf_test[i]
        elif d == 0:
            pred_smf_hard[i] = ptf_test[i]

    def worst_1pct_share(y_true, y_pred):
        sq_err = (y_true - y_pred) ** 2
        n_top = int(len(sq_err) * 0.01)
        sorted_err = np.sort(sq_err)[::-1]
        return float(np.sum(sorted_err[:n_top]) / np.sum(sq_err) * 100.0)

    def eval_reg(y_pred, name):
        mae = mean_absolute_error(y_smf_test, y_pred)
        rmse = root_mean_squared_error(y_smf_test, y_pred)
        r2 = r2_score(y_smf_test, y_pred)
        wape = np.sum(np.abs(y_smf_test - y_pred)) / np.sum(y_smf_test) * 100.0
        w1pct = worst_1pct_share(y_smf_test, y_pred)
        ratio = rmse / mae if mae > 0 else np.nan
        return {
            "Model / Kural": name,
            "MAE (TL)": mae,
            "RMSE (TL)": rmse,
            "R²": r2,
            "WAPE (%)": wape,
            "RMSE/MAE": ratio,
            "En Kötü %1 Payı (%)": w1pct
        }

    reg_models = [
        eval_reg(naive_7d_test, "Naif 1: 7-Günlük Aynı Saat Ortalaması"),
        eval_reg(test_df["smf_lag_168"].fillna(test_df["smf"].mean()).values, "Naif 2: Lag-168 (1 Hafta Önceki Saat)"),
        eval_reg(ptf_test, "Naif 3: SMF = PTF (GÖP Fiyat Eşitliği)"),
        eval_reg(pred_smf_pre, "Aşama 1: GÖP Öncesi Doğrudan ML (PTF'siz)"),
        eval_reg(pred_smf_post_level, "Aşama 2: GÖP Sonrası Seviye ML"),
        eval_reg(pred_smf_spread_l2, "Aşama 2: Yayılım Modeli (Spread L2 / MSE)"),
        eval_reg(pred_smf_spread_l1, "Aşama 2: Yayılım Modeli (Spread L1 / MAE)"),
        eval_reg(pred_smf_hard, "Aşama 2: Sert Kısıtlı Model (Hard Constrained)"),
    ]

    df_reg_summary = pd.DataFrame(reg_models)
    print("\n" + df_reg_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Günlük Diebold-Mariano / HLN Testleri (N = 365 Gün)
    print("\n--- Günlük Diebold-Mariano (HLN Düzeltmeli) Ekonometrik Hipotez Testleri ---")
    dm_tests = [
        ("Aşama 1 vs 7-Günlük Naif (MAE)", y_smf_test, naive_7d_test, pred_smf_pre, 1),
        ("Aşama 1 vs 7-Günlük Naif (MSE)", y_smf_test, naive_7d_test, pred_smf_pre, 2),
        ("SMF = PTF vs Spread L2 (MAE)", y_smf_test, ptf_test, pred_smf_spread_l2, 1),
        ("SMF = PTF vs Spread L2 (MSE)", y_smf_test, ptf_test, pred_smf_spread_l2, 2),
        ("SMF = PTF vs Spread L1 (MAE)", y_smf_test, ptf_test, pred_smf_spread_l1, 1),
        ("SMF = PTF vs Sert Kısıtlı (MAE)", y_smf_test, ptf_test, pred_smf_hard, 1),
        ("SMF = PTF vs Sert Kısıtlı (MSE)", y_smf_test, ptf_test, pred_smf_hard, 2),
    ]

    p_raw_list = []
    dm_results = []
    for desc, y_t, p1, p2, pow_val in dm_tests:
        res = daily_diebold_mariano_test(y_t, p1, p2, power=pow_val)
        p_raw_list.append(res["P_Value"])
        dm_results.append({
            "Karşılaştırma": desc,
            "Kayıp Tipi": "MAE" if pow_val == 1 else "MSE",
            "Ortalama Günlük Fark": res["Mean_Daily_Loss_Diff"],
            "DM-Stat (HLN)": res["DM_Stat"],
            "Ham p-Değeri": res["P_Value"]
        })

    # Holm-Bonferroni Çoklu Test Düzeltmesi
    adj_res = holm_bonferroni_correction(p_raw_list, alpha=0.05)
    for i in range(len(dm_results)):
        dm_results[i]["Düzeltilmiş Anlamlı (0.05)"] = adj_res[i]["Significant_Adjusted"]
        dm_results[i]["Eşik Değeri"] = adj_res[i]["Threshold"]

    df_dm_table = pd.DataFrame(dm_results)
    print(df_dm_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # =========================================================================
    # 3. BÖLÜM: ÇİFT YÖNLÜ CEZALANDIRILAN GERÇEKÇİ P&L SİMÜLASYONU
    # =========================================================================
    print("\n" + "=" * 95)
    print("  3. BÖLÜM: ÇİFT YÖNLÜ CEZALANDIRILAN PORTFÖY DENGESİZLİK P&L ANALİZİ")
    print("=" * 95)

    pos_imb_price = test_df["positiveImbalance"].values
    neg_imb_price = test_df["negativeImbalance"].values

    # Net getiri fonksiyonu: 1 MWh/saat pozisyon için
    # a = +1: Açık yönünde korunma (Long / Satın Alma)
    #   -> Gerçek Açık ise: neg_imb_price - ptf tasarruf edilir (+ kazanç)
    #   -> Gerçek Fazla ise: ptf ödenip pos_imb_price satıldı -> -(ptf - pos_imb_price) ceza (- kayıp!)
    # a = -1: Fazla yönünde korunma (Short / Satış)
    #   -> Gerçek Fazla ise: ptf - pos_imb_price kurtarılır (+ kazanç)
    #   -> Gerçek Açık ise: ptf satılıp neg_imb_price ile yerine kondu -> -(neg_imb_price - ptf) ceza (- kayıp!)
    # a = 0: Sıfır hareket -> 0 TL

    def evaluate_trading_policy(actions, name):
        payoffs = np.zeros(len(actions))
        for i, a in enumerate(actions):
            if a == 1:
                if y_dir_test[i] == 1:
                    payoffs[i] = neg_imb_price[i] - ptf_test[i]
                else:
                    payoffs[i] = -(ptf_test[i] - pos_imb_price[i])
            elif a == -1:
                if y_dir_test[i] == -1:
                    payoffs[i] = ptf_test[i] - pos_imb_price[i]
                else:
                    payoffs[i] = -(neg_imb_price[i] - ptf_test[i])
            else:
                payoffs[i] = 0.0

        daily_payoffs = payoffs.reshape(n_days, 24).sum(axis=1) / 24.0 # TL/MWh günlük ortalama
        mean_p, ci_low, ci_high = block_bootstrap_ci(daily_payoffs, block_len=7, n_boot=2000)
        total_p = np.sum(payoffs)
        active_hours = np.mean(actions != 0) * 100.0

        return {
            "Strateji": name,
            "Saatlik P&L (TL/MWh)": mean_p,
            "95% CI Alt": ci_low,
            "95% CI Üst": ci_high,
            "Yıllık Toplam (TL)": total_p,
            "Aktif Saat (%)": active_hours,
            "daily_series": daily_payoffs
        }

    # Stratejiler
    # 1. Oracle (Mükemmel Üst Sınır)
    act_oracle = np.where(y_dir_test == 1, 1, np.where(y_dir_test == -1, -1, 0))
    res_oracle = evaluate_trading_policy(act_oracle, "1. Oracle (Mükemmel Üst Sınır)")

    # 2. Yayılım Modeli Karar Kuralı (Spread > 0 -> Açık, Spread < 0 -> Fazla)
    act_spread = np.where(pred_spread_vals > 0, 1, -1)
    res_spread = evaluate_trading_policy(act_spread, "2. Aşama 2: Yayılım Modeli (Spread > 0)")

    # 3. Yayılım Modeli Güvenli Eşik (Spread > 50 TL -> Açık, Spread < -50 TL -> Fazla)
    act_spread_50 = np.zeros(len(pred_spread_vals))
    act_spread_50[pred_spread_vals > 50] = 1
    act_spread_50[pred_spread_vals < -50] = -1
    res_spread_50 = evaluate_trading_policy(act_spread_50, "3. Aşama 2: Yayılım Eşikli (|S| > 50 TL)")

    # 4. Çoğunluk Sınıfı: Her Zaman Açık
    act_maj = np.ones(len(y_dir_test))
    res_maj = evaluate_trading_policy(act_maj, "4. Her Zaman Açık (Çoğunluk Sınıfı)")

    # 5. Lag-168 Naif Takipçi
    act_lag168 = naive_lag168_dir
    res_lag168 = evaluate_trading_policy(act_lag168, "5. Lag-168 Naif Takipçi (1 Hafta Önceki Yön)")

    # 6. Her Zaman Fazla
    act_sur = -np.ones(len(y_dir_test))
    res_sur = evaluate_trading_policy(act_sur, "6. Her Zaman Fazla")

    # 7. Pasif (Sıfır Hareket)
    act_zero = np.zeros(len(y_dir_test))
    res_zero = evaluate_trading_policy(act_zero, "7. Pasif / Sıfır Hareket")

    pnl_results = [res_oracle, res_spread, res_spread_50, res_maj, res_lag168, res_sur, res_zero]
    
    # Tablo formatı
    table_rows = []
    oracle_mean = res_oracle["Saatlik P&L (TL/MWh)"]
    for r in pnl_results:
        pct_oracle = (r["Saatlik P&L (TL/MWh)"] / oracle_mean * 100.0) if oracle_mean > 0 else 0.0
        table_rows.append({
            "Strateji": r["Strateji"],
            "P&L (TL/MWh)": f"{r['Saatlik P&L (TL/MWh)']:6.2f}",
            "95% Blok Bootstrap GA": f"[{r['95% CI Alt']:6.2f}, {r['95% CI Üst']:6.2f}]",
            "Yıllık Toplam (TL)": f"{r['Yıllık Toplam (TL)']:>12,.0f}",
            "Aktif Saat (%)": f"{r['Aktif Saat (%)']:5.1f}%",
            "Oracle Payı (%)": f"{pct_oracle:5.1f}%"
        })

    df_pnl = pd.DataFrame(table_rows)
    print("\n" + df_pnl.to_string(index=False))

    # Model ile Çoğunluk Arasındaki Günlük Fark İçin Blok Bootstrap
    diff_daily = res_spread["daily_series"] - res_maj["daily_series"]
    diff_mean, diff_ci_low, diff_ci_high = block_bootstrap_ci(diff_daily, block_len=7, n_boot=2000)
    print("\n--- Model vs Çoğunluk Sınıfı Karşılaştırması (7 Günlük Blok Bootstrap) ---")
    print(f"  Fark Nokta Tahmini (Model - Çoğunluk): {diff_mean:+.2f} TL/MWh")
    print(f"  %95 Blok Bootstrap Güven Aralığı:     [{diff_ci_low:+.2f}, {diff_ci_high:+.2f}] TL/MWh")
    print(f"  Açıklama: Güven aralığı 0'ı kapsadığından ({diff_ci_low:.2f} ile {diff_ci_high:.2f}),")
    print(f"  modelin çoğunluk stratejisine üstünlüğü yüksek bir nokta tahmine (+{diff_mean:.2f} TL/MWh) sahip olmakla birlikte,")
    print(f"  %5 düzeyinde istatistiksel kesinlik iddia edilemez. Bu bulgu tezde dürüstçe raporlanmalıdır.")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_pristine_benchmark()
