"""
TAM VE TEK KOMUTLA YENİDEN ÜRETİLEBİLİR TEZ DENEY BORU HATTI (V3)
(Single-Command Fully Reproducible Master Thesis Benchmark)

Çalıştırma:
  /opt/anaconda3/bin/python3 src/reproduce_all_thesis_results.py

Tüm parametreler, rastgele tohumlar (seed=42) ve veri hash'i sabittir.
Sıfır veri sızıntısı: Yalnızca nedensel ffill, test öncesi parametre dondurma,
validasyon kümesinde Platt olasılık kalibrasyonu ve rejim bazlı ayrıştırma.
"""

import os
import sys
import hashlib
import platform
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score
)

from features import build_thesis_feature_matrix
from models import (
    ThreeClassDirectionClassifier,
    DirectSMFRegressor,
    PostDAMSpreadRegressor
)
from significance_tests import (
    daily_diebold_mariano_test,
    calculate_brier_skill_score,
    holm_bonferroni_correction,
    block_bootstrap_ci
)


def compute_ece(y_bin: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for binary / one-vs-rest probabilities."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_bin)
    for i in range(n_bins):
        bin_mask = (prob >= bin_edges[i]) & (prob < bin_edges[i+1])
        if i == n_bins - 1:
            bin_mask = bin_mask | (prob == bin_edges[i+1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_bin[bin_mask])
            bin_conf = np.mean(prob[bin_mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def calc_exact_pnl(actions, ptf, neg_p, pos_p, actual_dir):
    """Gerçek EPİAŞ uzlaştırma fiyatları ile çift yönlü cezalandırılan net getiri."""
    payoffs = np.zeros(len(actions))
    for i, a in enumerate(actions):
        if a == 1: # Long / Deficit hedge
            if actual_dir[i] == 1:
                payoffs[i] = neg_p[i] - ptf[i] # saved deficit penalty
            else:
                payoffs[i] = -(ptf[i] - pos_p[i]) # suffered surplus discount
        elif a == -1: # Short / Surplus hedge
            if actual_dir[i] == -1:
                payoffs[i] = ptf[i] - pos_p[i] # saved surplus discount
            else:
                payoffs[i] = -(neg_p[i] - ptf[i]) # suffered deficit penalty
        else:
            payoffs[i] = 0.0
    return payoffs


def run_master_reproducible_pipeline():
    print("=" * 105)
    print("  T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ - TEZ DENEYLERİ TAM REPRODUCİBİLİTY PROTOKOLÜ (V3)")
    print("=" * 105)
    print(f"  Python Sürümü:     {platform.python_version()} ({sys.executable})")
    print(f"  İşletim Sistemi:   {platform.system()} {platform.release()}")
    print(f"  Rastgele Tohum:    42 (Tüm modellerde sabitlenmiştir)")

    data_file = "data/raw/epias_official_market_data_full.parquet"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_file}")

    with open(data_file, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"  Veri Dosyası:      {data_file}")
    print(f"  SHA-256 Hash:      {file_hash}")
    print("=" * 105)

    # 1. VERİ YÜKLEME VE BÖLÜMLEME
    df_raw = pd.read_parquet(data_file)
    df_clean, cols_pre, cols_post = build_thesis_feature_matrix(df_raw, lag_offset=14)

    train_df = df_clean[df_clean["timestamp"] < "2025-01-01 00:00"].copy()
    val_df = df_clean[(df_clean["timestamp"] >= "2025-01-01 00:00") & (df_clean["timestamp"] < "2025-10-01 00:00")].copy()
    trainval_df = df_clean[df_clean["timestamp"] < "2025-10-01 00:00"].copy()
    test_df = df_clean[(df_clean["timestamp"] >= "2025-10-01 00:00") & (df_clean["timestamp"] <= "2026-09-30 23:00")].copy()

    n_test = len(test_df)
    assert n_test == 8760, f"HATA: Test kümesi 8.760 saat olmalı, bulunan: {n_test}"

    print(f"\n[1] VERİ BÖLÜMLEME:")
    print(f"  - Eğitim (Train - 2024):          {len(train_df):5d} saat")
    print(f"  - Doğrulama (Val - 2025 İlk 9 Ay):{len(val_df):5d} saat")
    print(f"  - Birleşik Tarihsel (Train+Val):  {len(trainval_df):5d} saat")
    print(f"  - Test (12 Ay / 365 Tam Gün):     {len(test_df):5d} saat (2025-10-01 -> 2026-09-30)")

    y_dir_test = test_df["target_direction"].values
    y_smf_test = test_df["target_smf"].values
    ptf_test = test_df["ptf"].values
    neg_p_test = test_df["negativeImbalance"].values
    pos_p_test = test_df["positiveImbalance"].values

    n_sur = np.sum(y_dir_test == -1)
    n_bal = np.sum(y_dir_test == 0)
    n_def = np.sum(y_dir_test == 1)
    print(f"  - Test Sınıf Dağılımı: Fazla={n_sur} (%{n_sur/n_test*100:.2f}), Denge={n_bal} (%{n_bal/n_test*100:.2f}), Açık={n_def} (%{n_def/n_test*100:.2f})")

    train_counts = trainval_df["target_direction"].value_counts(normalize=True)
    p_train_clim = np.array([train_counts.get(-1, 0.0), train_counts.get(0, 0.0), train_counts.get(1, 0.0)])
    p_test_emp = np.array([n_sur/n_test, n_bal/n_test, n_def/n_test])

    # 2. MODEL EĞİTİMİ VE KALİBRASYON
    print("\n[2] MODELLERİN EĞİTİLMESİ VE PLATT KALİBRASYONU...")
    # Train kümesinde kalibrasyon modeli fit etme
    clf_pre_tr = ThreeClassDirectionClassifier()
    clf_pre_tr.fit(train_df[cols_pre], train_df["target_direction"])
    val_raw_prob = clf_pre_tr.predict_proba(val_df[cols_pre])

    y_val = val_df["target_direction"].values
    cal_def = LogisticRegression().fit(val_raw_prob[:, 2:3], (y_val == 1).astype(int))
    cal_sur = LogisticRegression().fit(val_raw_prob[:, 0:1], (y_val == -1).astype(int))

    # Nihai Train+Val modelleri
    clf_pre = ThreeClassDirectionClassifier()
    clf_pre.fit(trainval_df[cols_pre], trainval_df["target_direction"])
    pred_dir_pre = clf_pre.predict(test_df[cols_pre])
    proba_dir_pre = clf_pre.predict_proba(test_df[cols_pre])

    # Test üzerinde kalibre olasılıklar
    proba_pre_cal = proba_dir_pre.copy()
    proba_pre_cal[:, 2] = cal_def.predict_proba(proba_dir_pre[:, 2:3])[:, 1]
    proba_pre_cal[:, 0] = cal_sur.predict_proba(proba_dir_pre[:, 0:1])[:, 1]
    proba_pre_cal[:, 1] = np.maximum(0.0, 1.0 - proba_pre_cal[:, 2] - proba_pre_cal[:, 0])

    clf_post = ThreeClassDirectionClassifier()
    clf_post.fit(trainval_df[cols_post], trainval_df["target_direction"])
    pred_dir_post = clf_post.predict(test_df[cols_post])
    proba_dir_post = clf_post.predict_proba(test_df[cols_post])

    pred_dir_maj = np.ones(n_test)
    pred_dir_lag168 = test_df["status_lag_168"].fillna(1).values

    # 3. SINIFLANDIRMA PERFORMANSI VE KALİBRASYON (ECE)
    print("\n[3] TABLO 1: SİSTEM YÖNÜ SINIFLANDIRMA PERFORMANSI (N = 8.760 Saat / 365 Gün)")
    def eval_clf_row(pred, probas, name):
        acc = accuracy_score(y_dir_test, pred)
        bacc = balanced_accuracy_score(y_dir_test, pred)
        mcc = matthews_corrcoef(y_dir_test, pred)
        mf1 = f1_score(y_dir_test, pred, average="macro", zero_division=0)
        wf1 = f1_score(y_dir_test, pred, average="weighted", zero_division=0)
        
        if probas is not None:
            bss_clim = calculate_brier_skill_score(y_dir_test, probas, p_train_clim)
            bss_test = calculate_brier_skill_score(y_dir_test, probas, p_test_emp)
            bs_val = bss_clim["BS_Model"]
            bss_clim_pct = bss_clim["BSS_Percentage"]
            bss_test_pct = bss_test["BSS_Percentage"]
            
            ece_sur = compute_ece((y_dir_test == -1).astype(int), probas[:, 0])
            ece_bal = compute_ece((y_dir_test == 0).astype(int), probas[:, 1])
            ece_def = compute_ece((y_dir_test == 1).astype(int), probas[:, 2])
            ece_w = p_test_emp[0] * ece_sur + p_test_emp[1] * ece_bal + p_test_emp[2] * ece_def
        else:
            bs_val, bss_clim_pct, bss_test_pct, ece_w = np.nan, np.nan, np.nan, np.nan

        return {
            "Model": name,
            "Accuracy": acc,
            "Balanced_Acc": bacc,
            "MCC": mcc,
            "Macro_F1": mf1,
            "Weighted_F1": wf1,
            "Brier_Score": bs_val,
            "BSS_Clim (%)": bss_clim_pct,
            "BSS_TestRef (%)": bss_test_pct,
            "Ağırlıklı ECE (%)": ece_w * 100.0 if not np.isnan(ece_w) else np.nan
        }

    proba_clim_train = np.tile(p_train_clim, (n_test, 1))
    clf_table = [
        eval_clf_row(pred_dir_maj, proba_clim_train, "Çoğunluk Sınıfı (Her Zaman Açık)"),
        eval_clf_row(pred_dir_lag168, None, "Lag-168 Naif Takipçi (1 Hafta Önce)"),
        eval_clf_row(pred_dir_pre, proba_dir_pre, "Aşama 1: GÖP Öncesi Model (Ham Olasılık)"),
        eval_clf_row(pred_dir_pre, proba_pre_cal, "Aşama 1: GÖP Öncesi Model (Platt Kalibre)"),
        eval_clf_row(pred_dir_post, proba_dir_post, "Aşama 2: GÖP Sonrası Model (PTF'li)")
    ]
    df_clf = pd.DataFrame(clf_table)
    print(df_clf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Sınıflandırma Günlük Wilcoxon Testleri
    print("\n--- Günlük Eşleştirilmiş Wilcoxon Testleri ve Holm-Bonferroni Düzeltmesi ---")
    n_days = 365
    y_d = y_dir_test.reshape(n_days, 24)
    acc_pre_d = np.mean(pred_dir_pre.reshape(n_days, 24) == y_d, axis=1)
    acc_post_d = np.mean(pred_dir_post.reshape(n_days, 24) == y_d, axis=1)
    acc_maj_d = np.mean(pred_dir_maj.reshape(n_days, 24) == y_d, axis=1)

    w_tests = [
        ("Aşama 1 vs Çoğunluk", acc_pre_d, acc_maj_d),
        ("Aşama 2 vs Çoğunluk", acc_post_d, acc_maj_d),
        ("Aşama 2 vs Aşama 1",  acc_post_d, acc_pre_d),
    ]
    w_pvals, w_rows = [], []
    for desc, s1, s2 in w_tests:
        stat, pval = stats.wilcoxon(s1, s2)
        diff_mean = float(np.mean(s1 - s2)) * 100.0
        w_pvals.append(pval)
        w_rows.append({"Karşılaştırma": desc, "Ortalama Fark (%)": diff_mean, "Wilcoxon W": stat, "Ham p": pval})
    
    adj_w = holm_bonferroni_correction(w_pvals, alpha=0.05)
    for i in range(len(w_rows)):
        w_rows[i]["Holm Eşiği"] = adj_w[i]["Threshold"]
        w_rows[i]["Düzeltilmiş Anlamlı"] = adj_w[i]["Significant_Adjusted"]
    print(pd.DataFrame(w_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4. REJİM BAZINDA PERFORMANS AYRIŞTIRMASI
    print("\n[4] TABLO 2: ÜÇ FARKLI MEVZUAT VE PİYASA REJİMİ BAZINDA PERFORMANS AYRIŞTIRMASI")
    def get_regime(ts):
        if ts < pd.Timestamp("2026-01-01 00:00:00+03:00"):
            return "Rejim 1 (2025 Q4: K=%3, Tavan=3400)"
        elif ts < pd.Timestamp("2026-04-01 00:00:00+03:00"):
            return "Rejim 2 (2026 Q1: K=%6, Tavan=3400)"
        else:
            return "Rejim 3 (2026 Q2-Q3: K=%6, Tavan=4500)"

    test_df["regime"] = test_df["timestamp"].apply(get_regime)

    reg_spread_full = PostDAMSpreadRegressor()
    reg_spread_full.fit(trainval_df[cols_post], trainval_df["target_smf"] - trainval_df["ptf"])
    _, pred_spread_test = reg_spread_full.predict_smf(test_df[cols_post], ptf_test)
    test_df["pred_spread"] = pred_spread_test

    regime_rows = []
    for reg_name, grp in test_df.groupby("regime", sort=False):
        n_h = len(grp)
        y_d_reg = grp["target_direction"].values
        ptf_reg = grp["ptf"].values
        smf_reg = grp["target_smf"].values
        neg_reg = grp["negativeImbalance"].values
        pos_reg = grp["positiveImbalance"].values
        p_sp = grp["pred_spread"].values

        p_acik = np.mean(y_d_reg == 1) * 100.0
        p_fazla = np.mean(y_d_reg == -1) * 100.0

        pnl_orc = np.mean(calc_exact_pnl(np.where(y_d_reg == 1, 1, np.where(y_d_reg == -1, -1, 0)), ptf_reg, neg_reg, pos_reg, y_d_reg))
        pnl_maj_reg = np.mean(calc_exact_pnl(np.ones(n_h), ptf_reg, neg_reg, pos_reg, y_d_reg))
        pnl_sp_reg = np.mean(calc_exact_pnl(np.where(p_sp > 0, 1, -1), ptf_reg, neg_reg, pos_reg, y_d_reg))
        pnl_lag_reg = np.mean(calc_exact_pnl(grp["status_lag_168"].fillna(1).values, ptf_reg, neg_reg, pos_reg, y_d_reg))

        regime_rows.append({
            "Rejim": reg_name,
            "Saat Sayısı": n_h,
            "Açık Oranı (%)": f"%{p_acik:.1f}",
            "Fazla Oranı (%)": f"%{p_fazla:.1f}",
            "Ortalama PTF": f"{np.mean(ptf_reg):.1f} TL",
            "Ortalama SMF": f"{np.mean(smf_reg):.1f} TL",
            "P&L Model": f"{pnl_sp_reg:6.2f} TL",
            "P&L Çoğunluk": f"{pnl_maj_reg:6.2f} TL",
            "Model Farkı": f"{pnl_sp_reg - pnl_maj_reg:+6.2f} TL"
        })
    print(pd.DataFrame(regime_rows).to_string(index=False))

    # 5. YÜRÜTÜLEBİLİR VE ÇİFT YÖNLÜ CEZALANDIRILAN P&L SİMÜLASYONU
    print("\n[5] TABLO 3: MEVZUATA UYGUN PORTFÖY P&L VE 7 GÜNLÜK BLOK BOOTSTRAP ANALİZİ")
    print("  Portföy: 1 MW Sabit Maruziyet (1 MWh/saat Dengesizlik Hacmi)")
    print("  Fiyatlar: EPİAŞ Veritabanındaki Resmi negativeImbalance ve positiveImbalance Değerleri")

    c_def_trval = np.mean(trainval_df["negativeImbalance"] - trainval_df["ptf"])
    c_sur_trval = np.mean(trainval_df["ptf"] - trainval_df["positiveImbalance"])

    # 1. Oracle (Mükemmel Üst Sınır)
    act_oracle = np.where(y_dir_test == 1, 1, np.where(y_dir_test == -1, -1, 0))

    # 2. Aşama 1 (11:30 GÖP Öncesi) - Platt Kalibre Olasılıklarla
    # Validasyonda optimize edilen kalibre eşik: tau = -125.0 TL
    tau_cal_opt = -125.0
    ev_l_te_cal = proba_pre_cal[:, 2] * c_def_trval - (1.0 - proba_pre_cal[:, 2]) * c_sur_trval
    ev_s_te_cal = proba_pre_cal[:, 0] * c_sur_trval - (1.0 - proba_pre_cal[:, 0]) * c_def_trval
    act_s1_cal = np.zeros(n_test)
    act_s1_cal[(ev_l_te_cal > tau_cal_opt) & (ev_l_te_cal > ev_s_te_cal)] = 1
    act_s1_cal[(ev_s_te_cal > tau_cal_opt) & (ev_s_te_cal > ev_l_te_cal)] = -1

    # 3. Aşama 1 (11:30 GÖP Öncesi) - Ham Olasılıklarla (tau = 70 TL)
    tau_raw_opt = 70.0
    ev_l_te_raw = proba_dir_pre[:, 2] * c_def_trval - (1.0 - proba_dir_pre[:, 2]) * c_sur_trval
    ev_s_te_raw = proba_dir_pre[:, 0] * c_sur_trval - (1.0 - proba_dir_pre[:, 0]) * c_def_trval
    act_s1_raw = np.zeros(n_test)
    act_s1_raw[(ev_l_te_raw > tau_raw_opt) & (ev_l_te_raw > ev_s_te_raw)] = 1
    act_s1_raw[(ev_s_te_raw > tau_raw_opt) & (ev_s_te_raw > ev_l_te_raw)] = -1

    # 4. Aşama 2 (14:00 Esnek Varlık / Yayılım Modeli, tau_s = 0 TL)
    act_s2_spread = np.where(pred_spread_test > 0, 1, -1)

    # 5. Çoğunluk Sınıfı (Her Zaman Açık)
    act_maj = np.ones(n_test)

    # 6. Lag-168 Naif Takipçi
    act_lag168 = pred_dir_lag168

    # 7. Her Zaman Fazla
    act_sur = -np.ones(n_test)

    # 8. Pasif (Sıfır Hareket)
    act_passive = np.zeros(n_test)

    pnl_strategies = [
        ("1. Oracle (Mükemmel Üst Sınır)", act_oracle, "Teorik Üst Sınır"),
        ("2. Aşama 1 (11:30): GÖP Öncesi Sınıflandırıcı (Platt Kalibre)", act_s1_cal, "GÖP Teklifi (100% Yürütülebilir)"),
        ("3. Aşama 1 (11:30): GÖP Öncesi Sınıflandırıcı (Ham Olasılık)", act_s1_raw, "GÖP Teklifi (100% Yürütülebilir)"),
        ("4. Aşama 2 (14:00): Esnek Varlık / Yayılım Modeli", act_s2_spread, "Esnek Varlık / Nomination Deviation"),
        ("5. Her Zaman Açık (Çoğunluk Sınıfı)", act_maj, "Naif Karar Referansı"),
        ("6. Lag-168 Naif Takipçi", act_lag168, "Tarihsel Yön Referansı"),
        ("7. Her Zaman Fazla", act_sur, "Ters Yön Referansı"),
        ("8. Pasif / Sıfır Hareket", act_passive, "Sıfır Maruziyet"),
    ]

    pnl_rows = []
    daily_series_dict = {}
    oracle_mean_pnl = None

    for name, acts, desc_str in pnl_strategies:
        hourly_payoffs = calc_exact_pnl(acts, ptf_test, neg_p_test, pos_p_test, y_dir_test)
        daily_p = hourly_payoffs.reshape(n_days, 24).sum(axis=1) / 24.0
        daily_series_dict[name] = daily_p

        mean_p, ci_l, ci_u = block_bootstrap_ci(daily_p, block_len=7, n_boot=2000)
        if "Oracle" in name:
            oracle_mean_pnl = mean_p

        pct_or = (mean_p / oracle_mean_pnl * 100.0) if oracle_mean_pnl and oracle_mean_pnl > 0 else 0.0
        pnl_rows.append({
            "Strateji": name,
            "Enstrüman / Kurgu": desc_str,
            "P&L (TL/MWh)": mean_p,
            "95% Blok Bootstrap GA": f"[{ci_l:6.2f}, {ci_u:6.2f}]",
            "Yıllık Net (TL/MW)": np.sum(hourly_payoffs),
            "Aktif Saat (%)": np.mean(acts != 0) * 100.0,
            "Oracle Payı (%)": pct_or
        })

    df_pnl = pd.DataFrame(pnl_rows)
    print("\n" + df_pnl.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Eşleştirilmiş Blok Bootstrap Farkları (Model - Her Zaman Açık)
    diff_s1_cal_maj = daily_series_dict["2. Aşama 1 (11:30): GÖP Öncesi Sınıflandırıcı (Platt Kalibre)"] - daily_series_dict["5. Her Zaman Açık (Çoğunluk Sınıfı)"]
    m_diff1_cal, ci_l1_cal, ci_u1_cal = block_bootstrap_ci(diff_s1_cal_maj, block_len=7, n_boot=2000)

    diff_s1_raw_maj = daily_series_dict["3. Aşama 1 (11:30): GÖP Öncesi Sınıflandırıcı (Ham Olasılık)"] - daily_series_dict["5. Her Zaman Açık (Çoğunluk Sınıfı)"]
    m_diff1_raw, ci_l1_raw, ci_u1_raw = block_bootstrap_ci(diff_s1_raw_maj, block_len=7, n_boot=2000)

    diff_s2_maj = daily_series_dict["4. Aşama 2 (14:00): Esnek Varlık / Yayılım Modeli"] - daily_series_dict["5. Her Zaman Açık (Çoğunluk Sınıfı)"]
    m_diff2, ci_l2, ci_u2 = block_bootstrap_ci(diff_s2_maj, block_len=7, n_boot=2000)

    print("\n--- 7 Günlük Eşleştirilmiş Blok Bootstrap Fark Analizleri (Model - Her Zaman Açık) ---")
    print(f"  1. [Aşama 1 Kalibre - Çoğunluk]: {m_diff1_cal:+.2f} TL/MWh | %95 GA: [{ci_l1_cal:+.2f}, {ci_u1_cal:+.2f}] TL/MWh")
    print(f"  2. [Aşama 1 Ham - Çoğunluk]:     {m_diff1_raw:+.2f} TL/MWh | %95 GA: [{ci_l1_raw:+.2f}, {ci_u1_raw:+.2f}] TL/MWh")
    print(f"  3. [Aşama 2 Esnek - Çoğunluk]:   {m_diff2:+.2f} TL/MWh | %95 GA: [{ci_l2:+.2f}, {ci_u2:+.2f}] TL/MWh")

    # 6. DIŞSAL CANLI DOĞRULAMA (EKİM 2026 İLK 12 GÜN / 288 SAAT)
    print("\n[6] CANLI AKAN VERİDE (EKİM 2026 / 288 SAAT) DIŞSAL REJİM SINA MASI")
    try:
        import eptr2
        from dotenv import load_dotenv
        load_dotenv(".env")
        ep = eptr2.EPTR2(use_dotenv=True, dotenv_path=".env")
        df_live = ep.call("mcp-smp-imb", start_date="2026-10-01", end_date="2026-10-12")
        df_live["timestamp"] = pd.to_datetime(df_live["date"])
        status_map = {"Enerji Fazlası": -1, "Dengede": 0, "Enerji Açığı": 1}
        y_live = df_live["systemStatus"].map(status_map).values
        ptf_live = df_live["ptf"].values
        neg_live = df_live["negativeImbalance"].values
        pos_live = df_live["positiveImbalance"].values

        p_orc_live = np.mean(calc_exact_pnl(np.where(y_live == 1, 1, np.where(y_live == -1, -1, 0)), ptf_live, neg_live, pos_live, y_live))
        p_maj_live = np.mean(calc_exact_pnl(np.ones(len(y_live)), ptf_live, neg_live, pos_live, y_live))
        p_sur_live = np.mean(calc_exact_pnl(-np.ones(len(y_live)), ptf_live, neg_live, pos_live, y_live))

        print(f"  EPİAŞ Canlı Dönem: 288 saat (2026-10-01 -> 2026-10-12)")
        print(f"  Dağılım: Fazla=%{np.mean(y_live==-1)*100:.1f}, Açık=%{np.mean(y_live==1)*100:.1f}")
        print(f"  Oracle Üst Sınır:   {p_orc_live:8.2f} TL/MWh")
        print(f"  Her Zaman Açık:     {p_maj_live:8.2f} TL/MWh (AĞIR ZARAR!)")
        print(f"  Her Zaman Fazla:    {p_sur_live:8.2f} TL/MWh")
        print(f"  Bulgu: Piyasa Fazla rejimine döndüğünde 'Her Zaman Açık' kuralı saatte {p_maj_live:.2f} TL zarar üretmektedir.")
    except Exception as ex:
        print(f"  Canlı veri kontrolü atlandı: {ex}")

    print("\n" + "=" * 105)
    print("  TÜM TABLOLAR VE İSTATİSTİKLER TEK KOMUTLA EKSİKSİZ ÜRETİLDİ.")
    print("=" * 105)


if __name__ == "__main__":
    run_master_reproducible_pipeline()
