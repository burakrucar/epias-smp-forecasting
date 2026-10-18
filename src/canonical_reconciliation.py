import pandas as pd
import numpy as np
import sys
import os

sys.path.append("src")
from features import build_thesis_feature_matrix
from models import ThreeClassDirectionClassifier, PostDAMSpreadRegressor
from reproduce_all_thesis_results import calc_exact_pnl
from significance_tests import block_bootstrap_ci
from sklearn.linear_model import LogisticRegression

print("=========================================================================================")
print("  KANONİK MUTABAKAT VE TÜM EKONOMETRİK METRİKLER PROTOKOLÜ (V5)")
print("=========================================================================================")

df_raw = pd.read_parquet("data/raw/epias_official_market_data_full.parquet")
df_f = df_raw.copy()
df_f["load_forecast_mw"] = df_f["load_forecast_mw"].ffill()
df_f["wind_forecast_mw"] = df_f["wind_forecast_mw"].ffill()
df_clean, cols_pre, cols_post = build_thesis_feature_matrix(df_f, lag_offset=14)

test_start = pd.Timestamp("2025-10-01 00:00:00+03:00")
test_end = pd.Timestamp("2026-09-30 23:00:00+03:00")
val_start = pd.Timestamp("2025-01-01 00:00:00+03:00")

tr_mask = df_clean["timestamp"] < val_start
val_mask = (df_clean["timestamp"] >= val_start) & (df_clean["timestamp"] < test_start)
trval_mask = df_clean["timestamp"] < test_start
test_mask = (df_clean["timestamp"] >= test_start) & (df_clean["timestamp"] <= test_end)

test_df = df_clean[test_mask].copy().reset_index(drop=True)
ptf_test = test_df["ptf"].values
smf_test = test_df["smf"].values
neg_test = test_df["negativeImbalance"].values
pos_test = test_df["positiveImbalance"].values
y_dir_test = test_df["target_direction"].values
timestamps_test = test_df["timestamp"]
n_test = len(test_df)

# Kesin Rejim Bölünmesi:
# Rejim 1: 2025-10-01 -> 2025-12-31 (2208 saat)
# Rejim 2: 2026-01-01 -> 2026-04-03 (2232 saat) - 14030 Kararı, Tavan 3400
# Rejim 3: 2026-04-04 -> 2026-09-30 (4320 saat) - 14030 Kararı, Tavan 4500
r1 = timestamps_test < pd.Timestamp("2026-01-01 00:00:00+03:00")
r2 = (timestamps_test >= pd.Timestamp("2026-01-01 00:00:00+03:00")) & (timestamps_test < pd.Timestamp("2026-04-04 00:00:00+03:00"))
r3 = timestamps_test >= pd.Timestamp("2026-04-04 00:00:00+03:00")
n1, n2, n3 = r1.sum(), r2.sum(), r3.sum()

# 1. Oracle
act_orc = np.where(y_dir_test == 1, 1, np.where(y_dir_test == -1, -1, 0))
pay_orc = calc_exact_pnl(act_orc, ptf_test, neg_test, pos_test, y_dir_test)

# 2. Her Zaman Açık
act_maj = np.ones(n_test)
pay_maj = calc_exact_pnl(act_maj, ptf_test, neg_test, pos_test, y_dir_test)

# 3. Her Zaman Fazla
act_sur = -np.ones(n_test)
pay_sur = calc_exact_pnl(act_sur, ptf_test, neg_test, pos_test, y_dir_test)

# 4. Lag-168
act_lag = test_df["status_lag_168"].fillna(1).values
pay_lag = calc_exact_pnl(act_lag, ptf_test, neg_test, pos_test, y_dir_test)

# 5. Aşama 2: GÖP Sonrası Yayılım Regresörü (PostDAMSpreadRegressor)
reg_spread = PostDAMSpreadRegressor()
reg_spread.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
_, pred_spread_test = reg_spread.predict_smf(test_df[cols_post], ptf_test)
act_s2 = np.where(pred_spread_test > 0, 1, -1)
pay_s2 = calc_exact_pnl(act_s2, ptf_test, neg_test, pos_test, y_dir_test)

# 6. Aşama 1: GÖP Öncesi Sınıflandırıcı (ThreeClassDirectionClassifier)
clf_tr = ThreeClassDirectionClassifier()
clf_tr.fit(df_clean.loc[tr_mask, cols_pre], df_clean.loc[tr_mask, "target_direction"])
val_raw_prob = clf_tr.predict_proba(df_clean.loc[val_mask, cols_pre])
y_val = df_clean.loc[val_mask, "target_direction"].values

cal_def = LogisticRegression().fit(val_raw_prob[:, 2:3], (y_val == 1).astype(int))
cal_sur = LogisticRegression().fit(val_raw_prob[:, 0:1], (y_val == -1).astype(int))

clf_pre = ThreeClassDirectionClassifier()
clf_pre.fit(df_clean.loc[trval_mask, cols_pre], df_clean.loc[trval_mask, "target_direction"])
proba_dir_pre = clf_pre.predict_proba(test_df[cols_pre])

proba_cal = proba_dir_pre.copy()
proba_cal[:, 2] = cal_def.predict_proba(proba_dir_pre[:, 2:3])[:, 1]
proba_cal[:, 0] = cal_sur.predict_proba(proba_dir_pre[:, 0:1])[:, 1]
proba_cal[:, 1] = np.maximum(0.0, 1.0 - proba_cal[:, 2] - proba_cal[:, 0])

c_def = np.mean(df_clean.loc[trval_mask, "negativeImbalance"] - df_clean.loc[trval_mask, "ptf"])
c_sur = np.mean(df_clean.loc[trval_mask, "ptf"] - df_clean.loc[trval_mask, "positiveImbalance"])

# Aşama 1 Ham (tau = 70.0 TL)
ev_l_raw = proba_dir_pre[:, 2] * c_def - (1.0 - proba_dir_pre[:, 2]) * c_sur
ev_s_raw = proba_dir_pre[:, 0] * c_sur - (1.0 - proba_dir_pre[:, 0]) * c_def
act_s1_raw = np.zeros(n_test)
act_s1_raw[(ev_l_raw > 70.0) & (ev_l_raw > ev_s_raw)] = 1
act_s1_raw[(ev_s_raw > 70.0) & (ev_s_raw > ev_l_raw)] = -1
pay_s1_raw = calc_exact_pnl(act_s1_raw, ptf_test, neg_test, pos_test, y_dir_test)

# Aşama 1 Kalibre (tau = 70.0 TL - Sabit Karşılaştırılabilir Eşik)
ev_l_cal = proba_cal[:, 2] * c_def - (1.0 - proba_cal[:, 2]) * c_sur
ev_s_cal = proba_cal[:, 0] * c_sur - (1.0 - proba_cal[:, 0]) * c_def
act_s1_cal = np.zeros(n_test)
act_s1_cal[(ev_l_cal > 70.0) & (ev_l_cal > ev_s_cal)] = 1
act_s1_cal[(ev_s_cal > 70.0) & (ev_s_cal > ev_l_cal)] = -1
pay_s1_cal = calc_exact_pnl(act_s1_cal, ptf_test, neg_test, pos_test, y_dir_test)

print("\n--- MUTABAKAT TABLOSU: GENEL P&L VE REJİM AĞIRLIKLI TOPLAMI ---")
print(f"{'Strateji':35s} | {'Genel PnL':10s} | {'Rejim 1':10s} | {'Rejim 2':10s} | {'Rejim 3':10s} | {'Agirlikli Top':14s} | {'Fark':8s}")
print("-" * 115)

strategies = [
    ("Oracle (Mükemmel Üst Sınır)", pay_orc),
    ("Her Zaman Açık (Çoğunluk)", pay_maj),
    ("Her Zaman Fazla (Ters Kural)", pay_sur),
    ("Lag-168 Naif Takipçi", pay_lag),
    ("Aşama 1 (Ham, tau=70)", pay_s1_raw),
    ("Aşama 1 (Platt Kalibre, tau=70)", pay_s1_cal),
    ("Aşama 2 (GÖP Sonrası Yayılım)", pay_s2),
]

for name, p_vec in strategies:
    gen_m = np.mean(p_vec)
    r1_m = np.mean(p_vec[r1])
    r2_m = np.mean(p_vec[r2])
    r3_m = np.mean(p_vec[r3])
    w_sum = (r1_m * n1 + r2_m * n2 + r3_m * n3) / n_test
    diff = abs(gen_m - w_sum)
    print(f"{name:35s} | {gen_m:10.4f} | {r1_m:10.4f} | {r2_m:10.4f} | {r3_m:10.4f} | {w_sum:14.4f} | {diff:8.2e}")

print("\n--- GÜVEN ARALIKLARI VE FARK ANALİZLERİ (7 GÜNLÜK BLOK BOOTSTRAP) ---")
for name, p_vec in strategies:
    _, lb, ub = block_bootstrap_ci(p_vec, block_len=168, n_boot=2000, seed=42)
    print(f"{name:35s}: {np.mean(p_vec):6.2f} TL/MWh [%95 GA: {lb:6.2f}, {ub:6.2f}]")

# Eşleştirilmiş Farklar
print("\n--- Eşleştirilmiş Farklar (Strateji - Her Zaman Açık) ---")
for name, p_vec in [
    ("Aşama 1 Ham - Açık", pay_s1_raw - pay_maj),
    ("Aşama 1 Kalibre - Açık", pay_s1_cal - pay_maj),
    ("Aşama 2 Yayılım - Açık", pay_s2 - pay_maj),
]:
    _, lb, ub = block_bootstrap_ci(p_vec, block_len=168, n_boot=2000, seed=42)
    print(f"{name:35s}: {np.mean(p_vec):+6.2f} TL/MWh [%95 GA: {lb:6.2f}, {ub:6.2f}]")

# Rejim bazında farklar ve CIs
print("\n--- Rejim Bazında Farklar (Model - Her Zaman Açık) ---")
for r_name, r_mask in [("Rejim 1", r1), ("Rejim 2", r2), ("Rejim 3", r3)]:
    d1 = pay_s1_cal[r_mask] - pay_maj[r_mask]
    d2 = pay_s2[r_mask] - pay_maj[r_mask]
    _, d1_l, d1_u = block_bootstrap_ci(d1, block_len=168, n_boot=2000, seed=42)
    _, d2_l, d2_u = block_bootstrap_ci(d2, block_len=168, n_boot=2000, seed=42)
    print(f"{r_name}:")
    print(f"  Aşama 1 Kalibre - Açık: {np.mean(d1):+6.2f} TL/MWh [%95 GA: {d1_l:6.2f}, {d1_u:6.2f}]")
    print(f"  Aşama 2 Yayılım - Açık: {np.mean(d2):+6.2f} TL/MWh [%95 GA: {d2_l:6.2f}, {d2_u:6.2f}]")

# Duyarlılık Analizi: Yapısal Maruziyet Hizalanma Oranı (%50 -> %100)
print("\n--- DUYARLILIK ANALİZİ: YAPISAL MARUZİYET HİZALANMA ORANI ---")
# If portfolio is only aligned with probability alpha:
# Expected Payoff = alpha * Payoff_aligned + (1 - alpha) * Payoff_misaligned
# Where Payoff_misaligned is the reverse action payoff
for alpha in [1.0, 0.90, 0.80, 0.70, 0.60, 0.50]:
    # Under alpha alignment:
    pnl_s2_alpha = alpha * np.mean(pay_s2) + (1.0 - alpha) * (-np.mean(pay_s2))
    pnl_s1_alpha = alpha * np.mean(pay_s1_cal) + (1.0 - alpha) * (-np.mean(pay_s1_cal))
    pnl_maj_alpha = alpha * np.mean(pay_maj) + (1.0 - alpha) * (-np.mean(pay_maj))
    print(f"Hizalanma Oranı %{alpha*100:3.0f}: Aşama 2 = {pnl_s2_alpha:6.2f} TL/MWh | Aşama 1 Kalibre = {pnl_s1_alpha:6.2f} TL/MWh | Her Zaman Açık = {pnl_maj_alpha:6.2f} TL/MWh")

# Simetrik Kırpma Analizi (Alt %1 ve Üst %1)
print("\n--- SİMETRİK KIRPMA ANALİZİ (REJİM 3: ALT %1 VE ÜST %1 EKSTREMLER) ---")
test_df["spread"] = smf_test - ptf_test
df_r3 = test_df[r3].copy()
q_low = df_r3["spread"].quantile(0.01)
q_high = df_r3["spread"].quantile(0.99)
trimmed_mask = (df_r3["spread"] >= q_low) & (df_r3["spread"] <= q_high)

pay_maj_r3 = pay_maj[r3]
pay_s2_r3 = pay_s2[r3]
pay_s1_r3 = pay_s1_cal[r3]

print(f"Rejim 3 Ham Ortalama:        Her Zaman Açık = {np.mean(pay_maj_r3):6.2f} TL | Aşama 2 = {np.mean(pay_s2_r3):6.2f} TL | Aşama 1 = {np.mean(pay_s1_r3):6.2f} TL")
print(f"Rejim 3 Simetrik Kırpılmış:  Her Zaman Açık = {np.mean(pay_maj_r3[trimmed_mask]):6.2f} TL | Aşama 2 = {np.mean(pay_s2_r3[trimmed_mask]):6.2f} TL | Aşama 1 = {np.mean(pay_s1_r3[trimmed_mask]):6.2f} TL")

# Tüm 12 Ayın Aylık P&L Tablosu
print("\n--- TÜM 12 AYIN AYLIK P&L TABLOSU (TL/MWh) ---")
test_df["pay_maj"] = pay_maj
test_df["pay_s1_cal"] = pay_s1_cal
test_df["pay_s2"] = pay_s2
test_df["month"] = timestamps_test.dt.strftime("%Y-%m")
monthly_table = test_df.groupby("month")[["pay_maj", "pay_s1_cal", "pay_s2", "ptf", "smf"]].mean()
print(monthly_table.to_string(float_format=lambda x: f"{x:7.2f}"))
