import pandas as pd
import numpy as np
import sys
import os

sys.path.append("src")
from features import build_thesis_feature_matrix
from models import ThreeClassDirectionClassifier, PostDAMSpreadRegressor, DirectSMFRegressor
from reproduce_all_thesis_results import calc_exact_pnl
from significance_tests import block_bootstrap_ci
from sklearn.linear_model import LogisticRegression

print("=========================================================================================")
print("  AUDIT ROUND 8: 7 KRİTİK HATA VE ŞAİBELİ NOKTA DERİN DENETİMİ")
print("=========================================================================================")

# 1. VERİ YÜKLEME VE HAZIRLIK
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

val_df = df_clean[val_mask].copy().reset_index(drop=True)
test_df = df_clean[test_mask].copy().reset_index(drop=True)

ptf_test = test_df["ptf"].values
smf_test = test_df["smf"].values
neg_test = test_df["negativeImbalance"].values
pos_test = test_df["positiveImbalance"].values
y_dir_test = test_df["target_direction"].values
timestamps_test = test_df["timestamp"]
n_test = len(test_df)

# Fit Canonical Models
reg_spread = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_spread.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
_, pred_spread_test = reg_spread.predict_smf(test_df[cols_post], ptf_test)
act_s2 = np.where(pred_spread_test > 0, 1, -1)
pay_s2 = calc_exact_pnl(act_s2, ptf_test, neg_test, pos_test, y_dir_test)

# Stage 1 Direction & Calibration
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

ev_l_cal = proba_cal[:, 2] * c_def - (1.0 - proba_cal[:, 2]) * c_sur
ev_s_cal = proba_cal[:, 0] * c_sur - (1.0 - proba_cal[:, 0]) * c_def
act_s1_cal = np.zeros(n_test)
act_s1_cal[(ev_l_cal > 70.0) & (ev_l_cal > ev_s_cal)] = 1
act_s1_cal[(ev_s_cal > 70.0) & (ev_s_cal > ev_l_cal)] = -1
pay_s1_cal = calc_exact_pnl(act_s1_cal, ptf_test, neg_test, pos_test, y_dir_test)

act_maj = np.ones(n_test)
pay_maj = calc_exact_pnl(act_maj, ptf_test, neg_test, pos_test, y_dir_test)

# =======================================================================================
# [1] REJİM 3 SİMETRİK KIRPMA BOOTSTRAP GA DENETİMİ
# =======================================================================================
print("\n--- [1] REJİM 3 SİMETRİK KIRPMA GA HESABI ---")
r3 = timestamps_test >= pd.Timestamp("2026-04-04 00:00:00+03:00")
spread_r3 = smf_test[r3] - ptf_test[r3]
q_low = np.percentile(spread_r3, 1.0)
q_high = np.percentile(spread_r3, 99.0)
trim_mask = (spread_r3 >= q_low) & (spread_r3 <= q_high)

diff_trim_hourly = pay_s1_cal[r3][trim_mask] - pay_maj[r3][trim_mask]
pt_est = np.mean(diff_trim_hourly)

# Check block bootstrap implementations
n_trim = len(diff_trim_hourly)
block_len = 168
n_blocks = n_trim // block_len
rng = np.random.default_rng(42)

boot_means = []
for _ in range(2000):
    start_idx = rng.integers(0, n_trim - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in start_idx])
    boot_means.append(np.mean(diff_trim_hourly[idx]))

boot_means = np.array(boot_means)
print(f"Point estimate: {pt_est:.4f}")
print(f"Mean of bootstrap replicates: {np.mean(boot_means):.4f}")
print(f"Std of bootstrap replicates:  {np.std(boot_means):.4f}")
print(f"Empirical Percentile CI:      [{np.percentile(boot_means, 2.5):.2f}, {np.percentile(boot_means, 97.5):.2f}]")
print(f"Normal-approx CI:             [{pt_est - 1.96*np.std(boot_means):.2f}, {pt_est + 1.96*np.std(boot_means):.2f}]")

# Check block_bootstrap_ci from significance_tests.py
_, func_lb, func_ub = block_bootstrap_ci(diff_trim_hourly, block_len=168, n_boot=2000, seed=42)
print(f"Function block_bootstrap_ci output: [{func_lb:.2f}, {func_ub:.2f}]")

# =======================================================================================
# [2] TABLO 9 (AYLIK TABLO) İLE YILLIK ORTALAMALAR MUTABAKATI
# =======================================================================================
print("\n--- [2] AYLIK TABLO VE YILLIK ORTALAMALAR MUTABAKATI ---")
test_df["pay_maj"] = pay_maj
test_df["pay_s1_cal"] = pay_s1_cal
test_df["pay_s2"] = pay_s2
test_df["month_str"] = timestamps_test.dt.strftime("%Y-%m")

monthly_grp = test_df.groupby("month_str").agg(
    hours=("timestamp", "count"),
    pay_maj=("pay_maj", "mean"),
    pay_s1_cal=("pay_s1_cal", "mean"),
    pay_s2=("pay_s2", "mean"),
    ptf=("ptf", "mean"),
    smf=("smf", "mean")
).reset_index()

monthly_grp["spread"] = monthly_grp["smf"] - monthly_grp["ptf"]

print("Monthly Table:")
print(monthly_grp.to_string(index=False))

# Weighted averages
w_maj = np.sum(monthly_grp["pay_maj"] * monthly_grp["hours"]) / n_test
w_s1 = np.sum(monthly_grp["pay_s1_cal"] * monthly_grp["hours"]) / n_test
w_s2 = np.sum(monthly_grp["pay_s2"] * monthly_grp["hours"]) / n_test

print(f"\nWeighted Monthly Sum: Maj = {w_maj:.4f} | S1 = {w_s1:.4f} | S2 = {w_s2:.4f}")
print(f"Overall Test Means:   Maj = {np.mean(pay_maj):.4f} | S1 = {np.mean(pay_s1_cal):.4f} | S2 = {np.mean(pay_s2):.4f}")
print(f"Difference:           Maj = {abs(w_maj - np.mean(pay_maj)):.2e} | S1 = {abs(w_s1 - np.mean(pay_s1_cal)):.2e} | S2 = {abs(w_s2 - np.mean(pay_s2)):.2e}")

# Simple unweighted average
print(f"Simple Unweighted Monthly Mean: Maj = {monthly_grp['pay_maj'].mean():.2f} | S1 = {monthly_grp['pay_s1_cal'].mean():.2f} | S2 = {monthly_grp['pay_s2'].mean():.2f}")

# =======================================================================================
# [3] 12-36 SAAT TAHMİN UFKUNA GÖRE PERFORMANS (HORIZON DECAY)
# =======================================================================================
print("\n--- [3] 12-36 SAAT TAHMİN UFKUNA GÖRE PERFORMANS KIRILIMI ---")
test_df["target_hour"] = test_df["timestamp"].dt.hour
test_df["horizon_hours"] = 12.5 + test_df["target_hour"]

pred_smf_s2 = ptf_test + pred_spread_test
test_df["err_s2_mae"] = np.abs(pred_smf_s2 - smf_test)
test_df["err_s2_mse"] = (pred_smf_s2 - smf_test)**2

test_df["err_naive_mae"] = np.abs(ptf_test - smf_test)
test_df["err_naive_mse"] = (ptf_test - smf_test)**2

test_df["pred_s1_dir"] = np.argmax(proba_cal, axis=1) # 0: -1, 1: 0, 2: 1
test_df["true_s1_dir"] = np.where(y_dir_test==-1, 0, np.where(y_dir_test==0, 1, 2))
test_df["acc_s1"] = (test_df["pred_s1_dir"] == test_df["true_s1_dir"]).astype(int)

def get_h_bin(h):
    if h < 6: return "1. Çeyrek (12-18 Saat / Gece)"
    elif h < 12: return "2. Çeyrek (19-24 Saat / Sabah)"
    elif h < 18: return "3. Çeyrek (25-30 Saat / Gündüz)"
    else: return "4. Çeyrek (31-36 Saat / Akşam Zirvesi)"

test_df["h_bin"] = test_df["target_hour"].apply(get_h_bin)

h_summary = test_df.groupby("h_bin").agg(
    hours=("timestamp", "count"),
    s1_acc=("acc_s1", "mean"),
    naive_mae=("err_naive_mae", "mean"),
    naive_rmse=("err_naive_mse", lambda x: np.sqrt(np.mean(x))),
    s2_mae=("err_s2_mae", "mean"),
    s2_rmse=("err_s2_mse", lambda x: np.sqrt(np.mean(x))),
    s2_pnl=("pay_s2", "mean"),
    s1_pnl=("pay_s1_cal", "mean"),
    maj_pnl=("pay_maj", "mean")
).reset_index()

print("Horizon Breakdown Summary:")
print(h_summary.to_string(index=False))

# =======================================================================================
# [4] DOĞRUDAN SMF MODELLERİ VE TEST/VAL KÜMESİ SMF=PTF REFERANSLARI
# =======================================================================================
print("\n--- [4] GÖP ÖNCESİ DOĞRUDAN SMF vs GÖP SONRASI YAYILIM vs NAİF SMF=PTF ---")
val_ptf = val_df["ptf"].values
val_smf = val_df["smf"].values

val_naive_mae = np.mean(np.abs(val_smf - val_ptf))
val_naive_rmse = np.sqrt(np.mean((val_smf - val_ptf)**2))

reg_direct = DirectSMFRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_direct.fit(df_clean.loc[tr_mask, cols_pre], df_clean.loc[tr_mask, "target_smf"])
val_pred_direct = reg_direct.predict(val_df[cols_pre])
val_direct_mae = np.mean(np.abs(val_smf - val_pred_direct))
val_direct_rmse = np.sqrt(np.mean((val_smf - val_pred_direct)**2))

reg_sp_val = PostDAMSpreadRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_sp_val.fit(df_clean.loc[tr_mask, cols_post], df_clean.loc[tr_mask, "target_smf"] - df_clean.loc[tr_mask, "ptf"])
_, val_pred_sp = reg_sp_val.predict_smf(val_df[cols_post], val_ptf)
val_sp_mae = np.mean(np.abs(val_smf - val_pred_sp))
val_sp_rmse = np.sqrt(np.mean((val_smf - val_pred_sp)**2))

print(f"VALİDASYON KÜMESİ (N=6552):")
print(f"  Modelsiz SMF = PTF (Yayılım=0): MAE = {val_naive_mae:6.2f} TL | RMSE = {val_naive_rmse:6.2f} TL")
print(f"  GÖP Öncesi Doğrudan SMF Modeli: MAE = {val_direct_mae:6.2f} TL | RMSE = {val_direct_rmse:6.2f} TL")
print(f"  GÖP Sonrası Yayılım Modeli:     MAE = {val_sp_mae:6.2f} TL | RMSE = {val_sp_rmse:6.2f} TL")

test_naive_mae = np.mean(np.abs(smf_test - ptf_test))
test_naive_rmse = np.sqrt(np.mean((smf_test - ptf_test)**2))

reg_direct_test = DirectSMFRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42)
reg_direct_test.fit(df_clean.loc[trval_mask, cols_pre], df_clean.loc[trval_mask, "target_smf"])
test_pred_direct = reg_direct_test.predict(test_df[cols_pre])
test_direct_mae = np.mean(np.abs(smf_test - test_pred_direct))
test_direct_rmse = np.sqrt(np.mean((smf_test - test_pred_direct)**2))

test_sp_mae = np.mean(np.abs(smf_test - pred_smf_s2))
test_sp_rmse = np.sqrt(np.mean((smf_test - pred_smf_s2)**2))

print(f"\nTEST KÜMESİ (N=8760):")
print(f"  Modelsiz SMF = PTF (Yayılım=0): MAE = {test_naive_mae:6.2f} TL | RMSE = {test_naive_rmse:6.2f} TL")
print(f"  GÖP Öncesi Doğrudan SMF Modeli: MAE = {test_direct_mae:6.2f} TL | RMSE = {test_direct_rmse:6.2f} TL")
print(f"  GÖP Sonrası Yayılım Modeli:     MAE = {test_sp_mae:6.2f} TL | RMSE = {test_sp_rmse:6.2f} TL")

# =======================================================================================
# [5] POZİTİF DENGESİZLİK 1.538 SAATLİK KALINTI ANALİZİ (-103 TL SINIRI)
# =======================================================================================
print("\n--- [5] POZİTİF DENGESİZLİK 1.538 SAAT KALINTI İNCELEMESİ ---")
p2_mask = test_df["timestamp"] >= pd.Timestamp("2026-01-01 00:00:00+03:00")
p2_df = test_df[p2_mask].copy()

pos_statutory = np.where(
    p2_df["systemStatus"] == "Enerji Fazlası",
    np.minimum(p2_df["ptf"], p2_df["smf"]) * 0.94,
    np.minimum(p2_df["ptf"], p2_df["smf"]) * 0.97
)

diff_pos = p2_df["positiveImbalance"] - pos_statutory
pos_mismatches = p2_df[np.abs(diff_pos) >= 0.1].copy()
pos_mismatches["residual"] = diff_pos[np.abs(diff_pos) >= 0.1]
pos_mismatches["statutory"] = pos_statutory[np.abs(diff_pos) >= 0.1]

print(f"Total pos mismatches: {len(pos_mismatches)}")
print("Distribution of residuals across all 1538:")
print(pos_mismatches["residual"].describe())

zero_smf = pos_mismatches["smf"] <= 0.01
print(f"\nZero SMF hours count: {zero_smf.sum()}")

non_zero = pos_mismatches[~zero_smf].copy()
print(f"\nNon-zero SMF hours count: {len(non_zero)}")
print("Non-zero residual stats:")
print(non_zero["residual"].describe())

print("\nSample 10 rows of non-zero SMF pos mismatches:")
sample_cols = ["timestamp", "systemStatus", "ptf", "smf", "positiveImbalance", "statutory", "residual"]
print(non_zero[sample_cols].head(10).to_string(index=False))

# Check ratio positiveImbalance / min(ptf, smf)
non_zero["ratio"] = non_zero["positiveImbalance"] / np.minimum(non_zero["ptf"], non_zero["smf"])
print("\nRatio of positiveImbalance / min(ptf, smf) for non-zero hours:")
print(non_zero["ratio"].describe())

print("\nSystem status distribution of non-zero:")
print(non_zero["systemStatus"].value_counts())

# =======================================================================================
# [6] AKTİF SAAT BAŞINA GETİRİ VE YUVARLAMA HASSASİYETİ
# =======================================================================================
print("\n--- [6] AKTİF SAAT BAŞINA GETİRİ VE YUVARLAMA HASSASİYETİ ---")
act_hours = np.sum(act_s1_cal != 0)
total_payoff = np.sum(pay_s1_cal)
pnl_per_act = total_payoff / act_hours
pnl_overall = np.mean(pay_s1_cal)

print(f"Total Hours: {n_test}")
print(f"Active Hours: {act_hours} ({act_hours/n_test*100:.3f}%)")
print(f"Passive Hours: {n_test - act_hours} ({(n_test - act_hours)/n_test*100:.3f}%)")
print(f"Total Net Payoff (TL): {total_payoff:.2f}")
print(f"Overall Mean Payoff (TL/MWh): {pnl_overall:.4f}")
print(f"Exact Payoff per Active Hour (sum / act): {pnl_per_act:.4f}")
print(f"Payoff per Active Hour from rounded (67.33 * 8760 / 4287): {(67.33 * 8760) / 4287:.4f}")

# Long / Short breakdown
long_hours = np.sum(act_s1_cal == 1)
short_hours = np.sum(act_s1_cal == -1)
passive_hours = np.sum(act_s1_cal == 0)
print(f"Test Eylem Kırılımı:")
print(f"  Uzun:  {long_hours} ({long_hours/n_test*100:.2f}%)")
print(f"  Kısa:  {short_hours} ({short_hours/n_test*100:.2f}%)")
print(f"  Pasif: {passive_hours} ({passive_hours/n_test*100:.2f}%)")
print(f"  Toplam: {long_hours/n_test*100 + short_hours/n_test*100 + passive_hours/n_test*100:.2f}%")

# Tablo 10 Canlı Pencere eylem oranları
live_long = 97
live_short = 12
live_passive = 179
live_total = 288
print(f"\nCanlı Pencere Eylemleri (N=288):")
print(f"  Uzun:    {live_long} ({live_long/live_total*100:.2f}%)")
print(f"  Kısa:    {live_short} ({live_short/live_total*100:.2f}%)")
print(f"  Pasif:   {live_passive} ({live_passive/live_total*100:.2f}%)")
print(f"  Toplam:  {live_long/live_total*100 + live_short/live_total*100 + live_passive/live_total*100:.2f}%")
