import pandas as pd
import numpy as np
import sys
import os

sys.path.append("src")
from features import build_thesis_feature_matrix
from models import ThreeClassDirectionClassifier, PostDAMSpreadRegressor
from reproduce_all_thesis_results import calc_exact_pnl
from sklearn.linear_model import LogisticRegression

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
ptf = test_df["ptf"].values
smf = test_df["smf"].values
neg = test_df["negativeImbalance"].values
pos = test_df["positiveImbalance"].values
y_dir = test_df["target_direction"].values
ts = test_df["timestamp"]
n_test = len(test_df)

# Stage 2 L2
reg_spread = PostDAMSpreadRegressor()
reg_spread.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
_, pred_spread = reg_spread.predict_smf(test_df[cols_post], ptf)
act_s2_l2 = np.where(pred_spread > 0, 1, -1)
pay_s2_l2 = calc_exact_pnl(act_s2_l2, ptf, neg, pos, y_dir)

# Stage 1 Calibre
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
pay_s1_cal = calc_exact_pnl(act_s1_cal, ptf, neg, pos, y_dir)

# Always Deficit
act_maj = np.ones(n_test)
pay_maj = calc_exact_pnl(act_maj, ptf, neg, pos, y_dir)

# Stage 2 Classifier (Post-DAM direction classifier)
clf_post = ThreeClassDirectionClassifier()
clf_post.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_direction"])
proba_post = clf_post.predict_proba(test_df[cols_post])

# Argmax action
act_s2_clf_argmax = np.where(np.argmax(proba_post, axis=1) == 2, 1, np.where(np.argmax(proba_post, axis=1) == 0, -1, 0))
pay_s2_clf_argmax = calc_exact_pnl(act_s2_clf_argmax, ptf, neg, pos, y_dir)

# EV with tau=70
ev_l_post = proba_post[:, 2] * c_def - (1.0 - proba_post[:, 2]) * c_sur
ev_s_post = proba_post[:, 0] * c_sur - (1.0 - proba_post[:, 0]) * c_def
act_s2_clf_ev70 = np.zeros(n_test)
act_s2_clf_ev70[(ev_l_post > 70.0) & (ev_l_post > ev_s_post)] = 1
act_s2_clf_ev70[(ev_s_post > 70.0) & (ev_s_post > ev_l_post)] = -1
pay_s2_clf_ev70 = calc_exact_pnl(act_s2_clf_ev70, ptf, neg, pos, y_dir)

# Quantile Regressor L1 (Median)
import lightgbm as lgb
reg_l1 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)
reg_l1.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
pred_spread_l1 = reg_l1.predict(test_df[cols_post])
act_s2_l1 = np.where(pred_spread_l1 > 0, 1, -1)
pay_s2_l1 = calc_exact_pnl(act_s2_l1, ptf, neg, pos, y_dir)

print("="*90)
print("P&L ÖZETİ (TL/MWh):")
print(f"Her Zaman Açık:                     {np.mean(pay_maj):.2f}")
print(f"Aşama 1 (Platt Kalibre, tau=70):    {np.mean(pay_s1_cal):.2f}")
print(f"Aşama 2 (L2 Yayılım Regresyonu):    {np.mean(pay_s2_l2):.2f}")
print(f"Aşama 2 (L1 Medyan Regresyonu):     {np.mean(pay_s2_l1):.2f}")
print(f"Aşama 2 (Sınıflandırıcı Argmax):    {np.mean(pay_s2_clf_argmax):.2f}")
print(f"Aşama 2 (Sınıflandırıcı EV tau=70): {np.mean(pay_s2_clf_ev70):.2f}")
print("="*90)

# Saat Türleri Tanımları:
# 1. Üst %1 Mutlak Yayılım (|SMF - PTF|)
spread = smf - ptf
abs_spread = np.abs(spread)
q99 = np.percentile(abs_spread, 99)
mask_top1 = abs_spread >= q99

# 2. Tavan Saatleri (SMF veya PTF ilgili rejim tavanına ulaşanlar)
mask_ceiling = (
    ((ts < "2026-01-01") & ((smf >= 3000) | (ptf >= 3000))) |
    ((ts >= "2026-01-01") & (ts < "2026-04-04") & ((smf >= 3400) | (ptf >= 3400))) |
    ((ts >= "2026-04-04") & ((smf >= 4500) | (ptf >= 4500)))
)

# 3. İdari Anomali Saatleri (-106 ve -103)
mask_anom = (pos == -106) | (neg == -103)

# 4. PTF == SMF Saatleri
mask_eq = (smf == ptf)

# Mutually Exclusive Partisyon (Hiyerarşik / Ayrık Sınıflandırma):
# Sıra:
# A. Üst %1 Ekstrem Yayılım (|SMF - PTF| >= q99)
# B. Tavan Saatleri (A dışında kalan tavan saatleri)
# C. İdari Anomali Saatleri (-106 / -103, A ve B dışında kalan)
# D. PTF = SMF Saatleri (A, B, C dışında kalan)
# E. Geri Kalan Normal Saatler
cat = np.full(n_test, "5. Geri Kalan Normal Saatler", dtype=object)
cat[mask_eq] = "4. PTF = SMF Saatleri"
cat[mask_anom] = "3. -106 / -103 Anomali Saatleri"
cat[mask_ceiling] = "2. Tavan Fiyat Saatleri"
cat[mask_top1] = "1. Üst %1 Ekstrem Yayılım"

print("\nHİYERARŞİK AYRIK PARTİSYON SAAT SAYILARI:")
for c in sorted(np.unique(cat)):
    m = (cat == c)
    print(f"{c:35s}: {m.sum()} saat (%{m.sum()/n_test*100:.2f})")

print("\n" + "="*120)
print(f"{'Saat Türü':32s} | {'Saat':6s} | {'Pay %':6s} | {'Açık':9s} | {'Aşama 1':9s} | {'Aşama 2 L2':10s} | {'Aşama 2 L1':10s} | {'Aşama 2 Clf':11s}")
print("-"*120)

for c in sorted(np.unique(cat)):
    m = (cat == c)
    cnt = m.sum()
    pct = cnt / n_test * 100
    p_maj_m = np.mean(pay_maj[m])
    p_s1_m = np.mean(pay_s1_cal[m])
    p_s2l2_m = np.mean(pay_s2_l2[m])
    p_s2l1_m = np.mean(pay_s2_l1[m])
    p_s2ev_m = np.mean(pay_s2_clf_ev70[m])
    print(f"{c:32s} | {cnt:6d} | %{pct:4.1f} | {p_maj_m:9.2f} | {p_s1_m:9.2f} | {p_s2l2_m:10.2f} | {p_s2l1_m:10.2f} | {p_s2ev_m:11.2f}")

print("-"*120)
print(f"{'TOPLAM / YILLIK ORTALAMA':32s} | {n_test:6d} | %100.0 | {np.mean(pay_maj):9.2f} | {np.mean(pay_s1_cal):9.2f} | {np.mean(pay_s2_l2):10.2f} | {np.mean(pay_s2_l1):10.2f} | {np.mean(pay_s2_clf_ev70):11.2f}")
print("="*120)

# Katkı (P&L Contribution = Mean * Saat / Toplam Saat)
print("\n" + "="*120)
print("TOPLAM P&L'E KATKI DÖKÜMÜ (TL/MWh cinsinden payı):")
print(f"{'Saat Türü':32s} | {'Saat':6s} | {'Açık':9s} | {'Aşama 1':9s} | {'Aşama 2 L2':10s} | {'Aşama 2 L1':10s} | {'Aşama 2 Clf':11s}")
print("-"*120)

for c in sorted(np.unique(cat)):
    m = (cat == c)
    cnt = m.sum()
    c_maj = np.sum(pay_maj[m]) / n_test
    c_s1 = np.sum(pay_s1_cal[m]) / n_test
    c_s2l2 = np.sum(pay_s2_l2[m]) / n_test
    c_s2l1 = np.sum(pay_s2_l1[m]) / n_test
    c_s2ev = np.sum(pay_s2_clf_ev70[m]) / n_test
    print(f"{c:32s} | {cnt:6d} | {c_maj:9.2f} | {c_s1:9.2f} | {c_s2l2:10.2f} | {c_s2l1:10.2f} | {c_s2ev:11.2f}")

print("-"*120)
print(f"{'GENEL TOPLAM P&L':32s} | {n_test:6d} | {np.mean(pay_maj):9.2f} | {np.mean(pay_s1_cal):9.2f} | {np.mean(pay_s2_l2):10.2f} | {np.mean(pay_s2_l1):10.2f} | {np.mean(pay_s2_clf_ev70):11.2f}")
print("="*120)
