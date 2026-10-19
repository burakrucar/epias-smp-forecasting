import pandas as pd
import numpy as np
import sys
import os

sys.path.append("src")
from features import build_thesis_feature_matrix
from models import ThreeClassDirectionClassifier, PostDAMSpreadRegressor
from reproduce_all_thesis_results import calc_exact_pnl
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

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

# 1. Stage 2 L2 Canonical
reg_l2 = PostDAMSpreadRegressor()
reg_l2.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
_, pred_spread_l2 = reg_l2.predict_smf(test_df[cols_post], ptf)
act_s2_l2 = np.where(pred_spread_l2 > 0, 1, -1)
pay_s2_l2 = calc_exact_pnl(act_s2_l2, ptf, neg, pos, y_dir)

# 2. Stage 2 L1 Canonical (n=250, depth=5, lr=0.03, min_child=30)
reg_l1 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, verbose=-1, random_state=42)
reg_l1.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_smf"] - df_clean.loc[trval_mask, "ptf"])
pred_l1 = reg_l1.predict(test_df[cols_post])
act_s2_l1 = np.where(pred_l1 > 0, 1, -1)
pay_s2_l1 = calc_exact_pnl(act_s2_l1, ptf, neg, pos, y_dir)

# 3. Stage 1 Calibre
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

# 4. Always Deficit
act_maj = np.ones(n_test)
pay_maj = calc_exact_pnl(act_maj, ptf, neg, pos, y_dir)

# 5. Stage 2 Classifier Calibrated EV (tau=70) as in Tablo 4 (-20.81 TL)
clf_post_tr = ThreeClassDirectionClassifier(n_estimators=100, learning_rate=0.05, max_depth=None, min_child_samples=20, class_weight=None)
clf_post_tr.fit(df_clean.loc[tr_mask, cols_post], df_clean.loc[tr_mask, "target_direction"])
val_post_raw = clf_post_tr.predict_proba(df_clean.loc[val_mask, cols_post])
cal_post_def = LogisticRegression().fit(val_post_raw[:, 2:3], (y_val == 1).astype(int))
cal_post_sur = LogisticRegression().fit(val_post_raw[:, 0:1], (y_val == -1).astype(int))

clf_post_full = ThreeClassDirectionClassifier(n_estimators=100, learning_rate=0.05, max_depth=None, min_child_samples=20, class_weight=None)
clf_post_full.fit(df_clean.loc[trval_mask, cols_post], df_clean.loc[trval_mask, "target_direction"])
proba_post_raw = clf_post_full.predict_proba(test_df[cols_post])

proba_post_cal = proba_post_raw.copy()
proba_post_cal[:, 2] = cal_post_def.predict_proba(proba_post_raw[:, 2:3])[:, 1]
proba_post_cal[:, 0] = cal_post_sur.predict_proba(proba_post_raw[:, 0:1])[:, 1]
proba_post_cal[:, 1] = np.maximum(0.0, 1.0 - proba_post_cal[:, 2] - proba_post_cal[:, 0])

ev_post_l = proba_post_cal[:, 2] * c_def - (1.0 - proba_post_cal[:, 2]) * c_sur
ev_post_s = proba_post_cal[:, 0] * c_sur - (1.0 - proba_post_cal[:, 0]) * c_def

act_s2_clf_ev70 = np.zeros(n_test)
act_s2_clf_ev70[(ev_post_l > 70.0) & (ev_post_l > ev_post_s)] = 1
act_s2_clf_ev70[(ev_post_s > 70.0) & (ev_post_s > ev_post_l)] = -1
pay_s2_clf_ev70 = calc_exact_pnl(act_s2_clf_ev70, ptf, neg, pos, y_dir)

# 6. Stage 2 Classifier Argmax (-33.18 TL)
act_s2_clf_argmax = clf_post_full.predict(test_df[cols_post])
pay_s2_clf_argmax = calc_exact_pnl(act_s2_clf_argmax, ptf, neg, pos, y_dir)

# Categorization & Priority:
# 1. Top 1% abs spread (|SMF - PTF| >= 3101.66 TL)
# 2. Ceiling hours (not in 1)
# 3. Anomaly hours (-106/-103, not in 1 or 2)
# 4. PTF == SMF hours (not in 1, 2, or 3)
# 5. Rest normal
spread = smf - ptf
abs_spread = np.abs(spread)
q99 = np.percentile(abs_spread, 99)
mask_top1 = abs_spread >= q99

mask_ceiling = (
    ((ts < "2026-01-01") & ((smf >= 3000) | (ptf >= 3000))) |
    ((ts >= "2026-01-01") & (ts < "2026-04-04") & ((smf >= 3400) | (ptf >= 3400))) |
    ((ts >= "2026-04-04") & ((smf >= 4500) | (ptf >= 4500)))
)
mask_anom = (pos == -106) | (neg == -103)
mask_eq = (smf == ptf)

cat = np.full(n_test, "5. Geri Kalan Normal Saatler", dtype=object)
cat[mask_eq] = "4. PTF = SMF Saatleri"
cat[mask_anom] = "3. -106 / -103 Anomali Saatleri"
cat[mask_ceiling] = "2. Tavan Fiyat Saatleri"
cat[mask_top1] = "1. Üst %1 Ekstrem Yayılım"

print("========================================================================================================================")
print("KANONİK TABLO 4B: P&L SAAT TÜRLERİ ATIF VE NET KATKI TABLOSU (TABLO 4 İLE TAM MUTABIK)")
print("========================================================================================================================")
print(f"{'Kategori':32s} | {'Saat':5s} | {'Pay %':5s} | {'Açık':8s} | {'Aşama 1':8s} | {'Aşama 2 L2':10s} | {'Aşama 2 L1':10s} | {'Aşama 2 Clf':11s}")
print("-" * 120)

for c in sorted(np.unique(cat)):
    m = (cat == c)
    cnt = m.sum()
    pct = cnt / n_test * 100
    p_maj = np.mean(pay_maj[m])
    p_s1 = np.mean(pay_s1_cal[m])
    p_l2 = np.mean(pay_s2_l2[m])
    p_l1 = np.mean(pay_s2_l1[m])
    p_clf = np.mean(pay_s2_clf_ev70[m])
    print(f"{c:32s} | {cnt:5d} | %{pct:4.1f} | {p_maj:8.2f} | {p_s1:8.2f} | {p_l2:10.2f} | {p_l1:10.2f} | {p_clf:11.2f}")

print("-" * 120)
print(f"{'GENEL YILLIK ORTALAMA / TOPLAM':32s} | {n_test:5d} | %100.0 | {np.mean(pay_maj):8.2f} | {np.mean(pay_s1_cal):8.2f} | {np.mean(pay_s2_l2):10.2f} | {np.mean(pay_s2_l1):10.2f} | {np.mean(pay_s2_clf_ev70):11.2f}")
print("=" * 120)

print("\nNET KATKI DÖKÜMÜ (TL/MWh cinsinden pay):")
print(f"{'Kategori':32s} | {'Saat':5s} | {'Açık':8s} | {'Aşama 1':8s} | {'Aşama 2 L2':10s} | {'Aşama 2 L1':10s} | {'Aşama 2 Clf':11s}")
print("-" * 120)
for c in sorted(np.unique(cat)):
    m = (cat == c)
    cnt = m.sum()
    c_maj = np.sum(pay_maj[m]) / n_test
    c_s1 = np.sum(pay_s1_cal[m]) / n_test
    c_l2 = np.sum(pay_s2_l2[m]) / n_test
    c_l1 = np.sum(pay_s2_l1[m]) / n_test
    c_clf = np.sum(pay_s2_clf_ev70[m]) / n_test
    print(f"{c:32s} | {cnt:5d} | {c_maj:8.2f} | {c_s1:8.2f} | {c_l2:10.2f} | {c_l1:10.2f} | {c_clf:11.2f}")
print("-" * 120)
print(f"{'TOPLAM P&L KATKISI':32s} | {n_test:5d} | {np.mean(pay_maj):8.2f} | {np.mean(pay_s1_cal):8.2f} | {np.mean(pay_s2_l2):10.2f} | {np.mean(pay_s2_l1):10.2f} | {np.mean(pay_s2_clf_ev70):11.2f}")
print("=" * 120)
