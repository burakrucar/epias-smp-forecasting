"""
Modelleme Modülü: GÖP Öncesi ve GÖP Sonrası Modeller
- 3 Sınıflı Yön Sınıflandırıcı (LightGBM Multiclass)
- GÖP Öncesi Doğrudan SMF Regresörü
- GÖP Sonrası Yayılım (Spread: SMF - PTF) Regresörü
- GÖP Sonrası Sert Kısıtlı (Hard-Constrained) Model (Aykırı Değer Analizi İçin)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator


class ThreeClassDirectionClassifier(BaseEstimator):
    """
    3 Sınıflı Sistem Yönü Sınıflandırıcısı:
    -1: Enerji Fazlası
     0: Dengede
     1: Enerji Açığı
    """
    def __init__(self, n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=40, class_weight="balanced", random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
        self.label_mapping = {-1: 0, 0: 1, 1: 2}
        self.inv_label_mapping = {0: -1, 1: 0, 2: 1}

    def fit(self, X, y):
        y_mapped = np.array([self.label_mapping[val] for val in y])
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X, y_mapped)
        return self

    def predict(self, X):
        preds_mapped = self.model.predict(X)
        return np.array([self.inv_label_mapping[val] for val in preds_mapped])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class DirectSMFRegressor(BaseEstimator):
    """
    Aşama 1: GÖP Öncesi (PTF Bilinmiyor) Doğrudan SMF Seviye Regresörü.
    """
    def __init__(self, n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class PostDAMSpreadRegressor(BaseEstimator):
    """
    Aşama 2: GÖP Sonrası (PTF Biliniyor) Fiyat Yayılımı (Spread: S = SMF - PTF) Regresörü.
    Fiyat seviyesi kaymasını (enflasyon) elimine eder.
    Tahmin: SMF_pred = PTF + Spread_pred
    """
    def __init__(self, n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=30, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.model = None

    def fit(self, X, y_spread):
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X, y_spread)
        return self

    def predict_smf(self, X, ptf_values):
        pred_spread = self.model.predict(X)
        ptf = np.asarray(ptf_values, dtype=float)
        return ptf + pred_spread, pred_spread


class HardConstrainedHybridModel:
    """
    Aşama 2: Sert Kısıtlı Model (Hard Constraint)
    Modelin tahmin ettiği yön sınıfına göre SMF değerini PTF sınırına zorlar.
    Hata analizi: Yön yanlış tahmin edildiğinde neden RMSE patlar? Bu kurgu tezde incelenir.
    """
    def __init__(self, clf: ThreeClassDirectionClassifier, reg_direct):
        self.clf = clf
        self.reg = reg_direct

    def predict(self, X, ptf_values):
        pred_dir = self.clf.predict(X)
        probs = self.clf.predict_proba(X)
        X_aug = np.hstack([X, probs])
        raw_smf = self.reg.predict(X_aug)

        constrained_smf = raw_smf.copy()
        ptf = np.asarray(ptf_values, dtype=float)

        for i in range(len(constrained_smf)):
            d = pred_dir[i]
            if d == 1:       # Sistem Açığı öngörüsü -> SMF >= PTF
                if constrained_smf[i] < ptf[i]:
                    constrained_smf[i] = ptf[i]
            elif d == -1:    # Sistem Fazlası öngörüsü -> SMF <= PTF
                if constrained_smf[i] > ptf[i]:
                    constrained_smf[i] = ptf[i]
            elif d == 0:     # Denge öngörüsü -> SMF = PTF
                constrained_smf[i] = ptf[i]

        return pred_dir, constrained_smf, raw_smf
