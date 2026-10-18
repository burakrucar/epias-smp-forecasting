# İLERİYE DÖNÜK DOĞRULAMA PROTOKOLÜ (PRE-ANALYSIS PLAN)
## EPİAŞ Elektrik Piyasasında SMF ve Sistem Yönü Tahmin Modellerinin Bağımsız İleriye Dönük (Out-of-Sample) Sınanması

**Tez Danışmanı:** Prof. Ali Dinler  
**Kurum:** T.C. İstanbul Medeniyet Üniversitesi  
**Protokol Tarihi:** 18 Ekim 2026  
**Dondurulmuş Git Commit:** *(İlk commit SHA'sı eklenecektir)*  

---

### 1. Amaç ve Kapsam
Bu protokol, lisansüstü tez çalışması kapsamında geliştirilen ve 15 Ekim 2026 tarihi itibarıyla disk üzerinde dondurulan makine öğrenmesi modellerinin seçim yanlılığından (data snooping / p-hacking) tamamen arındırılmış biçimde test edilmesini garanti altına almak üzere hazırlanmıştır. 

Modeller test kümesi verisi (Ekim 2025 – Eylül 2026) üzerinde geliştirilip ayarlandığı için, gerçek out-of-sample başarı ancak modeller kilitlendikten sonra piyasada fiilen oluşacak verilerle ölçülebilir.

---

### 2. Değerlendirme Penceresi ve Veri Akışı
- **Zaman Aralığı:** 01 Ekim 2026 00:00:00+03:00 – 31 Mart 2027 23:00:00+03:00 (6 Ay / 182 Gün / 4.368 Saat).
- **Veri Kaynağı:** EPİAŞ Şeffaflık Platformu Resmi Web Servisleri (`eptr2` kütüphanesi):
  1. `mcp-smp-imb`: PTF, SMF, Pozitif Dengesizlik Fiyatı, Negatif Dengesizlik Fiyatı, Resmi Sistem Durumu.
  2. `load-plan`: TEİAŞ Gün Öncesi Yük Tahmini (`lep`).
  3. `wind-forecast`: TEİAŞ Gün Öncesi Rüzgar Üretim Tahmini (`forecast`).
- **Veri Bütünlüğü Kuralı:** Saatlik eksik veriler yalnızca geçmişe dönük ileri taşıma (`ffill`) ile doldurulacak, hiçbir aşamada geleceğe dönük bilgi sızıntısına (`lookahead bias`) izin verilmeyecektir.

---

### 3. Dondurulan Modeller ve Dosya Sağlama Toplamları (SHA-256)
Aşağıdaki modeller hiçbir yeniden eğitim, hiperparametre güncellemesi veya kalibrasyon katmanı değişikliğine tabi tutulmaksızın sabit kalacaktır:

1. **Aşama 2 Regresörü:** `models/canonical_stage2_regressor.pkl`  
   - SHA-256: `583a2912c3d0ea1ba58200030208d4aec131b6ab7d94707eec780b772cb24fa5`  
   - Algoritma: LightGBM Regressor ($S = SMF - PTF$), $n=250$, Derinlik=5, Öğrenme Oranı=0,03.
2. **Aşama 1 Sınıflandırıcısı:** `models/canonical_stage1_classifier.pkl`  
   - SHA-256: `d4aac8a6a6f4c2e9f25f786bec6e8fa1a66427585fce8f1331613ac9612429ff`  
   - Algoritma: LightGBM 3-Sınıflı Yön Sınıflandırıcısı, $n=100$, Yaprak=31.
3. **Aşama 1 Platt Kalibratörü:** `models/canonical_stage1_calibrator.pkl`  
   - SHA-256: `5db315973a293f9eb43d4f01b1543167c8f143e41883bd8a9ff732ac68e5106a`  
   - Algoritma: Doğrulama kümesinde uydurulmuş One-vs-Rest Lojistik Regresyon.

---

### 4. Sınanacak Stratejiler Listesi ve Karar Kuralları
1. **Oracle (Kusursuz Bilgi - Teorik Tavan):** Fiili yönün kusursuz bilindiği durum ($D_t=1 \rightarrow +1$, $D_t=-1 \rightarrow -1$, $D_t=0 \rightarrow 0$).
2. **Her Zaman Açık (Çoğunluk Sınıfı Naif Kural):** $a_t = +1$ (Her saat sabit Uzun pozisyon).
3. **Her Zaman Fazla (Ters Kural):** $a_t = -1$ (Her saat sabit Kısa pozisyon).
4. **Lag-168 Naif Takipçi:** $a_t = D_{t-168}$ (Tam bir hafta önceki aynı saatin yönü).
5. **Aşama 1 Kalibre Modeli (GÖP Öncesi):** $\tau = 70.0$ TL sabit eşik kuralı:  
   $EV_{\text{Long}} > 70$ ve $EV_{\text{Long}} > EV_{\text{Short}} \rightarrow a_t = +1$; aksi halde simetrik Kısa veya Pasif.
6. **Aşama 2 Yayılım Modeli (GÖP Sonrası):** $\hat{S}_{L2} > 0 \rightarrow a_t = +1$, aksi halde $a_t = -1$.
7. **Aşama 2 Sınıflandırıcısı (Post-DAM EV):** $EV > 70$ TL kuralı.

---

### 5. Hipotezler, Ölçütler ve Ekonometrik Testler
1. **Fiyat Seviyesi Hata Analizi:**  
   - $SMF=PTF$ naif kuralı ile Aşama 2 yayılım regresörünün MAE ve RMSE karşılaştırması.  
   - Karesel hata üstünlüğü için Harvey-Leybourne-Newbold (HLN) testi ($H_0: \text{MSE}_{\text{Aşama 2}} \ge \text{MSE}_{\text{Naif}}$).
2. **Sınıflandırma ve Kalibrasyon:**  
   - 3-sınıflı doğruluk, Brier skoru, Test Brier Beceri Skoru (BSS), Ağırlıklı Beklenen Kalibrasyon Hatası (ECE).  
   - Günlük doğruluk farkı için Wilcoxon işaretli sıra testi ve Holm-Bonferroni düzeltmesi.
3. **Ekonomik Getiri ve Üstünlük Sınaması:**  
   - Yıllıklandırılmış birim getiri (TL/MWh).  
   - Eşleştirilmiş 7 günlük blok bootstrap ile fark analizi ($H_0: \mu_{\text{Model}} - \mu_{\text{Açık}} \le 0$).
4. **Risk ve Maruziyet Sınaması (2.000 Tekrarlı Permütasyon Testi):**  
   - Modelin aktif maruziyet oranı dondurulup 2.000 kez rastgele zamanlama permütasyonu oluşturulacak.  
   - 6 ekonometrik gösterge test edilecek: (1) Tüm Dönem P&L, (2) Aylık Oynaklık (SD), (3) Günlük %95 CVaR, (4) En Kötü Ay Getirisi, (5) Kümülatif Günlük MDD, (6) Tek ay dışlama duyarlılığı.  
   - İki yönlü $p$-değerlerine Holm-Bonferroni adım-küçültme ($m=6$) çoklu test düzeltmesi uygulanacaktır.
