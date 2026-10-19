# İLERİYE DÖNÜK DOĞRULAMA PROTOKOLÜ (PRE-ANALYSIS PLAN)
## EPİAŞ Elektrik Piyasasında SMF ve Sistem Yönü Tahmin Modellerinin Bağımsız İleriye Dönük (Out-of-Sample) Sınanması

**Tez Danışmanı:** Prof. Ali Dinler  
**Kurum:** T.C. İstanbul Medeniyet Üniversitesi  
**Protokol Tarihi:** 19 Ekim 2026  
**Dondurulmuş Git Versiyon Etiketi:** `v1.0-frozen-manifest` (Referans Commit: `3470811`, 18 Ekim 2026, 18:12)  

---

### 1. Amaç ve Kapsam
Bu protokol, lisansüstü tez çalışması kapsamında geliştirilen ve 18 Ekim 2026 tarihi itibarıyla disk üzerinde kilitlenen makine öğrenmesi modellerinin seçim yanlılığından (data snooping / p-hacking) tamamen arındırılmış biçimde test edilmesini garanti altına almak üzere hazırlanmıştır. 

Modeller test kümesi verisi (Ekim 2025 – Eylül 2026) üzerinde geliştirilip ayarlandığı ve 01–18 Ekim 2026 dönemi geliştirme/denetim süreçlerine konu olduğu için; gerçek ve tarafsız out-of-sample başarı ancak modeller kilitlendikten sonraki ilk tam gün başından itibaren piyasada fiilen oluşacak verilerle ölçülebilir.

---

### 2. Bağımsız Değerlendirme Penceresi, İstatistiksel Güç ve Veri Akışı
- **Zaman Aralığı:** **19 Ekim 2026 00:00:00+03:00 – 31 Mart 2027 23:00:00+03:00** (164 Gün / 3.936 Saat).  
  *Gerekçe:* 01–18 Ekim 2026 dönemi geliştirme, denetim ve model dondurma takviminde yer aldığından, ileriye dönük bağımsız test penceresi modellerin dondurulmasını takip eden ilk tam gün olan 19 Ekim 2026 saat 00:00 itibarıyla başlatılmıştır.
  - Ekim 2026 (19–31 Ekim): 13 gün $\times$ 24 = 312 saat
  - Kasım 2026: 30 gün $\times$ 24 = 720 saat
  - Aralık 2026: 31 gün $\times$ 24 = 744 saat
  - Ocak 2027: 31 gün $\times$ 24 = 744 saat
  - Şubat 2027: 28 gün $\times$ 24 = 672 saat
  - Mart 2027: 31 gün $\times$ 24 = 744 saat  
  *Toplam:* **164 gün / 3.936 saat**.
- **İstatistiksel Güç ve Örneklem Sınırlılığı Uyarısı:**  
  164 günlük pencerede ($N = 164$ gün), eşleştirilmiş günlük fark serisinin ($\text{Aşama 1} - \text{Açık}$) standart hatası yaklaşık $SE \approx 39,0$ TL/MWh seviyesindedir (365 günlük tam yılda $SE \approx 26,5$ TL/MWh idi). Bu durum, %95 güven aralığı yarı-genişliğinin (margin of error) yaklaşık $\pm 76,5$ TL/MWh olacağı anlamına gelmektedir. Dolayısıyla 164 günlük pencere, ancak Aşama 1'in tüm yıllık getirisinden ($67,33$ TL/MWh) daha büyük devasa bir etki büyüklüğünü ($\Delta > 77$ TL/MWh) istatistiki olarak anlamlı biçimde tespit edebilecek bir güce sahiptir. Bu istatistiksel güç kısıtı önceden kabul edilmiştir.
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

*Git Model Değişmezliği Denetimi:*  
`git diff 3470811..HEAD models/` çıktısı boştur; `models/` altındaki 3 kanonik model ve `data/raw/` altındaki ham veri kütüğü 18 Ekim 2026 18:12 commit'inden bu yana tek bir bit dahi değişmemiştir.

---

### 4. Sınanacak Stratejiler Listesi ve Karar Kuralları
1. **Oracle (Tam Bilgi / Teorik Tavan):** Fiili yönün tam olarak bilindiği durum ($D_t=1 \rightarrow +1$, $D_t=-1 \rightarrow -1$, $D_t=0 \rightarrow 0$).
2. **Her Zaman Açık (Çoğunluk Sınıfı Naif Kural):** $a_t = +1$ (Her saat sabit Uzun pozisyon).
3. **Her Zaman Fazla (Ters Kural):** $a_t = -1$ (Her saat sabit Kısa pozisyon).
4. **Lag-168 Naif Takipçi:** $a_t = D_{t-168}$ (Tam bir hafta önceki aynı saatin yönü).
5. **Aşama 1 Kalibre Modeli (GÖP Öncesi):** $\tau = 70.0$ TL sabit eşik kuralı:  
   $EV_{\text{Long}} > 70$ ve $EV_{\text{Long}} > EV_{\text{Short}} \rightarrow a_t = +1$; aksi halde simetrik Kısa veya Pasif.
6. **Aşama 2 Yayılım Modeli (GÖP Sonrası):** $\hat{S}_{L2} > 0 \rightarrow a_t = +1$, aksi halde $a_t = -1$.
7. **Aşama 2 Sınıflandırıcısı (Post-DAM EV):** $EV > 70$ TL kuralı.

---

### 5. Önceden Tanımlı Hipotez Hiyerarşisi, Fiyat Spesifikasyonu ve Protokol Kuralları

1. **Tek Birincil Hipotez (Single Primary Hypothesis):**  
   Bu protokolün tek birincil hipotezi, Aşama 1 modelinin ekonomik olarak çoğunluk kuralını geçip geçemeyeceğidir:  
   $$H_0: \mu_{\text{Aşama 1 Kalibre}} - \mu_{\text{Her Zaman Açık}} \le 0 \quad \text{vs} \quad H_1: \mu_{\text{Aşama 1 Kalibre}} - \mu_{\text{Her Zaman Açık}} > 0$$  
   Test istatistiği eşleştirilmiş 7 günlük blok bootstrap ile hesaplanacak ve tek taraflı $\alpha = 0,05$ anlamlılık düzeyinde değerlendirilecektir. Diğer tüm metrikler ikincil veya keşifsel niteliktedir.

2. **Birincil ve İkincil Fiyat Spesifikasyonları:**  
   - *Birincil Spesifikasyon:* EPİAŞ Şeffaflık Platformu'nun fiilen yayımladığı ham nakit akışı serileri (`positiveImbalance`, `negativeImbalance`).
   - *İkincil / Sağlamlık Spesifikasyonu:* EPDK Karar 14030 yasal formül rekonstrüksiyonu ($P_{\text{neg}}^{\text{stat}}$, $P_{\text{pos}}^{\text{stat}}$).

3. **Rejim Değişikliği Önceden Tanımlı Kuralı:**  
   19 Ekim 2026 – 31 Mart 2027 döneminde EPDK tarafından yeni bir azami fiyat limiti (tavan fiyat) veya dengesizlik katsayısı ($k/l$) kararı alınması hâlinde:  
   - Yeni tavan veya katsayılar EPDK'nın resmi yürürlük tarihi itibarıyla dinamik olarak uzlaştırma formülüne yansıtılacaktır.  
   - Dondurulan modellerin özniteliklerinde, katsayılarında veya karar eşiklerinde geriye ya da ileriye dönük hiçbir değişiklik yapılmayacaktır.

4. **Durdurma Kuralı (Pre-defined Stopping Rule):**  
   - Değerlendirme süresi 164 takvim günü ile kesin olarak sabittir.  
   - 31 Mart 2027 saat 23:00'e kadar hiçbir erken durdurma (early stopping) veya ara veri dikizleme (peeking) yapılmayacak; sonuçlar pencere tamamlandığında topluca hesaplanacaktır.

5. **İkincil Hipotezler ve Risk Sınamaları:**  
   - Fiyat seviyesi hata analizi: $SMF=PTF$ naif kuralı ile Aşama 2 yayılım regresörünün RMSE karşılaştırması ve HLN testi.  
   - Sınıflandırma ve kalibrasyon: Doğruluk, Brier skoru, BSS ve ECE.  
   - Maruziyet-eşleşmiş 2.000 tekrarlı permütasyon testi (Tüm Dönem P&L, SD, CVaR, En Kötü Ay, MDD) ve $m=6$ Holm-Bonferroni çoklu test düzeltmesi.
