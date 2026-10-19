# İLERİYE DÖNÜK DOĞRULAMA PROTOKOLÜ (PRE-ANALYSIS PLAN)
## EPİAŞ Elektrik Piyasasında SMF ve Sistem Yönü Tahmin Modellerinin Bağımsız İleriye Dönük (Out-of-Sample) Sınanması

**Tez Danışmanı:** Prof. Ali Dinler  
**Kurum:** T.C. İstanbul Medeniyet Üniversitesi  
**Protokol Tarihi:** 19 Ekim 2026  
**Dondurulmuş Git Versiyon Etiketi:** `protocol-v1` (Değişmez Kalıcı Etiket)  
**Harici Zaman Damgası:** Danışman Öğretim Üyesine İletilen Kayıt E-postası  

---

### 1. Amaç ve Kapsam
Bu protokol, lisansüstü tez çalışması kapsamında geliştirilen ve disk üzerinde dondurulan makine öğrenmesi modellerinin seçim yanlılığından (data snooping / p-hacking) tamamen arındırılmış biçimde test edilmesini garanti altına almak üzere hazırlanmıştır. 

Modeller test kümesi verisi (Ekim 2025 – Eylül 2026) üzerinde geliştirilip ayarlandığı ve 01–19 Ekim 2026 dönemi geliştirme/denetim süreçlerine konu olduğu için; gerçek ve tarafsız out-of-sample başarı ancak modeller kilitlendikten ve harici zaman damgası alındıktan sonraki ilk tam gün başından itibaren piyasada fiilen oluşacak verilerle ölçülebilir.

---

### 2. Bağımsız Değerlendirme Penceresi, İstatistiksel Güç ve Veri Akışı
- **Zaman Aralığı:** **20 Ekim 2026 00:00:00+03:00 – 31 Mart 2027 23:00:00+03:00** (163 Gün / 3.912 Saat).  
  *Gerekçe:* 19 Ekim 2026 gün içi saatleri protokolün ve harici e-posta zaman damgasının oluşturulduğu gün olduğundan, ileriye dönük bağımsız test penceresi zaman damgasını takip eden ilk tam gün olan 20 Ekim 2026 saat 00:00 itibarıyla başlatılmıştır.
  - Ekim 2026 (20–31 Ekim): 12 gün $\times$ 24 = 288 saat
  - Kasım 2026: 30 gün $\times$ 24 = 720 saat
  - Aralık 2026: 31 gün $\times$ 24 = 744 saat
  - Ocak 2027: 31 gün $\times$ 24 = 744 saat
  - Şubat 2027: 28 gün $\times$ 24 = 672 saat
  - Mart 2027: 31 gün $\times$ 24 = 744 saat  
  *Toplam:* **163 gün / 3.912 saat**.

- **İstatistiksel Güç, Örneklem Sınırlılığı ve Bilgisiz Sonuç Protokolü:**  
  163 günlük pencerede ($N = 163$ gün / yaklaşık 23 tam hafta), eşleştirilmiş günlük fark serisinin ($\text{Aşama 1} - \text{Açık}$) standart hatası yaklaşık $SE \approx 39,0$ TL/MWh seviyesindedir.  
  - Tek yönlü $\alpha = 0,05$ düzeyinde istatistiksel anlamlılık için gereken kritik fark $\approx 64,1$ TL/MWh, %80 güç ($1 - \beta = 0,80$) için gereken gerçek etki büyüklüğü ise $\approx 97,0$ TL/MWh'dir.  
  - Tezin örneklem içi test yılında gözlenen fark yalnızca $-4,41$ TL/MWh olduğundan, birincil hipotezin 163 gün sonunda $p > 0,05$ vererek **bilgisiz (inconclusive / yetersiz güç)** kalma olasılığı çok yüksektir.  
  - *Önceden Tanımlı Bilgisiz Sonuç Aksiyon Kuralı:* Eğer 163 gün sonunda birincil test $p > 0,05$ ile sonuçlanırsa:
    1. **Süre Uzatımı:** Değerlendirme süresi ara müdahale yapılmaksızın önceden tanımlı olarak tam 12 aya (19 Ekim 2027 23:00'e kadar, $N = 365$ gün / 8.760 saat) uzatılacaktır.
    2. **Eşdeğerlik Testi (TOST):** İki tek-taraflı test (Two One-Sided Tests) uygulanarak $\Delta \in [-35, +35]$ TL/MWh eşdeğerlik marjı dahilinde Aşama 1 ile naif kuralın ekonomik olarak denk olup olmadığı formel olarak test edilecektir.

- **Veri Kaynağı:** EPİAŞ Şeffaflık Platformu Resmi Web Servisleri (`eptr2` kütüphanesi):
  1. `mcp-smp-imb`: PTF, SMF, Pozitif Dengesizlik Fiyatı, Negatif Dengesizlik Fiyatı, Resmi Sistem Durumu.
  2. `load-plan`: TEİAŞ Gün Öncesi Yük Tahmini (`lep`).
  3. `wind-forecast`: TEİAŞ Gün Öncesi Rüzgar Üretim Tahmini (`forecast`).
- **Veri Bütünlüğü Kuralı:** Saatlik eksik veriler yalnızca geçmişe dönük ileri taşıma (`ffill`) ile doldurulacak, hiçbir aşamada geleceğe dönük bilgi sızıntısına (`lookahead bias`) izin verilmeyecektir.

---

### 3. Dondurulan Modeller, Kaynak Kodlar ve Değişmezlik Güvencesi
Aşağıdaki modeller hiçbir yeniden eğitim, hiperparametre güncellemesi veya kalibrasyon katmanı değişikliğine tabi tutulmaksızın dondurulmuştur:

1. **Aşama 2 Regresörü:** `models/canonical_stage2_regressor.pkl`  
   - SHA-256: `583a2912c3d0ea1ba58200030208d4aec131b6ab7d94707eec780b772cb24fa5`  
   - İlk Kayıt: 15 Ekim 2026, 21:10 (15–19 Ekim arasında model ağırlıkları hiç değişmemiştir).  
   - Algoritma: LightGBM Regressor ($S = SMF - PTF$), $n=250$, Derinlik=5, Öğrenme Oranı=0,03.
2. **Aşama 1 Sınıflandırıcısı:** `models/canonical_stage1_classifier.pkl`  
   - SHA-256: `d4aac8a6a6f4c2e9f25f786bec6e8fa1a66427585fce8f1331613ac9612429ff`  
   - İlk Kayıt: 15 Ekim 2026, 21:10 (Değişmemiştir).  
   - Algoritma: LightGBM 3-Sınıflı Yön Sınıflandırıcısı, $n=100$, Yaprak=31.
3. **Aşama 1 Platt Kalibratörü:** `models/canonical_stage1_calibrator.pkl`  
   - SHA-256: `5db315973a293f9eb43d4f01b1543167c8f143e41883bd8a9ff732ac68e5106a`  
   - İlk Kayıt: 15 Ekim 2026, 21:10 (Değişmemiştir).  
   - Algoritma: Doğrulama kümesinde uydurulmuş One-vs-Rest Lojistik Regresyon.

*Git Model ve Değerlendirme Kodu Değişmezliği Denetimi:*  
Değerlendirme kodu (`src/reproduce_all_thesis_results.py`, `src/features.py`, `src/significance_tests.py`), model dosyaları ve ham veri kümesi `protocol-v1` etiketi altında kalıcı olarak kilitlenmiştir.

---

### 4. Sınanacak Stratejiler Listesi ve Karar Kuralları

#### A. Önceden Tanımlı Birincil ve Karşılaştırmalı Stratejiler
1. **Oracle (Tam Bilgi / Teorik Tavan):** Fiili yönün tam olarak bilindiği durum ($D_t=1 \rightarrow +1$, $D_t=-1 \rightarrow -1$, $D_t=0 \rightarrow 0$).
2. **Her Zaman Açık (Çoğunluk Sınıfı Naif Kural - Birincil Benchmark):** $a_t = +1$ (Her saat sabit Uzun pozisyon).
3. **Her Zaman Fazla (Ters Kural):** $a_t = -1$ (Her saat sabit Kısa pozisyon).
4. **Lag-168 Naif Takipçi:** $a_t = D_{t-168}$ (Tam bir hafta önceki aynı saatin yönü).
5. **Aşama 1 Kalibre Modeli (GÖP Öncesi - Birincil Aday):** $\tau = 70.0$ TL sabit eşik kuralı:  
   $EV_{\text{Long}} > 70$ ve $EV_{\text{Long}} > EV_{\text{Short}} \rightarrow a_t = +1$; aksi halde simetrik Kısa veya Pasif.
6. **Aşama 2 Yayılım Modeli (GÖP Sonrası):** $\hat{S}_{L2} > 0 \rightarrow a_t = +1$, aksi halde $a_t = -1$.
7. **Aşama 2 Sınıflandırıcısı (Post-DAM EV):** $EV > 70$ TL kuralı.

#### B. Post-hoc / Keşifsel Referans Stratejiler (Sabit Katsayılarla)
8. **Deterministik PTF Eşik Kuralı (Keşifsel):**  
   $PTF_t < 2.500,00$ TL ise $a_t = +1$, aksi halde $a_t = -1$.  
   *Parametre:* 2.500,00 TL sabit eşik (Eğitim+Doğrulama dönemi medyanı).
9. **Doğrusal Yayılım Kuralı (Keşifsel):**  
   $\hat{S}_t = -0,0755 \cdot PTF_t + 105,92$. Eğer $\hat{S}_t > 0$ ise $a_t = +1$, aksi halde $a_t = -1$.  
   *Parametreler:* Eğitim+Doğrulama kümesi OLS regresyon katsayıları (eğim: -0,0755, sabit: 105,92).

---

### 5. Önceden Tanımlı Hipotez Hiyerarşisi, Fiyat Spesifikasyonu ve Test Metodolojisi

1. **Tek Birincil Hipotez (Single Primary Hypothesis):**  
   Bu protokolün tek birincil hipotezi, GÖP öncesi Aşama 1 modelinin ekonomik olarak çoğunluk kuralını geçip geçemeyeceğidir:  
   $$H_0: \mu_{\text{Aşama 1 Kalibre}} - \mu_{\text{Her Zaman Açık}} \le 0 \quad \text{vs} \quad H_1: \mu_{\text{Aşama 1 Kalibre}} - \mu_{\text{Her Zaman Açık}} > 0$$  
   - **Birincil Test Yöntemi:** Küçük örneklemde (yaklaşık 23 hafta) blok bootstrap'ın serisel bağımlılık kırılganlığını önlemek amacıyla, birincil test **haftalık blok ortalamaları üzerinde işaret-çevirme permütasyon testi (sign-flip permutation test)** olarak yürütülecek ve tek taraflı $\alpha = 0,05$ düzeyinde değerlendirilecektir. 7 günlük blok bootstrap güven aralığı destekleyici olarak sunulacaktır.
   - Diğer tüm karşılaştırmalar ikincil veya keşifsel niteliktedir.

2. **Birincil ve İkincil Fiyat Spesifikasyonları:**  
   - **Birincil Spesifikasyon (Tek Temel Seçim):** EPİAŞ Şeffaflık Platformu'nun fiilen yayımladığı ham nakit akışı serileri (`positiveImbalance`, `negativeImbalance`).  
     *Gerekçe:* Piyasa katılımcılarının teminatlandırma, faturalandırma ve fiili nakit uzlaştırmaları EPİAŞ'ın yayımladığı bu seriler üzerinden yürütülmektedir; dolayısıyla fiili piyasa gerçekliğini temsil eden tek resmi nakit akışı budur.  
   - **İkincil / Sağlamlık Spesifikasyonu:** EPDK Karar 14030 yasal formül rekonstrüksiyonu ($P_{\text{neg}}^{\text{stat}}$, $P_{\text{pos}}^{\text{stat}} \ge 0$). Veri ambarı veri kalıntısı veya tarife sapmalarına karşı sağlamlık kontrolü olarak paralel raporlanacaktır.

3. **Rejim Değişikliği Önceden Tanımlı Kuralı:**  
   20 Ekim 2026 – 31 Mart 2027 döneminde EPDK tarafından yeni bir azami fiyat limiti (tavan fiyat) veya dengesizlik katsayısı ($k/l$) kararı alınması hâlinde:  
   - Yeni tavan veya katsayılar EPDK'nın resmi yürürlük tarihi itibarıyla dinamik olarak uzlaştırma formülüne yansıtılacaktır.  
   - Dondurulan modellerin özniteliklerinde, katsayılarında veya karar eşiklerinde geriye ya da ileriye dönük hiçbir değişiklik yapılmayacaktır.

4. **Durdurma Kuralı (Pre-defined Stopping Rule):**  
   - Değerlendirme süresi 163 takvim günü ile kesin olarak sabittir.  
   - 31 Mart 2027 saat 23:00'e kadar hiçbir erken durdurma (early stopping) veya ara veri dikizleme (peeking) yapılmayacak; sonuçlar pencere tamamlandığında topluca hesaplanacaktır.

5. **İkincil Hipotezler ve Risk Sınamaları:**  
   - Fiyat seviyesi hata analizi: $SMF=PTF$ naif kuralı ile Aşama 2 yayılım regresörünün RMSE karşılaştırması ve HLN testi.  
   - Sınıflandırma ve kalibrasyon: Doğruluk, Brier skoru, BSS ve ECE.  
   - Maruziyet-eşleşmiş 2.000 tekrarlı permütasyon testi (Tüm Dönem P&L, SD, CVaR, En Kötü Ay, MDD) ve $m=6$ Holm-Bonferroni çoklu test düzeltmesi.
