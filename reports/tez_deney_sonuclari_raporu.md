# T.C. İSTANBUL MEDENİYET ÜNİVERSİTESİ
## LİSANSÜSTÜ TEZ ÇALIŞMASI: EPİAŞ ELEKTRİK PİYASASINDA SİSTEM MARJİNAL FİYATI (SMF) VE SİSTEM YÖNÜ TAHMİNİ
### Deneysel Bulgular, Ekonometrik Doğrulama ve Tartışma Bölümü

**Tez Danışmanı:** Prof. Ali Dinler  
**Çalışma Alanı:** Türkiye Elektrik Piyasası (EPİAŞ Gün Öncesi Piyasası - GÖP / TEİAŞ Dengeleme Güç Piyasası - DGP)  
**Hedef Horizon:** 12 - 36 Saat İleriye Dönük (11:30 Karar Anı / 12:30 GÖP Teklif Kapanışı Öncesi)  
**Kanonik Doğrulama Komutları:** `python3 src/canonical_reconciliation.py` ve `python3 src/reproduce_all_thesis_results.py`  
**Veri Dosyası:** `data/raw/epias_official_market_data_full.parquet`  
**SHA-256 Sağlama Toplamı:** `e4748553a38b4fff3f013b75b1b473190f35059972c984f9e4192b4eef941dd5`  
**Rastgele Tohum (Seed):** 42 (Tüm model eğitimlerinde ve bootstrap örneklemelerinde sabitlenmiştir)

---

## 1. Tahmin Mimarisi, Geliştirme Süreci ve Metodolojik Sınırlılıklar

### 1.1. Piyasa Zaman Kısıtları ve Çift Aşamalı Karar Mimarisi
Türkiye elektrik piyasası işleyişinde Gün Öncesi Piyasası (GÖP) teklifleri her gün saat 12:30'da kapanmakta; piyasa takas fiyatı ($PTF$) ise itiraz sürecinin tamamlanmasıyla kesinleşmektedir. Dengeleme Güç Piyasası'nda (DGP) oluşan Sistem Marjinal Fiyatı ($SMF$) ve Sistem Denge Yönü ($D_t$) ise fiziksel teslimat gününün her saati için gerçek zamanlı olarak TEİAŞ tarafından belirlenir. Bu zaman kısıtları tez kapsamında iki ayrı karar anı ve aşaması olarak modellenmiştir:

1. **Aşama 1 (GÖP Öncesi - Saat 11:30):** $PTF$ henüz bilinmemektedir (GÖP kapanışından 1 saat önce portföy optimizasyonu için karar üretilir).
   - **Yön Sınıflandırması:** 3 sınıflı LightGBM sınıflandırıcısı (`ThreeClassDirectionClassifier`), TEİAŞ net talimat yönünü (Enerji Açığı: $+1$, Dengede: $0$, Enerji Fazlası: $-1$) tahmin eder.
   - **Doğrudan Fiyat Regresyonu:** `DirectSMFRegressor`, PTF girdisi olmaksızın makroekonomik göstergeler, hava durumu ve yük tahminleriyle ham $SMF$ seviyesini tahmin eder.
2. **Aşama 2 (GÖP Sonrası - Saat 14:00):** $PTF$ kesinleşmiş olup deterministik bir dışsal girdidir.
   - Model doğrudan fiyat seviyesi yerine fiyat yayılımını ($S_t = SMF_t - PTF_t$) hedefler (`PostDAMSpreadRegressor`).
   - Nihai SMF tahmini: $\widehat{SMF}_t = PTF_t + \widehat{S}_t$ olarak elde edilir.
   - **Ekonometrik Özdeşlik:** $\widehat{SMF}_t - SMF_t = (PTF_t + \widehat{S}_t) - (PTF_t + S_t) = \widehat{S}_t - S_t$ olduğundan, yayılım regresyonunun hata metrikleri doğrudan SMF fiyat tahmininin hata metriklerine özdeştir.

---

### 1.2. Model Geliştirme Süreci, İterasyonlar ve Seçim Yanlılığı (Data Snooping) Açıklığı

Bilimsel dürüstlük ve ekonometrik şeffaflık gereğince, tez çalışmasında kullanılan modellerin tek seferde ve test kümesi hiç incelenmeden kilitlendiği iddia edilemez. Modelleme sürecinde hem 2025 ilk 9 ayı (doğrulama kümesi) hem de Ekim 2025 – Eylül 2026 dönemi (test kümesi) defalarca incelenmiş; hiperparametre seçimleri, kalibrasyon katmanları ve karar kuralları bu gözlemler ışığında şekillendirilmiştir:

#### Tablo 1: Model Geliştirme Süreci ve Sürümler Arası İterasyon Geçmişi

| Geliştirme Aşaması | İnceleme Kümesi ve Kurgu | Temel Metodolojik Değişiklikler | Test Dönemi Aşama 1 P&L | Test Dönemi Aşama 2 P&L | Metodolojik Not ve Sınırlılıklar |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Aşama 0 (İlk Geliştirme)** | 2025 İlk 9 Ay ($N=6.552$) | İlk modeller bu kümede eğitildi. $SMF=PTF$ naif kuralının üstünlüğü burada görüldü, yayılım regresyonu ve sert kısıt kuralları bu kümede denendi. | — | — | Bu küme daha sonra "doğrulama" kümesi yapılmıştır; ancak geliştirme kararları bu veriye bakılarak alındığından tamamen el değmemiş bir doğrulama değildir. |
| **Sürüm 1 (Ham Seviye Modeli)** | Test Dönemi (8.637 saat) | Doğrudan SMF seviye regresyonu, ikili yön sınıflandırması, basitleştirilmiş ceza katsayısı formülü. | +344,03 TL/MWh | — | 123 saatlik veri boşluğu bulunmaktaydı; formül basitleştirmesi P&L'i yapay olarak yüksek gösterdi. |
| **Sürüm 2 ($n=100$ Yayılım Modeli)** | Test Dönemi (8.760 saat) | Seviye yerine yayılım ($S=SMF-PTF$) modeli ($n=100$), 3-sınıflı DUY yönü, ffill ile 8.760 tam saat. | +55,95 TL/MWh (Ham) | +135,58 TL/MWh ($n=100$) | Aşama 2 rejim değerleri (87,09 / 139,85 / 158,16 TL) ağırlıklı ortalamada +135,58 TL getirdi. |
| **Sürüm 3 ($n=250$ ve Kalibrasyon)** | Test Dönemi (8.760 saat) | Aşama 2 hiperparametreleri $n=250$, derinlik 5 olarak güncellendi. Aşama 1'e Platt kalibrasyonu ve $\tau=70$ TL eşiği eklendi. | +70,55 TL/MWh (Validasyon-Platt) | +119,14 TL/MWh ($n=250$) | Aşama 2 P&L'i 135,58'den 119,14 TL'ye geriledi; Aşama 1 Platt doğrulama eşiği incelemesinde +70,55 TL verdi (ham OOF değerleri 59,15 ve 70,58 TL'dir). |
| **Sürüm 4 (Kanonik Mutabakat)** | Test Dönemi (8.760 saat) | EPİAŞ fiili `negativeImbalance`/`positiveImbalance` serileri bağlandı, Rejim 1-3 tam saat ağırlıklı mutabakatı sağlandı. Modeller diske kaydedildi (15 Ekim 2026, 21:10). | **+67,33 TL/MWh** | **+119,14 TL/MWh** | Tüm tablolar ve kodlar makine hassasiyetinde mutabık kılındı. Disk üzerindeki kanonik dosya kimlikleri donduruldu. |
| **Post-hoc Teşhis Modelleri (Keşifsel)** | Test Dönemi (8.760 saat) | Aşama 2'nin mekanizmasını ayrıştırmak üzere test kümesi sonuçları görüldükten sonra $PTF < 2.500$ TL eşik kuralı (+140,57 TL) ve doğrusal regresyon kuralları türetildi. | — | +140,57 TL (PTF<2.500) | Önceden kilitlenmiş model değildir; test kümesi sonuçlarına bakılarak geliştirilmiş geriye dönük (post-hoc) teşhis kurallarıdır. |

> [!NOTE]
> **Versiyon Kontrolü, Model Kütüğü ve Tarihsel Bütünlük:** Modeller 15 Ekim 2026 21:10 itibarıyla eğitilerek disk üzerine (`models/canonical_*.pkl`) kaydedilmiş ve SHA-256 sağlama toplamları ilk rapor kütüğünde sabitlenmiştir. 18 Ekim 2026'da çalışma dizini Git versiyon kontrolüne alınmış; 20 Ekim 2026'da açıklamalı `protocol-v1` etiketi oluşturulmuştur. Bu süreçte modellerin fiziksel boyutları (537.896 B, 2.058.822 B, 911 B) ve SHA-256 özetleri değişmeksizin korunmuş, 15–20 Ekim tarihli rapor ve disk kopyalarıyla mutabakatı doğrulanmıştır. Kriptografik SHA-256 manifestosu (`MANIFEST_SHA256.txt`) ve 21 Ekim 2026 – 31 Mart 2027 dönemini (162 gün / 3.888 saat) kapsayan bağımsız ileriye dönük doğrulama protokolü (`reports/prospective_evaluation_protocol_2026_2027.md`), dondurulan commit hash'i ve arşiv özetiyle birlikte harici bir zaman damgası tescil etmek üzere danışman öğretim üyesine e-posta ile iletilecektir.

#### Tablo 2: Kanonik Model Manifestosu (Kriptografik SHA-256 Sağlama Toplamları)

| Model Dosyası | Algoritma ve Kurgu | Hiperparametreler | Disk Boyutu | SHA-256 Sağlama Toplamı |
| :--- | :--- | :--- | :---: | :--- |
| `models/canonical_stage2_regressor.pkl` | LightGBM Regressor (Yayılım: $S = SMF - PTF$) | `n_est=250, lr=0.03, depth=5, min_child=30` | 537.896 B | `583a2912c3d0ea1ba58200030208d4aec131b6ab7d94707eec780b772cb24fa5` |
| `models/canonical_stage1_classifier.pkl`| LightGBM 3-Sınıflı Sınıflandırıcı (Yön) | `n_est=100, lr=0.05, num_leaves=31, seed=42` | 2.058.822 B | `d4aac8a6a6f4c2e9f25f786bec6e8fa1a66427585fce8f1331613ac9612429ff` |
| `models/canonical_stage1_calibrator.pkl`| Platt Kalibratörü (Lojistik Dönüştürücü) | One-vs-Rest Lojistik Regresyon (`lbfgs`) | 911 B | `5db315973a293f9eb43d4f01b1543167c8f143e41883bd8a9ff732ac68e5106a` |

---

### 1.3. Fiyat Seviyesi Hata Analizi: $L_1$ Medyan vs $L_2$ Koşullu Ortalama ve HLN Testi

#### Tablo 3: Fiyat Seviyesi Tahmin Başarımı: Doğrulama ve Test Kümeleri Karşılaştırması

| Örneklem | Model / Referans Kurgusu | MAE ($L_1$ - TL/MWh) | RMSE ($L_2$ - TL/MWh) | Günlük Ekonometrik Hipotez Testi | Strateji P&L (TL/MWh) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Doğrulama Kümesi** | **Modelsiz $SMF = PTF$ Naif Kuralı ($\hat{S} = 0$)** | **476,41** | 727,11 | — (Referans Kural) | -57,52 TL (Açık P&L) |
| ($N = 6.552$ Saat) | **GÖP Öncesi Doğrudan SMF Modeli (`DirectSMF`)** | 683,33 | 829,88 | $DM_{\text{stat}} = -8,41$, $p < 0,0001$ (MAE bazlı DM, Naif üstün) | — |
| (2025 İlk 9 Ay) | **Aşama 2: GÖP Sonrası Yayılım Modeli ($n=250$)** | 514,53 | **720,87** | $HLN_{\text{stat}} = 1,01$, $p = 0,3140$ (MSE bazlı HLN, Üstünlük yok) | **+98,28 TL** |
| **Test Kümesi** | **Modelsiz $SMF = PTF$ Naif Kuralı ($\hat{S} = 0$)** | **682,75** | 1.019,49 | — (Referans Kural) | +71,73 TL (Açık P&L) |
| ($N = 8.760$ Saat) | **GÖP Öncesi Doğrudan SMF Modeli (`DirectSMF`)** | 983,62 | 1.215,25 | $DM_{\text{stat}} = -11,29$, $p < 0,0001$ (MAE bazlı DM, Naif üstün)| +55,95 TL (Aşama 1 Ham) |
| (12 Ay / 365 Gün) | **Aşama 2: GÖP Sonrası Yayılım Modeli ($n=250$)** | 748,14 | **1.007,01** | **$HLN_{\text{stat}} = 1,1026$, $p = 0,2702$ (MSE bazlı HLN, Üstünlük yok)**| **+119,14 TL** |

*Ekonometrik Açıklama ve İstatistiki Çıkarımlar:*
1. **$L_1$ Medyan Avantajı:** Test kümesinde saatlerin %21,56'sında (1.889 saat) $PTF = SMF$ gerçekleşmiş olup yayılım $S = 0,00$ TL'dir. Mutlak hata kaybı ($L_1$), koşullu medyan tahmini ile minimize edilir. Sıfırda büyük yığılma gösteren dağılımda sabit $\hat{S} = 0$ ($SMF = PTF$) tahmini, MAE'yi en düşük seviyede tutmaktadır (682,75 TL vs 748,14 TL; günlük Diebold-Mariano $DM_{\text{stat}} = -7,001$, $p < 0,0001$).
2. **$L_2$ Koşullu Ortalama ve İstatistiki Üstünlük Yokluğu (HLN Testi):** LightGBM regresyonu karesel hatayı ($L_2$) minimize etmek üzere koşullu ortalamayı hedefler. Model ekstrem fiyat hareketlerini yakalayarak test RMSE'sini 1.019,49 TL'den 1.007,01 TL'ye (12,48 TL kazanç) düşürmüştür. Ancak Newey-West HAC otokorelasyon düzeltmeli ve küçük örneklem uyarlamalı **Harvey-Leybourne-Newbold (HLN) testi**, bu karesel hata farkının istatistiki olarak anlamlı bir üstünlük taşımadığını göstermektedir ($HLN_{\text{stat}} = 1,1026$, **$p = 0,2702 > 0,05$**).
3. **P&L Seviye Artışının Spread Genişliğiyle Tutarlılığı:** Doğrulama kümesinden test kümesine geçildiğinde Aşama 2 P&L'inin +98,28 TL'den +119,14 TL'ye yükselmesi model başarısı değildir. Sabit ceza katsayılarıyla ($k=0,03$) simüle edildiğinde dahi test P&L'i yükselmektedir (Doğrulama: 98,28 TL $\rightarrow$ Test: 136,39 TL). Seviye artışı, test yılındaki fiyat oynaklığı ve yayılım genişliğinin (ortalama $|SMF - PTF|$: 476 TL $\rightarrow$ 683 TL) çok daha yüksek olmasıyla tutarlıdır. Modelin naif kurala göre göreli üstünlüğü ise doğrulamada **+155,80 TL** iken (+98,28 vs -57,52 TL), test döneminde **+47,40 TL**'ye (+119,14 vs +71,73 TL) belirgin biçimde gerilemiştir.

---

### 1.4. Aşama 2 Karar Kuralı İncelemesi: $L_2$ Regresyon İşareti vs Alternatif Modeller

Aşama 2 modelinin kanonik P&L değeri (+119,14 TL/MWh), $L_2$ yayılım regresyonunun işaret kuralına ($\hat{S}_{L2} > 0$ ise Uzun, aksi halde Kısa) dayanmaktadır. GÖP sonrası 3-sınıflı yön sınıflandırıcısı, Aşama 1'e göre hem daha yüksek doğruluğa (0,6221 vs 0,5966) hem de belirgin biçimde daha iyi Brier skoruna (BSS %12,33 vs %6,43) sahiptir. Ancak aynı marjinal beklenen değer kuralı ($\tau=70$ TL) uygulandığında sınıflandırıcının P&L'i **-20,81 TL/MWh** seviyesinde kalmakta (doğrulama kümesinde ayrıca optimize edilen $\tau=140$ TL eşiğinde dahi **-7,46 TL/MWh**, argmax kuralında ise **-33,18 TL/MWh** zarar üretmekte); buna karşılık $L_2$ yayılım regresörü **+119,14 TL/MWh** getiri sağlamaktadır.

#### Tablo 4: Aşama 2 İçin Farklı Karar Kurallarının P&L Karşılaştırması

| Aşama 2 Karar Kurgusu | Karar Mantığı | Test P&L (TL/MWh) | Ekonometrik Açıklama |
| :--- | :--- | :---: | :--- |
| **1. Kanonik $L_2$ Ortalama Regresyonu** | $\hat{S}_{L2} > 0$ ise Uzun, $\le 0$ ise Kısa | **+119,14 TL** | Pozitif tepe fiyatların büyüklüğünü koşullu ortalamada yakalar. |
| **2. $L_1$ Medyan Regresyonu (Quantile 0,5)** | $\hat{S}_{L1} > 0$ ise Uzun, $\le 0$ ise Kısa | **+29,57 TL** | Medyan $PTF=SMF$ saatlerinde 0'a çekilir, tepe büyüklükleri dışlar. |
| **3. Post-DAM Yön Sınıflandırıcısı (Argmax)**| En yüksek olasılıklı yön sınıfı | **-33,18 TL** | Sınıf olasılığı hedefler, beklenmeyen tepe saatlerde terste kalır. |
| **4. Post-DAM Kalibre Beklenen Değer ($\tau=70$)**| $EV_{\text{Long}} > 70$ TL (Aşama 1 eşiği) | **-20,81 TL** | Marjinal ceza eşiğiyle pozisyon alır, pasiflik zararı sınırlar. |
| **5. Post-DAM Kalibre Beklenen Değer (Val-Tuned $\tau=140$)**| Doğrulamada optimize edilmiş $\tau=140$ TL | **-7,46 TL** | Aşama 2 için özel optimize edilen eşik de zararı sıfırlayamaz. |

---

#### Tablo 4B: P&L Saat Türleri Hiyerarşik Ayrık Atıf ve Net Katkı Tablosu (N = 8.760 Saat / 365 Gün)

*Hiyerarşik Öncelik Sıralaması (Kesişimsiz Ayrık Partisyon):*  
1. **Üst %1 Ekstrem Yayılım:** $|SMF_t - PTF_t| \ge 3.101,66$ TL olan tüm saatler (88 saat).  
2. **Tavan Fiyat Saatleri:** 1. dilim dışındaki tavan saatleri (2.113 saat; toplam 2.114 tavan saatinin 1'i üst %1 dilimindedir).  
3. **Anomali Saatleri (-106 / -103 TL):** 1. ve 2. dilim dışındaki anomali saatleri (1.534 saat; toplam 1.538 anomali saatinin 4'ü üst %1 dilimindedir: 2 saat -106, 2 saat -103 TL; tavan saatlerinde anomali yoktur).  
4. **$PTF = SMF$ Saatleri:** Yukarıdaki dilimler dışındaki sıfır yayılım saatleri (664 saat; test yılındaki toplam 1.852 sıfır yayılım saatinin 817'si tavan diliminde [Kategori 2], 371'i anomali diliminde [Kategori 3], kalan 664'ü bu kategoridedir).  
5. **Geri Kalan Normal Piyasa Saatleri:** Kalan normal saatler (4.361 saat).  
*(Toplam: $88 + 2.113 + 1.534 + 664 + 4.361 = 8.760$ saat).*

| Saat Türü / Piyasa Dilimi | Saat ($N$) | Pay (%) | Her Zaman Açık (Ort. TL) | Aşama 1 Kalibre (Ort. TL) | Aşama 2 $L_2$ Regresyon (Ort. TL) | Aşama 2 $L_1$ Medyan (Ort. TL) | Aşama 2 Sınıflandırıcı ($EV>70$, Ort. TL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Üst %1 Ekstrem Yayılım ($|SMF - PTF| \ge 3.101,66$ TL)** | 88 | %1,00 | +1.741,70 | +908,78 | **+2.131,96** | +1.311,73 | **-2.088,43** |
| **2. Tavan Fiyat Saatleri ($PTF$ veya $SMF$ Tavanında)** | 2.113 | %24,12 | +227,86 | +46,73 | -78,25 | +155,11 | +13,53 |
| **3. -106 / -103 TL Platform Anomali Saatleri** | 1.534 | %17,51 | -432,05 | -33,54 | -103,20 | +54,26 | +129,06 |
| **4. $PTF = SMF$ Sıfır Yayılım Saatleri** | 664 | %7,58 | +59,02 | +42,98 | -45,16 | -44,20 | +51,80 |
| **5. Geri Kalan Normal Piyasa Saatleri** | 4.361 | %49,78 | +141,53 | +99,52 | **+277,38** | -54,58 | -59,49 |
| **GENEL YILLIK ORTALAMA / TOPLAM** | **8.760** | **%100,0** | **+71,73 TL** | **+67,33 TL** | **+119,14 TL** | **+29,57 TL** | **-20,81 TL** |

##### Toplam Yıllık P&L'e Net Katkı Dökümü (Birim: TL/MWh cinsinden saat ağırlıklı pay: $\text{Ortalama} \times N / 8.760$)
| Saat Türü | Saat ($N$) | Açık Katkı | Aşama 1 Katkı | Aşama 2 $L_2$ Katkı | Aşama 2 $L_1$ Katkı | Aşama 2 Clf ($EV>70$) Katkı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Üst %1 Ekstrem Yayılım** | 88 | +17,50 TL | +9,13 TL | **+21,42 TL** | +13,18 TL | **-20,98 TL** |
| **2. Tavan Fiyat Saatleri** | 2.113 | +54,96 TL | +11,27 TL | -18,87 TL | +37,41 TL | +3,26 TL |
| **3. -106 / -103 TL Anomali Saatleri** | 1.534 | -75,66 TL | -5,87 TL | -18,07 TL | +9,50 TL | +22,60 TL |
| **4. $PTF = SMF$ Saatleri** | 664 | +4,47 TL | +3,26 TL | -3,42 TL | -3,35 TL | +3,93 TL |
| **5. Geri Kalan Normal Saatler** | 4.361 | +70,46 TL | +49,54 TL | **+138,09 TL** | -27,17 TL | -29,62 TL |
| **TOPLAM YILLIK P&L (TL/MWh)** | **8.760** | **+71,73 TL** | **+67,33 TL** | **+119,14 TL** | **+29,57 TL** | **-20,81 TL** |

##### Saat Türlerine Göre Model Eylem Payları, Doğruluk ve Aktif İsabet Oranları
| Saat Türü / Piyasa Dilimi | Saat ($N$) | Aşama 2 $L_2$ Eylem Dağılımı | $L_2$ Doğruluk | Aşama 2 Sınıflandırıcı ($EV>70$) Eylem Dağılımı | Sınıflandırıcı 3-Sınıflı Doğruluk (Pasif=$D_0$ Doğru) | Sınıflandırıcı Doğruluk (Pasif=Hata) | Sınıflandırıcı Aktif İsabet (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Üst %1 Ekstrem Yayılım** | 88 | %80,7 Uzun / %19,3 Kısa | **%78,4** | %29,5 Uzun / %31,8 Kısa / %38,6 Pasif | **%2,3** | **%2,3** | **%3,7 (2 / 54 aktif)** |
| **2. Tavan Fiyat Saatleri** | 2.113 | %8,6 Uzun / %91,4 Kısa | %20,3 | %68,4 Uzun / %1,5 Kısa / %30,1 Pasif | %58,6 | %58,3 | %83,4 |
| **3. -106 / -103 TL Anomali Saatleri** | 1.534 | %69,5 Uzun / %30,5 Kısa | %33,1 | %1,8 Uzun / %66,5 Kısa / %31,7 Pasif | %62,3 | %62,3 | %91,3 |
| **4. $PTF = SMF$ Sıfır Yayılım Saatleri** | 664 | %34,6 Uzun / %65,4 Kısa | %32,5 | %34,0 Uzun / %9,0 Kısa / %56,9 Pasif | **%50,8** | **%32,7** | %75,9 (217 / 286 aktif) |
| **5. Geri Kalan Normal Piyasa Saatleri** | 4.361 | %47,0 Uzun / %53,0 Kısa | **%55,8** | %29,2 Uzun / %14,1 Kısa / %56,7 Pasif | **%25,3** | **%25,3** | **%58,4** |
| **GENEL TEST YILI TOPLAMI** | **8.760** | **%41,1 Uzun / %58,9 Kısa** | **%41,7** | **%34,2 Uzun / %20,0 Kısa / %45,7 Pasif** | **%41,5** | **%40,1** | **%73,9 (3.510 / 4.753 aktif)** |

> [!NOTE]
> **Metodolojik Not – Doğruluk Tanımları ve Kategori 4 Ayrıştırması:**  
> - **$L_2$ Regresör Doğruluğu (4. sütun):** $L_2$ model hiçbir zaman Pasif kalmaz; doğruluk = tüm saatlerdeki fiili yönle ($D_t$) işaret eşleşmesidir.  
> - **Sınıflandırıcı 3-Sınıflı Doğruluk (6. sütun, %41,5):** Standart 3-sınıflı sınıflandırma metriğidir ($\hat{y}_t = y_t$). Sınıflandırıcının Pasif kaldığı saatler Dengede eylemi ($a_t = 0$) kabul edilir. Sistem fiilen Dengede ($D_t=0$) olduğunda Pasif kalmak **doğru tahmin** sayılır ($126 / 159$ saat eşleşmiştir). Kategori 4'teki 664 saatin 151'i $D_t=0$'dır; model bu 151 saatin 120'sinde Pasif kalmış ve doğru sayılmıştır. Aktif pozisyondaki 217 doğru ile birlikte toplam $217 + 120 = 337$ doğru elde edilmiş ve 3-sınıflı doğruluk **$337 / 664 = \%50,8$** çıkmıştır. Yıllık genel toplamda $(3.510 + 126) / 8.760 = \mathbf{\%41,5}$'tir.  
> - **Sınıflandırıcı Doğruluk (Pasif=Hata, 7. sütun, %40,1):** Pasif kalmanın her zaman hata kabul edildiği katı ticaret metriğidir ($a_t = y_t \land a_t \ne 0$). Bu tanımda Kategori 4'te yalnızca aktif saatlerdeki doğrular ($217$ saat) paya yazılır ve doğruluk **$217 / 664 = \%32,7$** olur ($= \text{Aktif Pay \%43,1} \times \text{Aktif İsabet \%75,9}$). Yıllık genel toplamda ise $3.510 / 8.760 = \mathbf{\%40,1}$'dir.  
> - **Sınıflandırıcı Aktif İsabet (8. sütun):** Modelin fiilen pozisyon aldığı aktif saatlerdeki yön isabetidir ($3.510 / 4.753 = \%73,9$; model yıllık saatlerin %54,3'ünde [2.999 Uzun + 1.754 Kısa = 4.753 saat] aktif pozisyona girmiş, 4.007 saatte pasif kalmış ve aktif olduğu saatlerin 3.510'unda doğru yönde kalmıştır; Kategori 4'te $217 / 286 = \%75,9$).  
> - **Tablo 9 Karşılaştırması:** Tablo 9'daki %62,21 değeri argmax kuralıyla (Pasif seçeneği olmaksızın, her saat zorunlu 3 sınıftan en yüksek olasılıklıya atanarak) elde edilen 3-sınıflı doğruluktur. Tablo 4B'deki değerler ise $\tau=70$ TL marjinal eşiği altındaki ekonomik kararları yansıtmaktadır.

> [!IMPORTANT]
> **P&L Saat Türleri Atıf Analizi ve Aşama 2 Çelişkisinin Ekonometrik Ayrıştırması:**  
> Tablo 4B, Aşama 2 $L_2$ regresörü (+119,14 TL) ile sınıflandırıcısı (-20,81 TL) arasındaki **139,95 TL/MWh**'lik net getiri farkının kaynaklarını ampirik olarak ortaya koymaktadır:  
> 1. **Farkın Ana Kaynağı Normal Piyasa Saatleridir (+167,71 TL Net Fark Katkısı, %119,8 Pay):**  
>    İki model arasındaki fark tek başına ekstrem sıçrama saatlerinden türememektedir. 4.361 normal piyasa saatinde (%49,78), $L_2$ regresörü saatlerin %47,0'sinde Uzun pozisyona geçerek **+138,09 TL** net getiri katkısı üretmiştir. Sınıflandırıcı model ise yön olasılıkları marjinal ceza eşiğini ($\tau=70$ TL) aşamadığından normal saatlerin %56,7'sinde pasif kalarak ve aldığı pozisyonlarda -59,49 TL ortalama zarara uğrayarak **-29,62 TL** kaybettirmiştir. Aradaki **+167,71 TL'lik net fark**, toplam 139,95 TL'lik farkın **%119,8'ini tek başına** oluşturmaktadır.  
> 2. **Üst %1 Ekstrem Yayılım Diliminin Rolü ve Sınıflandırıcının Çöküşü (+42,40 TL Net Fark Katkısı, %30,3 Pay):**  
>    $|SMF - PTF| \ge 3.101,66$ TL olan 88 saatlik küme simetrik mutlak yayılım ile tanımlandığından her iki kuyruğu birden içermektedir: **64 saat pozitif sıçrama** ($SMF > PTF$, ortalama PTF: 444 TL, SMF: 3.950 TL) ve **24 saat negatif çöküş** ($SMF < PTF$, ortalama PTF: 4.098 TL, SMF: 552 TL).  
>    - $L_2$ regresörü bu 88 saatte **%78,4 yön doğruluğuna** (71 Uzun, 17 Kısa) ulaşarak pozitif sıçramaları yakalamış ve **+2.131,96 TL/MWh (genel P&L'e +21,42 TL net katkı)** sağlamıştır.  
>    - Buna karşılık sınıflandırıcı model bu saatlerde aktif pozisyon aldığı 54 saatin yalnızca 2'sinde doğru yönü bulabilmiş (**%3,7 aktif isabet**) ve **-2.088,43 TL/MWh (-20,98 TL ceza katkısı)** kaybettirmiştir.  
>    - *Sınıflandırıcının Yapısal Terslik Mekanizması:* Model, eğitim verisinde $PTF$ fiyatını sistem sıkılığı vekili olarak öğrenmiştir (genel korelasyon $r=+0,836$, üst %1 saatlerinde $r=+0,855$). Ancak ekstrem şok saatlerinde bu bağıntı tersine dönmektedir: Pozitif sıçramalarda gün öncesi $PTF$ 444 TL gibi çok düşük bir seviyede kaldığı için model sistemi fazla sanmakta ($P(\text{Fazla})=0,524$), pasif kalmakta veya Kısa pozisyona düşmektedir. Negatif çöküşlerde ise $PTF$ 4.098 TL tavanında olduğundan model sisteme $P(\text{Açık})=0,956$ atayarak kesin Uzun pozisyona girmekte ve devasa dengesizlik cezalarına maruz kalmaktadır. Bu durum bir etiket hatası değil; sınıflandırıcının $PTF$ bazlı sıkılık varsayımının beklenmeyen fiziksel şok saatlerinde yapısal olarak terste kalmasıdır.  
> 3. **Tavan, Anomali ve Sıfır Yayılım Dilimlerinin Dengeleyici Rolü (-70,16 TL Fark, -50,1% Pay):**  
>    Tavan fiyat saatlerinde (-22,14 TL), anomali saatlerinde (-40,67 TL) ve sıfır yayılım saatlerinde (-7,35 TL) sınıflandırıcı pasif kalarak veya doğru yönde pozisyon alarak $L_2$'ye göre toplam **-70,16 TL** daha az ceza / daha çok getiri elde etmiştir.  
> *Net Ayrıştırma Sentezi:*  
> $$+167,71 \text{ TL (Normal Saatler)} + 42,40 \text{ TL (Ekstrem Dilim)} - 70,16 \text{ TL (Kalan Saatler)} = \mathbf{+139,95 \text{ TL/MWh}}$$  
> Böylece iki model arasındaki getiri farkı ayrık bileşenleriyle tutarlı biçimde dekompoze edilmiştir.

---

### 1.5. Aşama 2 P&L Kaynağının İzolasyonu: PTF Seviye ve Ortalamaya Dönüş Kuralı Hipotezi

Aşama 2 $L_2$ regresörünün genel yön isabetinin **%41,7** ile piyasanın çoğunluk sınıfının (%55,7) ve yazı-tura şans düzeyinin (%50,0) belirgin biçimde altında kalması; modelin başarısının bir yönsel tahmin üstünlüğünden değil, elektrik piyasasındaki yapısal fiyat sınırlarından (0 tabanı ve 3.400 / 4.500 TL tavanı) kaynaklanan bir **PTF seviyesi etkisi (ortalamaya dönüş)** olabileceği hipotezini gündeme getirmiştir. Bu hipotezi doğrudan test etmek amacıyla Tablo 4C'de 3 bağımsız sınama gerçekleştirilmiştir:

#### Tablo 4C: PTF Seviye ve Ortalamaya Dönüş Hipotezi Sınaması (N = 8.760 Saat)

| Karar Kurgusu / Sınama Modeli | Dahil Edilen Öznitelikler ve Karar Kuralı | Test P&L (%95 Blok GA) | L2 İle Fark (%95 Blok GA) | Yön İsabeti (%) | Uzun Payı (%) | Ekonometrik Açıklama ve Kaynak Örneklem |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Kanonik Aşama 2 ($L_2$ LightGBM)** | Tüm GÖP sonrası öznitelikler (32 öznitelik, $n=250$) | **+119,14 TL**<br>([+61,79; +184,53]) | — (Referans Model) | %41,7 | %41,1 | $L_2$ yayılım koşullu ortalama modeli (Tablo 12 ile mutabık seed=42 CI). |
| **2. (a) Doğrusal Regresyon (PTF + Saat)** | Yalnızca $PTF_t$ ve Doğrusal Saat ($N_{\text{feat}}=2$)<br>$\hat{S} = -0,0806 \cdot PTF + 3,50 \cdot h + 77,75$ | **+108,51 TL**<br>([+49,04; +167,17]) | -10,63 TL<br>([-68,71; +41,21]) | %33,2 | %29,6 | Eğitim+Doğrulama OLS ($h \in [0, 23]$ doğrusal saat); 30 ek ML değişkeni olmadan $L_2$'nin %91,1'ini üretir ($p>0,60$). |
| **3. (a2) Tek Değişkenli Doğrusal Regresyon (PTF)**| Yalnızca $PTF_t$ ($N_{\text{feat}}=1$)<br>$\hat{S} = -0,0755 \cdot PTF + 105,92$ | **+94,77 TL**<br>([+36,05; +153,55]) | -24,36 TL<br>([-87,65; +29,16]) | %33,0 | %28,2 | Eğitim+Doğrulama OLS; başabaş $PTF \approx 1.403$ TL; tek bir doğrusal katsayı dahi $L_2$ ile istatistiki olarak farksızdır. |
| **4. (b) Naif Medyan Kuralı (Eğitim Eşiği)** | $PTF_t < 2.500$ TL ise Uzun, $\ge 2.500$ TL ise Kısa | **+140,57 TL**<br>([+87,74; +205,77]) | **+21,43 TL**<br>([-14,85; +62,71]) | %35,3 | %48,5 | Eşik = Eğitim+Doğrulama PTF medyanı (2.500 TL). Nokta tahmini üstündür; GA sıfırı içerir. |
| **5. (c) PTF Özellikleri Çıkarılmış $L_2$ Modeli**| PTF hariç 26 öznitelikle eğitilen LightGBM | **+70,73 TL**<br>([+15,98; +132,41]) | **-48,40 TL**<br>([-84,83; -13,95]) | %43,9 | %39,6 | PTF çıkarıldığında $L_2$'ye göre anlamlı kayıp ($p<0,01$); getiri naif seviyede kalır. |

> [!NOTE]
> **Eşik ve Katsayıların Örneklem Kaynağı, Eşik Izgarası Duyarlılığı ve Güven Aralıkları:**  
> 1. **Parametrelerin Örneklem Kaynağı:**  
>    - **(a) ve (a2) Doğrusal Katsayıları:** Test öncesi dönemdeki (TrVal: 2024 – 2025 ilk 9 ayı, $N = 15.168$ saat; Eğitim: 8.616 saat, Doğrulama: 6.552 saat) veriler üzerinde OLS ile uydurulmuştur. 2023 verisi veri ambarında bulunmamakta olup test öncesi toplam veri kütüğü 15.168 saattir. (a2) tek değişkenli kuralı $\hat{S} = -0,0755 \cdot PTF + 105,92$ olup başabaş seviyesi $105,92 / 0,0755 \approx 1.403$ TL'dir (Uzun payı %28,2, test P&L: +94,77 TL). (a) satırı ise günün saatini $h \in [0, 23]$ olarak doğrusal biçimde modele katar ($+3,50 \cdot h$; kukla saat vektörü değildir).  
>    - **(b) 2.500 TL Eşiği:** Yine test öncesi dönemdeki (TrVal, $N=15.168$) fiili PTF medyan değeridir ($Medyan_{\text{TrVal}}(PTF) = 2.500,00$ TL; test yılı medyanı ise $2.567,00$ TL'dir).  
> 2. **Eşik Izgarası Duyarlılık Analizi (1.500 – 3.500 TL):**  
>    Sabit PTF eşiği test kümesinde sistematik olarak tarandığında getiri profili şu şekildedir:  
>    - $PTF < 1.500$ TL: +105,09 TL/MWh (Uzun: %30,9, Doğruluk: %33,2) | Fark vs $L_2$: -14,04 TL [%95 GA: -71,81; +37,21]  
>    - $PTF < 2.000$ TL: +127,14 TL/MWh (Uzun: %40,1, Doğruluk: %33,8) | Fark vs $L_2$: +8,01 TL [%95 GA: -34,74; +47,84]  
>    - $PTF < 2.250$ TL: +138,10 TL/MWh (Uzun: %43,8, Doğruluk: %34,5) | Fark vs $L_2$: +18,96 TL [%95 GA: -16,43; +59,85]  
>    - **$PTF < 2.500$ TL:** **+140,57 TL/MWh** (Uzun: %48,5, Doğruluk: %35,3) | Fark vs $L_2$: **+21,43 TL** [%95 GA: -14,85; +62,71]  
>    - $PTF < 2.750$ TL: +135,48 TL/MWh (Uzun: %54,9, Doğruluk: %36,2) | Fark vs $L_2$: +16,35 TL [%95 GA: -21,57; +61,93]  
>    - $PTF < 3.000$ TL: +108,77 TL/MWh (Uzun: %64,2, Doğruluk: %37,9) | Fark vs $L_2$: -10,37 TL [%95 GA: -57,11; +42,70]  
>    - $PTF < 3.500$ TL: +74,96 TL/MWh (Uzun: %91,6, Doğruluk: %51,2) | Fark vs $L_2$: -44,18 TL [%95 GA: -120,30; +26,89]  
>    Görüldüğü üzere 2.000–2.750 TL aralığındaki tüm eşikler +127 ile +141 TL bandında stabil ve yüksek getiri üretmektedir; 2.500 TL eşiğinin $L_2$'ye göre +21,43 TL'lik farkı ise 7 günlük blok bootstrap %95 güven aralığında **[-14,85; +62,71]** olup sıfırı rahatlıkla içermektedir ($p > 0,25$). Dolayısıyla *"deterministik kural LightGBM'i geride bırakmaktadır"* tespiti yalnızca bir **nokta tahmini üstünlüğüdür**; ekonometrik olarak iki strateji istatistiksel açıdan farksızdır.  
> 3. **(c) PTF Çıkarılmış Model Mekanizması:**  
>    PTF öznitelikleri modelden çıkarıldığında yön doğruluğu %43,9, Uzun pozisyon oranı ise **%39,6'dır (3.472 saat)**. Model naif kural gibi %100 Uzun kalmamakta; bilakis saatlerin %60,4'ünde (5.288 saat) Kısa pozisyon almaktadır. Ancak aldığı 3.472 Uzun saatinde saat başına ortalama +182,23 TL (toplam katkı: +72,23 TL) kazanırken, 5.288 Kısa saatinde saat başına ortalama -2,47 TL (toplam katkı: -1,49 TL) kaybetmiş ve net getiri tesadüfi bir dengeyle **+70,73 TL/MWh** seviyesinde kalmıştır. Bu durum, modelin naif kurala çöktüğünü değil; PTF öznitelikleri olmaksızın modelin naif kuralın (+71,73 TL) ötesinde hiçbir ekonomik alfa üretemediğini ve $L_2$'ye göre istatistiki olarak anlamlı düzeyde kayba uğradığını (fark: -48,40 TL [%95 GA: -84,83; -13,95], $p < 0,01$) gösterir.

*Ekonometrik Değerlendirme ve Çıkarımlar:*  
Tablo 4C bulguları, Aşama 2'nin ekonomik getirisinin karmaşık bir makine öğrenmesi örüntüsünden türemediğini doğrulamaktadır:
1. **Yapısal Sınırlar ve Ortalamaya Dönüş Dinamiği:** Fiyatların sıfır tabanı ile tavan limitleri arasına sıkışması, yayılım üzerinde güçlü bir seviye bağımlılığı yaratmaktadır. $PTF$ düşük olduğunda (gece/yenilenebilir saatleri), fiyat ancak yukarı sıçrayabilir veya sıfırda kalabilir; kural Uzun pozisyona geçerek pozitif sıçramaları toplar.  
   - **Tavan Saatlerinin Gerçek Yapısı ve Marjın Belirleyiciliği:** Tablo 4B'nin Kategori 2'sinde yer alan 2.113 tavan saati incelendiğinde, bu saatlerin 817'sinde (%38,7) hem $PTF$ hem $SMF$ tavan seviyesindedir ve yayılım tam sıfırdır ($PTF=SMF=\text{tavan}$). Toplam 1.852 sıfır yayılım saatinin 817'si tavan seviyesinde gerçekleşmiştir. Kalan saatlerin 1.036'sında (%49,0) $SMF > PTF$ iken, 260'ında (%12,3) $SMF < PTF$'dir.  
   - **Naif Kuralın +227,86 TL Kazancı Nereden Gelir?** "Her Zaman Açık" naif kuralı tavan saatlerinde devasa bir yayılım sıçramasından değil; yayılımın sıfır olduğu saatlerde dahi sistemin Enerji Açığında olması nedeniyle uzlaştırma formülünün ($P_{\text{neg}} = \text{tavan} \times 1,06$) sağladığı **$k$ dengesizlik ceza marjını (%6 $\times$ tavan $\approx$ 204–270 TL)** ve ampirik platform anomali çarpanlarını tahsil etmektedir. $L_2$ regresörü ise yüksek PTF seviyesini ortalamaya dönüş kuralıyla Kısa sinyali olarak okuduğundan bu marj cezalarına ters yönde maruz kalmış ve tavan kategorisinde **−78,25 TL/MWh** zarar üretmiştir.
2. **Algoritmik Beceri Değil, PTF Seviye Etkisi:** Eğitim kümesi medyanına dayanan deterministik bir kural ($PTF < 2.500$ TL ise Uzun) **+140,57 TL/MWh** nokta getiri sağlamakta ve LightGBM modelini (+119,14 TL) nokta tahmininde 21,43 TL aşmaktadır (blok bootstrap farkı [-14,85; +62,71] ile istatistiki başa başlık). PTF öznitelikleri çıkarıldığında ise getiri **+70,73 TL/MWh** seviyesine gerilemekte ve modelsiz "Her Zaman Açık" naif kuralının (+71,73 TL) düzeyini aşamamaktadır. Dolayısıyla Aşama 2'nin başarısı bir yön veya sıçrama tahmin becerisi değil; piyasa sınırlarının yarattığı yapısal ortalamaya dönüş dinamiğinin mekanik bir sonucudur.

---

#### Tablo 4D: P&L Bileşen Dekompozisyonu: Yayılım, Marj + Etiket-İşaret Uyumsuzluğu ve Platform Veri Kalıntısı (N = 8.760 Saat)

Ticari dengesizlik kararlarının getiri formülü ekonometrik olarak üç ayrık bileşene ayrıştırılabilir:
$$\text{Net P\&L}_t = \underbrace{a_t \cdot (SMF_t - PTF_t)}_{\text{1. Saf Yayılım (Spread)}} + \underbrace{\text{Marj}_t(k, l) + \text{Uyumsuzluk}_t}_{\text{2. Ceza Marjı + Etiket/İşaret Uyumsuzluğu}} + \underbrace{\text{Anomali}_t}_{\text{3. Platform Veri Kalıntısı}}$$

| Karar Stratejisi | Fiili Yıllık P&L | 1. Yayılım Bileşeni ($a_t \cdot S_t$) | 2. Marj + Etiket Uyumsuzluğu | 3. Platform Veri Kalıntısı | Strateji Mekanizması ve Temel Çıkarım |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Her Zaman Açık (Naif Kural)** | **+71,73 TL** | **+11,86 TL (%16,5)** | **+95,69 TL (%133,4)** | **-35,82 TL (-49,9%)** | Getirinin ana gövdesi yayılımdan değil; açık yönlü $k=0,06$ asimetrik ceza marjından gelir. |
| **Aşama 1 Kalibre Modeli** | **+67,33 TL** | **+24,79 TL (%36,8)** | **+47,73 TL (%70,9)** | **-5,19 TL (-7,7%)** | Pasif kalarak maruziyeti yarıya keser; yayılım payı artar, marj hasılatı yarı yarıya düşer. |
| **Aşama 2 $L_2$ Regresör Modeli** | **+119,14 TL** | **+150,50 TL (%126,3)** | **-31,64 TL (-26,6%)** | **+0,27 TL (+0,2%)** | **Yalnızca yayılımdan kazanır.** Pozisyonunu yayılım işaretine göre seçerek spread'i toplar, ceza marjında eksiye düşer. |
| **Deterministik $PTF < 2.500$ TL** | **+140,57 TL** | **+188,68 TL (%134,2)** | **-37,16 TL (-26,4%)** | **-10,96 TL (-7,8%)** | Düşük PTF'de Uzun, yüksekte Kısa kalarak yapısal ortalamaya dönüş yayılımını tam olarak hasat eder. |
| **Nötr Rüzgar Portföyü ($\alpha=0,50$)**| **+29,62 TL** | **+5,93 TL (%20,0)** | **+47,57 TL (%160,6)** | **-23,88 TL (-80,6%)** | Tablo 11 sentetik portföy getirisi; yayılım katkısı yalnızca +5,93 TL olup, marj yapısı, etiket uyumsuzlukları ve anomali terimleri belirleyicidir. |

> [!IMPORTANT]
> **P&L Mekanizmasının Ekonometrik Ayrımı:**  
> Tablo 4D, elektrik piyasasında *"P&L neyi ölçüyor?"* sorusunun ekonometrik dinamiklerini ortaya koyar:  
> 1. **İşaret Seçen Modeller ($L_2$ Regresörü, PTF Eşik Kuralı):** Gerçekten **yayılımın (spread) büyüklüğüne ve işaretine** bağlı olarak çalışır. $L_2$ regresörünün ürettiği +119,14 TL'nin +150,50 TL'si doğrudan yayılım bileşeninden gelir.  
> 2. **Naif Referans Kuralları ("Her Zaman Açık"):** Yayılıma değil, **yasal dengesizlik ceza katsayısı ($k=0,06$) asimetrisi ile yön ve yayılım uyumsuzluklarına** bağlıdır. Ortalama yayılım (+11,86 TL) naif getirinin yalnızca %16,5'ini açıklarken, yasal marj ve uyumsuzluk terimleri (+95,69 TL) getirinin asıl belirleyicisidir. Bu nedenle naif kural piyasa yönünü tahmin etmez; piyasanın ceza yapısını süzer.  
> 3. **Sentetik Nötr Portföy ($\alpha = 0,50$):** Portföyde ortalama yayılım (+11,86 TL) yalnızca $0,5 \times 11,86 \approx 5,93$ TL katkı üretmekte; getirinin kalan bölümü (+23,69 TL) ceza marjı yapısı, yön-yayılım uyumsuzlukları ve ampirik platform anomalilerinden kaynaklanmaktadır.

---

## 2. Gün İçi Saat Dilimi (Diurnal Blok) Analizi (12–36 Saat Gecikmeli Gün Öncesi Kısıtı)

Gün öncesi tahmin kurgusunda tahminler günde tek bir kez (saat 11:30'da) üretilmekte ve ertesi günün 24 saati (00:00–23:00) topluca hedeflenmektedir. Bu nedenle tahmin ufku (lead time: 12.5 saatten 36.5 saate kadar), günün saat dilimleri (gün içi tüketim ve aydınlatma profili / diurnal cycle) ile **tam doğrusal bağımlıdır (mükemmel eşdoğrusallık).**

Hedef günün 24 saati, karar anlarına (GÖP Öncesi 11:30 vs GÖP Sonrası 14:00) ve çeyrek bloklarına göre ayrıştırılmıştır:

#### Tablo 5: Gün İçi Saat Dilimlerine Göre Başarım Metrikleri ve Eylem Kırılımı (12–36 Saat Gecikmeli Gün Öncesi Kurgusu)

##### Panel A: GÖP Öncesi Karar Kurgusu (Saat 11:30 - PTF Bilinmiyor)
| Saat Dilimi ve Blok | Saat ($N$) | Çoğunluk Sınıfı Payı (Klimatoloji) | Aşama 1 Yön Doğruluğu | Aşama 1 Eylem Dağılımı (% Uzun / % Kısa / % Pasif) | Aşama 1 P&L (TL/MWh) | Her Zaman Açık P&L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1 (00:00–05:00 / Gece, 12–18h)** | 2.190 | %49,18 (Açık: 1.077, Fazla: 1.005) | **%46,21** (-2,97 puan) | **%12,51 Uzun, %9,63 Kısa, %77,85 Pasif** | +11,30 TL | -11,78 TL |
| **Q2 (06:00–11:00 / Sabah, 19–24h)** | 2.190 | %53,15 (Açık: 1.164, Fazla: 1.006) | **%65,02** (+11,87 puan) | %29,27 Uzun, %17,03 Kısa, %53,70 Pasif | +88,32 TL | +60,00 TL |
| **Q3 (12:00–17:00 / Gündüz, 25–30h)** | 2.190 | %53,24 (Açık: 1.166, Fazla: 1.019) | **%61,00** (+7,76 puan) | %42,19 Uzun, %14,29 Kısa, %43,52 Pasif | +83,71 TL | +117,00 TL |
| **Q4 (18:00–23:00 / Akşam Zirve, 31–36h)** | 2.190 | %67,40 (Açık: 1.476, Fazla: 688) | **%66,39** (-1,01 puan) | %69,27 Uzun, %1,55 Kısa, %29,18 Pasif | +85,97 TL | +121,72 TL |
| **Genel Ortalama / Toplam** | **8.760** | **%55,74** | **%59,66** (+3,92 puan) | **%38,31 Uzun, %10,63 Kısa, %51,06 Pasif** | **+67,33 TL** | **+71,73 TL** |

##### Panel B: GÖP Sonrası Karar Kurgusu (Saat 14:00 - PTF Deterministik Olarak Biliniyor)
| Saat Dilimi ve Blok | Naif $SMF=PTF$ MAE | Naif $SMF=PTF$ RMSE | Aşama 2 Yayılım MAE | Aşama 2 Yayılım RMSE | Aşama 2 RMSE Kazancı | Aşama 2 P&L (TL/MWh) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1 (00:00–05:00 / Gece)** | 726,89 TL | 1.000,55 TL | 788,62 TL | 1.009,01 TL | -8,46 TL | +79,62 TL |
| **Q2 (06:00–11:00 / Sabah)** | 611,80 TL | 935,36 TL | 701,89 TL | 936,20 TL | -0,84 TL | +97,13 TL |
| **Q3 (12:00–17:00 / Gündüz)** | 705,67 TL | 1.049,73 TL | 778,53 TL | 1.029,98 TL | +19,75 TL | +164,30 TL |
| **Q4 (18:00–23:00 / Akşam)** | 686,63 TL | 1.086,04 TL | 723,53 TL | 1.049,21 TL | **+36,83 TL** | +135,49 TL |
| **Genel Ortalama / Toplam** | **682,75 TL** | **1.019,49 TL** | **748,14 TL** | **1.007,01 TL** | **+12,48 TL** | **+119,14 TL** |

*Ekonometrik Değerlendirme:*
1. **Yön Doğruluğu Sağlaması:** Dört eşit çeyreğin doğruluğu $\frac{46,21 + 65,02 + 61,00 + 66,39}{4} = \mathbf{\%59,66}$ olup Tablo 9'daki genel 3-sınıflı test doğruluğu (0,5966) ile birebir örtüşmektedir.
2. **Akşam Zirvesinin Doğru Okunması:** Akşam saatlerinde yön doğruluğunun yüksek (%66,39) çıkması modelin uzak ufukta daha başarılı olmasından değil; akşam saatlerinde çoğunluk sınıfının (Enerji Açığı) payının %67,40'a fırlamasından kaynaklanmaktadır. Model bu saatlerde çoğunluk sınıfı referansının (%67,40) 1,01 puan gerisinde kalmaktadır.
3. **Model Katkısının Sabah ve Gündüz Saatlerinde Yoğunlaşması:** Sistemin yön açısından dengeli dağıldığı sabah ve gündüz saatlerinde (Q2 ve Q3), çoğunluk sınıfı %53 civarındayken model %61–%65 seviyesine ulaşarak sırasıyla **+11,87** ve **+7,76** puanlık yönsel öngörü üretmektedir.
4. **Gece Pasifliği Koruması:** Gece saatlerinde tüketim dip yaptığından sabit açık pozisyon almak negatif getiri üretmektedir (-11,78 TL). Aşama 1 modeli bu saatlerde %46,21 doğrulukla çoğunluğun (%49,18) altında kalmış; ancak **saatlerin %77,85'inde pasif kalarak** kaybı sınırlamış ve +11,30 TL getiri sağlamıştır.

---

## 3. EPDK Mevzuatı, Karar 14030 Katsayı Matrisi ve Uzlaştırma Kalıntıları

### 3.1. EPDK Karar 14030 Katsayı Yapısı ve Rejim Dönemleri
Test kümesindeki 8.760 saat, EPDK mevzuat kararları ve tavan fiyat dinamiklerine göre takvimsel olarak 3 kesin döneme ayrılmıştır:

#### Tablo 6: EPDK Karar 14030 Çift Yönlü Dengesizlik Ceza Katsayıları Matrisi (Karar Tarihi: 11.12.2025)

| Sistem Denge Durumu ($D_t$) | Negatif Dengesizlik Formülü ($P_{\text{neg}, t}$) | Pozitif Dengesizlik Formülü ($P_{\text{pos}, t}$) | Uygulanan Ceza Katsayıları ($k / l$) |
| :--- | :--- | :--- | :---: |
| **Enerji Açığı ($D_t = 1$, Net Talimat $>0$)** | $\max(PTF_t, SMF_t) \times (1 + k)$ | $\min(PTF_t, SMF_t) \times (1 - l)$ | $k = 0,06$ (%6 ceza) / $l = 0,03$ (%3 indirim) |
| **Enerji Fazlası ($D_t = -1$, Net Talimat $<0$)** | $\max(PTF_t, SMF_t) \times (1 + k)$ | $\min(PTF_t, SMF_t) \times (1 - l)$ | $k = 0,03$ (%3 ceza) / $l = 0,06$ (%6 indirim) |
| **Dengede ($D_t = 0$, Net Talimat $=0$)** | $\max(PTF_t, SMF_t) \times (1 + k)$ | $\min(PTF_t, SMF_t) \times (1 - l)$ | $k = 0,03$ (%3) / $l = 0,03$ (%3) |

*Mevzuat ve Tavan Fiyat Kararları Dipnotu:*  
- EPDK'nın 11.12.2025 tarih ve 14030 sayılı Kurul Kararı ile 01.01.2026'dan itibaren asimetrik $k/l$ katsayıları yürürlüğe girmiştir.  
- Azami uzlaştırma fiyat tavanları EPDK Kurul Kararları ile belirlenmiştir:  
  * **3.400 TL/MWh Tavanı:** EPDK'nın 27.03.2025 tarih ve 13423 sayılı Kurul Kararı (Resmî Gazete: 05.04.2025 tarih ve 32861 sayılı; yürürlük tarihi: 05.04.2025).  
  * **4.500 TL/MWh Tavanı:** EPDK'nın 02.04.2026 tarih ve 14459 sayılı Kurul Kararı (yürürlük tarihi: 04.04.2026).  
- **Rejim 1 ($N = 2.208$ Saat / 92 Gün):** 01.10.2025 00:00 – 31.12.2025 23:00. Tavan 3.400 TL/MWh, simetrik %3 katsayı.  
- **Rejim 2 ($N = 2.232$ Saat / 93 Gün):** 01.01.2026 00:00 – 03.04.2026 23:00. Karar 14030 katsayıları, tavan 3.400 TL/MWh.  
- **Rejim 3 ($N = 4.320$ Saat / 180 Gün):** 04.04.2026 00:00 – 30.09.2026 23:00. Karar 14030 katsayıları, tavan 4.500 TL/MWh.  
  *(Toplam: $2.208 + 2.232 + 4.320 = 8.760$ saat).*

---

### 3.2. 2026 Yılı Fiyat Uyuşmazlıkları ve -106 / -103 TL Platform Veri Kalitesi Alarmı

Test kümesindeki 8.760 saatte teorik mevzuat formülleri ile EPİAŞ'ın yayımladığı fiili uzlaştırma serileri (`negativeImbalance`, `positiveImbalance`) karşılaştırılmıştır:

1. **Negatif Dengesizlik Uyuşmazlıkları (Toplam: 983 Saat):**
   - 552 saatte fiyat tavan seviyesine ulaştığında fiili uzlaştırma serisinde mevzuat formülünün üzerinde çarpanlar gözlenmiştir.
   - 409 saatte sıfır taban fiyat uygulamaları gerçekleşmiştir.
   - 22 saatte ise açıklanamayan veri düzensizlikleri kaydedilmiştir.
2. **Pozitif Dengesizlik Serisindeki Negatif Fiyatlar (Toplam: 1.538 Saat):**
   - 2026 yılındaki 6.552 saatin **1.538'inde (%23,47)** EPİAŞ fiili pozitif dengesizlik serisi negatif değerler almıştır: **-106,00 TL/MWh** (1.468 saat) ve **-103,00 TL/MWh** (70 saat).
   - **Rejim Ayrımı:** Söz konusu negatif değerler Rejim 1'de (2025 son çeyreği, 2.208 saat) **hiç görülmemiştir (0 / 2.208 saat, %0,0)**; tamamen 2026 yılına (Rejim 2'de 318 saat, Rejim 3'te 1.220 saat) aittir.
   - **Mevzuat Rekonstrüksiyonu ve Formülsel Uyuşmazlık:** Tablo 6'daki yasal DUY formülü rekonstrüksiyonumuza göre Enerji Açığı saatlerinde $P_{\text{pos}, t} = \min(PTF_t, SMF_t) \times 0,97$ olmalıdır. $PTF_t \ge 0$ ve $SMF_t \ge 0$ piyasa taban kuralları gereğince bu formül kuramsal olarak daima $P_{\text{pos}, t} \ge 0$ üretir. Fiili serideki negatif değerler bu rekonstrüksiyonla uyuşmamaktadır.
   - **Formülsel Kaynak / Sentinel ve Tarife Olasılıkları:** Değerlerin sistem dengesizlik yönüne göre %100 ayrışması (Fazla saatlerinde $-106$, Açık saatlerinde $-103$), Karar 14030'daki ceza katsayılarıyla ($l=0,06 \rightarrow -100 \times (1 + 0,06) = -106$; $l=0,03 \rightarrow -100 \times (1 + 0,03) = -103$) tam bir matematiksel uyum sergilemektedir. Bu formülsel yapı iki muhtemel kaynağa işaret etmektedir:
     1. *Sistem Raporlama Yer Tutucusu (Sentinel Flag):* EPİAŞ veri ambarının taban fiyat veya istisnai saatlerde formül sonucu yerine ceza katsayısını kodlayan formülsel bir yer tutucu hata kodu iletmesi;
     2. *2026 Yılı İdari Kuralı / Tarife Uygulaması:* EPİAŞ'ın 2026'da DUY uzlaştırmasında kamuoyuna duyurulmamış veya tezin standart yasal rekonstrüksiyonundan farklılaşan yeni bir tarife ya da taban fiyat formülü uygulamış olması.  
   Doğrulanan husus $k$ ve $l$ marjlarının varlığıdır; 2026'daki iç hesaplama motorunun kesin hukuki ve teknik niteliğini netleştirmek üzere EPİAŞ piyasa destek birimine veya 2026 uzlaştırma metodolojisi belgelerine başvurulması tavsiye edilmektedir. Bu nedenle tezde her iki fiyatlandırma yaklaşımı eşit ağırlıkla analiz edilmiştir.

#### Tablo 7: Rejim × Sistem Yönü × Fiyat Değeri Çapraz Tablosu (N = 8.760 Saat)

| Piyasa Rejimi | Sistem Denge Durumu ($D_t$) | $P_{\text{pos}} = -103,00$ TL (Açık Anomali) | $P_{\text{pos}} = -106,00$ TL (Fazla Anomali) | Normal Fiyat ($P_{\text{pos}} \ge 0$) | Rejim / Durum Toplamı |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Rejim 1** (2025 Son Çeyrek) | **Dengede ($D_t = 0$)** | 0 | 0 | 70 | 70 saat |
| | **Enerji Açığı ($D_t = 1$)** | 0 | 0 | 1.525 | 1.525 saat |
| | **Enerji Fazlası ($D_t = -1$)** | 0 | 0 | 613 | 613 saat |
| *Rejim 1 Toplamı* | *(Anomali Yok)* | **0** | **0** | **2.208** | **2.208 saat** |
| **Rejim 2** (2026 İlk Çeyrek) | **Dengede ($D_t = 0$)** | 0 | 0 | 36 | 36 saat |
| | **Enerji Açığı ($D_t = 1$)** | **15** | 0 | 1.172 | 1.187 saat |
| | **Enerji Fazlası ($D_t = -1$)** | 0 | **303** | 706 | 1.009 saat |
| *Rejim 2 Toplamı* | *(318 Saat Anomali)* | **15** | **303** | **1.914** | **2.232 saat** |
| **Rejim 3** (2026 Yaz Dönemi) | **Dengede ($D_t = 0$)** | 0 | 0 | 53 | 53 saat |
| | **Enerji Açığı ($D_t = 1$)** | **55** | 0 | 2.116 | 2.171 saat |
| | **Enerji Fazlası ($D_t = -1$)** | 0 | **1.165** | 931 | 2.096 saat |
| *Rejim 3 Toplamı* | *(1.220 Saat Anomali)*| **55** | **1.165** | **3.100** | **4.320 saat** |
| **GENEL TOPLAM (N = 8.760)** | — | **70 saat (%0,8)** | **1.468 saat (%16,8)** | **7.222 saat (%82,4)** | **8.760 saat (%100,0)** |

*Yerel Dosya ve EPİAŞ API Akışı Tutarlılık Denetimi ve Örnek Saatler:*  
Yerel veri setindeki kayıtların API veri akışıyla tutarlılığını denetlemek üzere EPİAŞ Şeffaflık Platformu Resmi API servisleri üzerinden farklı aylardan münferit saatler sorgulanmış ve yerel parquet dosyasındaki değerlerin API'den çekilen ham akışla birebir tutarlı olduğu görülmüştür:
- **$-103,00$ TL (Enerji Açığı) Örnekleri:**  
  * 15 Şubat 2026, 08:00 $\rightarrow$ PTF: 50,01 TL | SMF: 1.450,00 TL | $P_{\text{pos}}$: **-103,00 TL** | Durum: Enerji Açığı
  * 07 Mart 2026, 12:00 $\rightarrow$ PTF: 100,00 TL | SMF: 3.000,00 TL | $P_{\text{pos}}$: **-103,00 TL** | Durum: Enerji Açığı
  * 16 Nisan 2026, 12:00 $\rightarrow$ PTF: 0,00 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-103,00 TL** | Durum: Enerji Açığı
- **$-106,00$ TL (Enerji Fazlası) Örnekleri:**  
  * 04 Ocak 2026, 14:00 $\rightarrow$ PTF: 1.400,00 TL | SMF: 100,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Şubat 2026, 09:00 $\rightarrow$ PTF: 1.547,40 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Nisan 2026, 00:00 $\rightarrow$ PTF: 1.499,99 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Nisan 2026, 01:00 $\rightarrow$ PTF: 1.250,00 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Nisan 2026, 02:00 $\rightarrow$ PTF: 950,00 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Nisan 2026, 03:00 $\rightarrow$ PTF: 750,00 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Nisan 2026, 04:00 $\rightarrow$ PTF: 500,00 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası
  * 01 Nisan 2026, 05:00 $\rightarrow$ PTF: 600,00 TL | SMF: 0,00 TL | $P_{\text{pos}}$: **-106,00 TL** | Durum: Enerji Fazlası

Bu denetim, yerel veri dosyası ile EPİAŞ API veri akışı arasında tam bir tutarlılık bulunduğunu; ayrışmanın yerel bir aktarım veya saklama hatasından değil, doğrudan API'nin sağladığı ham veri serisinden kaynaklandığını göstermektedir. Ancak bu kontrol aynı veri akışına yönelik bir iç tutarlılık doğrulaması olup bağımsız bir web arayüzü doğrulaması niteliği taşımamaktadır. Bu nedenle değerler tezde ihtiyatla karşılanmış ve hem fiili seri hem yasal formül paralel olarak değerlendirilmiştir.

#### Tablo 8: Fiyatlandırma Varsayımlarına Göre P&L Duyarlılık Analizi (TL/MWh)

| Simülasyon Senaryosu | Her Zaman Açık (Naif Kural) | Aşama 1 Platt Kalibre | Aşama 2 Yayılım (Teorik) | Model vs Açık Farkı | Fark %95 Blok Bootstrap GA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baz Senaryo: EPİAŞ Fiili Yayımlanan Fiyatlar** | **71,73 TL** | **67,33 TL** | **119,14 TL** | **-4,41 TL** | [-59,28; +44,81] ($p > 0,75$) |
| **Duyarlılık A: Yalnızca Pozitif Fiyat Formülü ($P_{\text{pos}}^{\text{stat}}$)** | **90,01 TL** | **64,80 TL** | **126,18 TL** | **-25,21 TL** | [-84,59; +26,50] ($p > 0,35$) |
| **Duyarlılık B: Tam Yasal Formül ($P_{\text{neg}}^{\text{stat}}$ ve $P_{\text{pos}}^{\text{stat}}$)** | **80,33 TL** | **60,35 TL** | **133,56 TL** | **-19,98 TL** | [-79,27; +32,57] ($p > 0,40$) |

*Ekonometrik Değerlendirme:* Ekonometrik tarafsızlık gereğince iki senaryo eşit ağırlıkla sunulmuştur: Baz Senaryo (EPİAŞ'ın fiilen yayımladığı ve piyasa katılımcılarının fiilen maruz kaldığı nakit akışını yansıtan ham fiyatlar) ve Duyarlılık B Senaryosu (Tezin Karar 14030 metninden türettiği kuramsal yasal formül). İster platformun ham değerleri baz alınsın (Aşama 1: 67,33 vs Açık: 71,73 TL, fark: -4,41 TL), ister tam yasal formül uygulansın (Aşama 1: 60,35 vs Açık: 80,33 TL, fark: -19,98 TL), Aşama 1 yön modelinin naif çoğunluk kuralına karşı istatistiksel olarak anlamlı bir ekonomik üstünlük sağlayamadığı ve tezin temel bilimsel çıkarımlarının her iki senaryoda da niteliksel ve yönsel olarak korunduğu görülmektedir.

> [!IMPORTANT]
> **Duyarlılık A'da Aşama 2 Getirisinin Artış Mekanizması ve Payda Tanımları:**  
> Duyarlılık A senaryosunda Aşama 2 getirisinin 119,14 TL'den 126,18 TL'ye yükselmesi model başarısı değildir:  
> - **Enerji Fazlası Saatlerindeki Hatalı Uzun Pozisyonlar:** Fiili seride $P_{\text{pos}} = -106$ TL gerçekleşen 1.468 Enerji Fazlası saatinde, Aşama 2 modeli sistem yönünün tersine düşerek saatlerin **%69,01'inde (1.013 / 1.468 saat)** hatalı olarak Uzun (+1) pozisyondadır ($-103$ TL seviyesindeki 70 saat Enerji Açığı olduğundan bu saatlerde Uzun pozisyon yönsel bir hata oluşturmamaktadır).  
> - Fiili seride bu hatalı Uzun pozisyonlar -106 TL ceza alırken, kuramsal formüle ($P_{\text{pos}}^{\text{stat}} \ge 0$) geçildiğinde ceza ortadan kalkmakta ve modelin hatalı pozisyonları yapay biçimde kurtulmaktadır.  
> - Buna karşılık Aşama 1 modeli aynı 1.538 saatlik kümede **274 saat (%17,82) Uzun**, **512 saat (%33,29) Kısa** ve **752 saat (%48,89) Pasif** kalarak hatalı maruziyetten büyük ölçüde kaçınmıştır.

---

## 4. Sistem Denge Yönü Sınıflandırma ve Kalibrasyon Başarımı

#### Tablo 9: 3-Sınıflı Sistem Denge Yönü Sınıflandırma Başarımı (N = 8.760 Saat / 365 Gün)

| Model / Referans Kural | Doğruluk | Dengeli Doğruluk | MCC | Macro-F1 | Weighted-F1 | Brier Skoru | BSS (Eğitim Klim.) | BSS (Test Empirik) | Ağırlıklı ECE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sabit Uzun (Deterministik Açık)** | 0,5574 | 0,3333 | 0,0000 | 0,2386 | 0,3990 | 0,8852 | -68,71% | -73,98% | — |
| **Eğitim Klimatolojisi Prior Tahmini** | 0,5574 | 0,3333 | 0,0000 | 0,2386 | 0,3990 | 0,5247 | %0,00 (Ref) | -3,13% | %8,61 |
| **Lag-168 Naif Takipçi (1 Hafta Önce)** | 0,5742 | 0,4128 | 0,1624 | 0,4129 | 0,5740 | — | — | — | — |
| **Aşama 1: GÖP Öncesi (Ham Olasılık)** | 0,5966 | 0,4653 | 0,2110 | 0,4434 | 0,5963 | 0,4959 | +%5,49 | +%2,53 | %7,58 |
| **Aşama 1: GÖP Öncesi (Platt Kalibre)** | **0,5966** | **0,4653** | **0,2110** | **0,4434** | **0,5963** | **0,4761** | **+%9,26** | **+%6,43** | **%4,95** |
| **Aşama 2: GÖP Sonrası Model (Ham Olasılık)** | 0,6221 | 0,4883 | 0,2721 | 0,4652 | 0,6275 | 0,4763 | +%9,22 | +%6,38 | %8,67 |
| **Aşama 2: GÖP Sonrası Model (Platt Kalibre)** | **0,6221** | **0,4883** | **0,2721** | **0,4652** | **0,6275** | **0,4460** | **+%14,99** | **+%12,33** | **%4,21** |

*İstatistiki ve Metodolojik Değerlendirme:*
1. **Brier Skoru Tanımı:** Brier skoru tek başına güvenilirliği (kalibrasyonu) değil; **güvenilirlik ve çözünürlük bileşenlerinin toplamını ölçen genel bir olasılık skorudur.** Deterministik sabit Uzun tahmininin Brier skoru **0,8852**'dir. Tablodaki 0,5247 değeri eğitim kümesinin prior olasılık vektörüne ($[0,329; 0,032; 0,639]$) aittir.
2. **Kalibrasyon Metodolojisi Sınırlılığı:** Platt kalibratörü, yalnızca eğitim kümesiyle eğitilen modelin doğrulama kümesi çıktıları üzerinde uydurulmuş; ardından birleşik eğitim+doğrulama kümesiyle eğitilen nihai modelin test tahminlerine uygulanmıştır.
3. **Aşama 2 Kalibrasyonu ve Olasılık Kalitesi:** Aşama 2 modeli kalibre edildiğinde Brier skoru 0,4763'ten **0,4460**'a düşmekte, test BSS'si ise **+%12,33** seviyesine çıkmaktadır. Bu bulgu, adil bir kıyaslama yapıldığında PTF bilgisinin olasılık tahmin kalitesini de belirgin biçimde iyileştirdiğini doğrulamaktadır.

#### Tablo 10: Günlük Eşleştirilmiş Wilcoxon Testleri ve Genişletilmiş Holm-Bonferroni Düzeltmesi (N = 365 Gün)

| Karşılaştırma Hipotezi | Ortalama Günlük Doğruluk Farkı | Wilcoxon $W$ | Ham $p$-Değeri | Holm Sırası ($i$) | Holm Eşiği ($\alpha=0,05$) | Düzeltilmiş Anlamlılık |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aşama 2 vs Çoğunluk** | +%6,47 | 19.993,0 | 0,0002 | 1 | 0,0125 ($0,05/4$) | Anlamlı ($p < 0,05$) |
| **Aşama 2 vs Aşama 1** | +%2,56 | 16.166,5 | 0,0003 | 2 | 0,0167 ($0,05/3$) | Anlamlı ($p < 0,05$) |
| **Aşama 1 vs Çoğunluk** | +%3,92 | 21.868,5 | 0,0169 | 3 | 0,0250 ($0,05/2$) | Anlamlı ($p_{\text{adj}} \approx 0,034$) |
| **Lag-168 vs Çoğunluk** | +%1,68 | 24.110,0 | 0,1140 | 4 | 0,0500 ($0,05/1$) | Anlamsız ($p > 0,05$) |

---

## 5. Portföy Hizalanma Analizi ve $\alpha = 0,5$ Ceza Asimetrisi İspatı

### 5.1. 2x2 Portföy Hizalanma Matrisi
Piyasa katılımcısının doğal dengesizliğinin sistem yönüyle $\alpha \in [0, 1]$ olasılıkla hizalı olduğu durumda birim megavat başına getiri matrisi:

$$\begin{array}{|c|c|c|}
\hline
\textbf{Strateji Eylemi } (a_t) & D_t = 1 \text{ (Sistem Açıkta)} & D_t = -1 \text{ (Sistem Fazlada)} \\
\hline
a_t = +1 \text{ (Uzun)} & \alpha (P_{\text{neg}, t} - PTF_t) - (1-\alpha)(PTF_t - P_{\text{pos}, t}) & -\alpha (PTF_t - P_{\text{pos}, t}) + (1-\alpha)(P_{\text{neg}, t} - PTF_t) \\
\hline
a_t = -1 \text{ (Kısa)} & -\alpha (P_{\text{neg}, t} - PTF_t) + (1-\alpha)(PTF_t - P_{\text{pos}, t}) & \alpha (PTF_t - P_{\text{pos}, t}) - (1-\alpha)(P_{\text{neg}, t} - PTF_t) \\
\hline
a_t = 0 \text{ (Pasif)} & 0,00 & 0,00 \\
\hline
\end{array}$$

*Not:* DUY ve Karar 14030 gereğince sistemin dengede olduğu ($D_t=0$, 159 saat) durumlarda hem Uzun hem Kısa pozisyon alan katılımcılar $-PTF_t \times 0,03$ ceza kesintisine katlanmaktadır. Bu 159 saat 2x2 matriste gösterilmemekle birlikte yıllık portföy getirisi hesaplamalarına doğrudan dahil edilmiştir.

#### Tablo 11: Portföy Hizalanma Olasılığına ($\alpha$) Göre Yıllık Ortalama Getiri (TL/MWh)

| Hizalanma Oranı ($\alpha$) | Aşama 2 Yayılım (Teorik) | Aşama 1 Platt Kalibre | Her Zaman Açık (Naif) | Fark [Aşama 1 − Açık] |
| :---: | :---: | :---: | :---: | :---: |
| **%100 (Tam Hizalanma)** | **119,14** | **67,33** | **71,73** | **-4,41 TL** |
| **%90** | 108,73 | 58,77 | 63,31 | -4,53 TL |
| **%80** | 98,32 | 50,22 | 54,89 | -4,66 TL |
| **%70** | 87,91 | 41,67 | 46,47 | -4,79 TL |
| **%60** | 77,50 | 33,12 | 38,04 | -4,92 TL |
| **%52,66 ($\alpha_{\text{wind}}$ Rüzgar Portföyü)** | **69,85** | **26,84** | **31,86** | **-5,02 TL** |
| **%50 (Tam Nötr / Rastgele)** | **67,08** | **24,57** | **29,62** | **-5,06 TL** |

*Rüzgar Portföyü ($\alpha_{\text{wind}} = \%52,66$) Tanımı, Sapma-Yön Eşlemesi ve Güven Aralığı:*  
$\alpha_{\text{wind}}$ parametresi, Türkiye rüzgar santrali kurulu gücünün gün öncesi üretim tahmin sapmaları ($\Delta W_t = W_{\text{fiili}, t} - W_{\text{tahmin}, t}$) ile TEİAŞ sistem dengesizlik yönü ($D_t$) arasındaki ampirik eşleşme sıklığını temsil etmektedir:  
- **Sapma-Yön Eşleme Kuralı ve Sıfır/Dengede Saatleri:** Rüzgar üretiminin tahminden fazla olduğu ($\Delta W_t > 0$) ve sistemin Enerji Fazlasında olduğu ($D_t = -1$) saatler ile rüzgarın yetersiz kaldığı ($\Delta W_t < 0$) ve sistemin Enerji Açığında olduğu ($D_t = 1$) saatler "doğal hizalı" sayılmıştır. Rüzgar sapmasının tam sıfır olduğu ($\Delta W_t = 0$) veya sistemin dengede bulunduğu ($D_t = 0$, 159 saat) saatler **hizalanmamış (nötr) kabul edilmiş ve paydaya tam olarak dahil edilmiştir**.  
- **Nokta Tahmin ve Gün Bloklu Güven Aralığı:** Yıllık $N = 8.760$ saatlik test kümesinde hizalanma oranı $\hat{\alpha}_{\text{wind}} = \mathbf{\%52,66}$'dır. Wilson asimptotik aralığı ([%51,61; %53,71]) saatlerin bağımsız olduğunu varsaydığından otokorelasyonu yok sayıp yapay biçimde dar bir aralık üretmektedir. Rüzgar sapmalarının ve sistem yönünün gün içi kümelenmesi ve güçlü serisel bağımlılığı dikkate alınarak gün bloklu bootstrap (Daily Block Bootstrap, $n_{\text{eff}} \approx 365$ gün) uygulandığında, %95 güven aralığı **[%48,2; %57,1]** olarak elde edilmektedir. Bu aralık tam nötr (%50,0) eşiğini açıkça içermektedir ($p > 0,20$). 159 Dengede saati dışlandığında dahi ($N = 8.601, \hat{\alpha} = \%53,63$) gün bloklu %95 GA **[%49,1; %58,2]** ile yine %50 eşiğini kapsamakta; rüzgar sapmalarının sistem yönüyle doğal hizalanmasının istatistiki olarak anlamlı bir yönsel üstünlük sağladığı gösterilememiştir.  
Bu ampirik parametre altında portföy getirileri ve 7 günlük blok bootstrap güven aralıkları:
- **Her Zaman Açık:** **31,86 TL/MWh** [%95 GA: -2,25; 74,92]
- **Aşama 1 Kalibre:** **26,84 TL/MWh** [%95 GA: 4,54; 57,59]
- **Fark [Aşama 1 − Açık]:** **-5,02 TL/MWh** [%95 GA: -34,41; +20,71] ($p > 0,60$, farksız).

#### $\alpha = 0,50$ Matematiksel İspatı (Sistem Yönünden Bağımsızlık):
Matriste $\alpha = 0,50$ yerine konduğunda, Uzun pozisyon ($a_t = +1$) için her iki sistem durumundaki getiri formülü:
- $D_t = 1$ durumunda: $0,5 (P_{\text{neg}, t} - PTF_t) - 0,5 (PTF_t - P_{\text{pos}, t}) = \mathbf{0,5 (P_{\text{neg}, t} + P_{\text{pos}, t} - 2 PTF_t)}$
- $D_t = -1$ durumunda: $-0,5 (PTF_t - P_{\text{pos}, t}) + 0,5 (P_{\text{neg}, t} - PTF_t) = \mathbf{0,5 (P_{\text{neg}, t} + P_{\text{pos}, t} - 2 PTF_t)}$

Formül $D_t$'den tamamen bağımsızdır. $\alpha = 0,50$ durumunda getirinin $+29,62$ TL çıkması sistemin açıkta olma frekansından değil; **piyasadaki dengesizlik ceza marjı yapısının asimetrisinden** ($(P_{\text{neg}} - PTF) > (PTF - P_{\text{pos}})$) ve ampirik platform anomalilerinden kaynaklanmaktadır. Nitekim Tablo 4D'de gösterildiği üzere, test yılı genelinde ortalama yayılım $+11,86$ TL/MWh'dir; $\alpha = 0,50$ portföyünde yayılım bileşeni yalnızca $0,5 \times 11,86 \approx 5,93$ TL ($\approx 6$ TL) katkı üretirken, kalan yaklaşık $23,69$ TL doğrudan ceza marjı yapısı, yön-yayılım uyumsuzlukları ve ampirik veri kalıntılarından türemektedir. Bu durum naif kuralın getirisinin bir yönsel öngörü becerisi değil, piyasa ceza kuralının asimetrik yapısı olduğunu gösterir.

---

## 6. Kanonik Mutabakat Tablosu: Genel P&L ve 3-Rejim Ayrıştırması

#### Tablo 12: Kanonik Mutabakat Tablosu (N = 8.760 Saat / 365 Gün)

| Strateji | Genel P&L (TL/MWh) | %95 Blok Bootstrap GA | Rejim 1 (N=2.208) | Rejim 2 (N=2.232) | Rejim 3 (N=4.320) | Ağırlıklı Toplam | Mutabakat Farkı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle (Tam Bilgi / Teorik Tavan)** | **820,38** | [756,25; 884,46] | 525,61 TL | 890,99 TL | 934,57 TL | **820,38 TL** | **0,00 TL** |
| **Her Zaman Açık (Çoğunluk)** | **71,73** | [6,51; 153,68] | -20,87 TL | -0,80 TL | 156,54 TL | **71,73 TL** | **0,00 TL** |
| **Her Zaman Fazla (Ters Kural)** | **-74,45** | [-156,16; -9,42] | +16,01 TL | -1,38 TL | -158,43 TL | **-74,45 TL** | **0,00 TL** |
| **Lag-168 Naif Takipçi** | **19,30** | [-21,41; 60,21] | +36,43 TL | +19,90 TL | +10,23 TL | **19,30 TL** | **0,00 TL** |
| **Aşama 1 (Ham, $\tau=70$ TL)** | **55,95** | [12,52; 110,58] | +47,64 TL | -68,66 TL | +124,59 TL | **55,95 TL** | **0,00 TL** |
| **Aşama 1 (Platt Kalibre, $\tau=70$ TL)** | **67,33** | [22,84; 127,00] | +28,39 TL | -59,58 TL | +152,80 TL | **67,33 TL** | **0,00 TL** |
| **Aşama 2 (GÖP Sonrası Yayılım)** | **119,14** | [61,79; 184,53] | +69,08 TL | +150,93 TL | +128,29 TL | **119,14 TL** | **0,00 TL** |

*Eşleştirilmiş 7 Günlük Blok Bootstrap Fark Analizleri (Strateji − Her Zaman Açık):*
- **[Aşama 1 Kalibre − Açık]:** **-4,41 TL/MWh** [%95 GA: -59,28; +44,81] ($p > 0,75$, istatistiki olarak farksız).
- **[Aşama 2 Yayılım − Açık]:** **+47,40 TL/MWh** [%95 GA: -30,95; +124,29] ($p = 0,22$, sıfırı içermektedir).

---

## 7. Risk Profili, Kuyruk Riski ve Permütasyon Sınaması

### 7.1. Risk, Çekilme (Drawdown) ve Granülarite Analizi

#### Tablo 13: Risk ve Oynaklık Profili Karşılaştırması

| Risk Göstergesi | Her Zaman Açık (Naif Kural) | Aşama 1 Platt Kalibre | Aşama 2 Yayılım (Teorik) |
| :--- | :---: | :---: | :---: |
| **Yıllık Ortalama Getiri (TL/MWh)** | 71,73 | 67,33 | 119,14 |
| **Aylık Standart Sapma (TL/MWh)** | 216,27 | **155,62 (%28 daha düşük)** | 178,16 |
| **En Kötü Ay Getirisi (TL/MWh)** | -169,35 (Şubat 2026) | **-75,76 (Mart 2026 - Kayıp %55 sınırlı)** | -103,43 (Eylül 2026) |
| **Negatif Getirili Ay Sayısı (N=12)** | 5 / 12 (Eki, Ara, Şub, Tem, Eyl) | 5 / 12 (Ara, Oca, Şub, Mar, Eyl) | 3 / 12 |
| **Günlük %95 CVaR (TL/Gün)** | -16.210,40 TL | **-13.646,40 TL** | -8.114,20 TL |
| **Kümülatif MDD (Günlük Granülarite)** | **224.340,12 TL/MW** | **183.188,61 TL/MW (%18,3 düşüş)** | 104.026,23 TL/MW |
| **Kümülatif MDD (Saatlik Granülarite)** | 236.184,49 TL/MW | 187.459,90 TL/MW (%20,6 düşüş) | 109.978,13 TL/MW |
| **Kümülatif MDD (Aylık Granülarite)** | 145.666,20 TL/MW | 126.102,00 TL/MW (%13,4 düşüş) | 97.444,10 TL/MW |
| **Aylıklandırılmış Sharpe Oranı ($N=12$)** | 1,14 [%95 GA: -0,92; 3,00] | 1,50 [%95 GA: -0,19; 3,08] | 2,36 [%95 GA: 0,82; 3,95] |
| **Aktif Saat Payı (%)** | %100,00 (8.760 saat) | **%48,94 (4.287 saat aktif)** | %100,00 (8.760 saat) |
| **Aktif Saat Başına P&L (TL/MWh)** | 71,73 TL/MWh | **137,57 TL/MWh (1,92 kat yoğunluk)** | 119,14 TL/MWh |

---

### 7.2. Haziran 2026 Ayı Bağımlılığı (June Dominance)

Yıllık nakit akışı ayrıştırıldığında sonuçların aşırı derecede tek bir aya bağımlı olduğu görülmektedir:
- **Haziran 2026 Nakit Akışı Payı:** Aşama 1 modelinin test yılındaki toplam net getirisi **589.780,73 TL** olup, bunun **364.926,88 TL'si (%61,9)** yalnızca Haziran 2026'da gerçekleşmiştir. Naif Açık kuralında da yıllık 628.380,18 TL'nin **414.178,00 TL'si (%65,9)** Haziran ayına aittir.
- **Haziran Hariç 11 Ayın Ortalaması (8.040 Saat):**
  - **Aşama 1 Kalibre:** **27,97 TL/MWh**
  - **Her Zaman Açık:** **26,64 TL/MWh**
  - **Fark [Aşama 1 − Açık]:** Sadece **+1,32 TL/MWh**!
- **Aylık Korelasyon:** İki stratejinin 12 aylık P&L serileri arasındaki Pearson korelasyonu **$r = 0,796 \approx 0,80$**'dir.

---

### 7.3. Maruziyet-Eşleşmiş Permütasyon Sınaması ve Aşağı Yönlü Risk Analizi

Aşama 1 modelinin sağladığı çekilme ve oynaklık düşüşünün bir "akıllı risk filtresi başarısı" mı yoksa maruziyeti yarıya (%48,94) indirmenin doğrudan mekanik bir sonucu mu olduğunu test etmek amacıyla; modelin eylemleri aynı aktiflik oranıyla 365 gün boyunca **2.000 kez rastgele karıştırılmış (maruziyet-eşleşmiş permütasyon boş dağılımı)** ve aşağı yönlü risk ölçüleri kıyaslanmıştır:

#### Tablo 14: Maruziyet-Eşleşmiş Permütasyon Boş Dağılım Sınaması (N = 2.000 Tekrar, Tüm Metrikler Tek Koşudan)

| Ekonometrik Gösterge | Aşama 1 Gerçek Değeri | Boş Dağılım Ortalaması | Boş Dağılım %95 GA | $p_{\text{tek}}$ (Tek Yönlü) | $p_{\text{çift}}$ (İki Yönlü) | Holm Düzeltmeli $p$ | İstatistiki Sonuç |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Aylık Standart Sapma (TL/MWh)** | **155,62** | 86,25 | [56,86; 117,51] | 1,0000 | < 0,001 | **0,003** | Model oynaklığı rastgeleden yüksek (Haziran etkisi) |
| **P&L (Tüm Yıl - TL/MWh)** | **67,33** | 30,62 | [5,37; 54,80] | 0,0010 | 0,0020 | **0,010** | Anlamlı ($p < 0,05$) |
| **Günlük %95 CVaR (TL/Gün)** | **-13.646,40** | -15.495,30 | [-17.837,81; -13.378,77] | 0,0470 | 0,0940 | **0,376** | Çoklu test sonrası anlamsız ($p > 0,05$) |
| **P&L (Haziran Hariç - TL/MWh)** | **27,97** | 15,96 | [-9,06; 39,83] | 0,1840 | 0,3680 | **1,000** | **Anlamsız ($p > 0,10$)** |
| **En Kötü Ay Getirisi (TL/MWh)** | **-75,76** | -96,80 | [-181,47; -34,96] | 0,3180 | 0,6360 | **1,000** | Anlamsız ($p > 0,50$) |
| **Kümülatif Günlük MDD (TL/MW)** | **183.188,61** | 173.215,05 | [77.992,45; 281.585,26] | 0,5925 | 0,8150 | **1,000** | Anlamsız ($p > 0,50$) |

*Metodolojik ve Yönsel İzah ($p$-Değerleri Tanımı ve Holm-Bonferroni Sıralaması):*
- $p_{\text{tek}}$: Boş dağılımın gerçek modele kıyasla daha iyi veya eşit performans gösterme olasılığıdır ($P(\text{Boş} \text{ daha iyi})$). 2.000 tekrarlı permütasyonda sayısal çözünürlük $\approx 0,0005$ olup sıfır gözlenen değerler $p < 0,001$ olarak raporlanmıştır.
- P&L göstergelerinde $P(\text{Boş} \ge \text{Model})$, CVaR ve En Kötü Ayda $P(\text{Boş} \ge \text{Model})$ (kayıp büyüklüğü daha az), MDD'de ise $P(\text{Boş} \le \text{Model})$ (çekilme daha düşük) olarak tanımlanmıştır.
- **Holm-Bonferroni Çoklu Test Düzeltmesi ($m=6$ İki Yönlü Test):** 6 gösterge üzerinden uygulanan adım-küçültme (step-down) prosedürü:
  1. Aylık SD ($p < 0,001 \rightarrow p_{\text{adj}} \approx 0,003 \le 0,05$, anlamlı: model oynaklığı Haziran sıçraması nedeniyle rastgeleden yüksektir).
  2. Tüm Yıl P&L ($p = 0,0020 \rightarrow p_{\text{adj}} = 0,010 \le 0,05$, anlamlı).
  3. Günlük CVaR ($p = 0,0940 \rightarrow p_{\text{adj}} = 0,376 > 0,05$, çoklu test sonrası anlamsız).
  4. Haziran Hariç P&L ($p = 0,3680 \rightarrow p_{\text{adj}} = 1,000$, anlamsız).
  5. En Kötü Ay ($p = 0,6360 \rightarrow p_{\text{adj}} = 1,000$, anlamsız).
  6. Günlük MDD ($p = 0,8150 \rightarrow p_{\text{adj}} = 1,000$, anlamsız).
- **Kuyruk Riski (CVaR) Yorumu:** Modelin günlük CVaR değeri (-13.646 TL), boş dağılım ortalamasından (-15.495 TL) tek yönlü bakıldığında iyi görünse de ($p_{\text{tek}} = 0,0470$, iki yönlü $p = 0,0940$), 6 test üzerinden uygulanan Holm-Bonferroni düzeltmesi ($p_{\text{adj}} = 0,376$) sonrasında istatistiksel anlamlılığını kaybetmektedir.
- **Genel Değerlendirme:** Aşama 1 maruziyeti yaklaşık yarıya indirir; getiri ve çekilmedeki azalma bununla tutarlıdır. Çoklu test düzeltmesi sonrası risk metrikleri rastgele zamanlamadan farksız olup, modelin akıllı bir risk filtresi olarak etkinliği gösterilememiştir.

---

### 7.4. 12 Aylık P&L Tablosu ve Yıllık Ağırlıklı Ortalamalar

#### Tablo 15: 12 Aylık P&L, Nakit Akışı ve Sistem Fiyat Göstergeleri

| Ay | Saat ($N$) | Her Zaman Açık (TL/MWh) | Açık Toplam Nakit ($N \times \text{P\&L}$) | Aşama 1 Kalibre (TL/MWh) | Aşama 1 Toplam Nakit ($N \times \text{P\&L}$) | Aşama 2 Yayılım (TL/MWh) | Aşama 2 Toplam Nakit ($N \times \text{P\&L}$) | Ortalama PTF | Ortalama SMF |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-10** | 744 | -94,13 | -70.032,66 TL | +25,92 | +19.283,73 TL | +86,05 | +64.019,65 TL | 2.739,50 TL | 2.599,90 TL |
| **2025-11** | 720 | +38,00 | +27.359,82 TL | +72,22 | +51.997,37 TL | +62,36 | +44.897,71 TL | 2.784,10 TL | 2.765,46 TL |
| **2025-12** | 744 | -4,57 | -3.400,29 TL | -11,55 | -8.594,00 TL | +58,62 | +43.612,80 TL | 2.973,04 TL | 2.911,50 TL |
| **2026-01** | 744 | +19,10 | +14.210,71 TL | -32,36 | -24.077,84 TL | +6,93 | +5.152,57 TL | 2.894,92 TL | 2.760,64 TL |
| **2026-02** | 672 | -169,35 | -113.801,76 TL | -55,16 | -37.069,37 TL | +301,35 | +202.507,65 TL | 2.078,20 TL | 1.850,99 TL |
| **2026-03** | 744 | +105,97 | +78.841,78 TL | -75,76 | -56.369,07 TL | +124,75 | +92.816,68 TL | 1.620,32 TL | 1.687,62 TL |
| **2026-04** | 720 | +274,99 | +197.989,52 TL | +82,98 | +59.745,55 TL | +169,61 | +122.119,27 TL | 921,06 TL | 1.185,86 TL |
| **2026-05** | 744 | +264,95 | +197.125,91 TL | +80,32 | +59.755,67 TL | +214,93 | +159.910,82 TL | 590,90 TL | 875,30 TL |
| **2026-06** | 720 | +575,25 | +414.178,00 TL | +506,84 | +364.926,88 TL | +563,95 | +406.043,75 TL | 1.240,16 TL | 1.761,27 TL |
| **2026-07** | 744 | -99,09 | -73.723,67 TL | +67,66 | +50.336,01 TL | -19,62 | -14.595,81 TL | 2.699,61 TL | 2.515,33 TL |
| **2026-08** | 744 | +104,39 | +77.666,34 TL | +171,76 | +127.789,97 TL | -11,26 | -8.376,03 TL | 2.907,81 TL | 2.905,90 TL |
| **2026-09** | 720 | -163,94 | -118.033,53 TL | -24,92 | -17.944,17 TL | -103,43 | -74.471,87 TL | 2.742,43 TL | 2.507,16 TL |
| **Kümülatif Toplam (TL)** | **8.760** | — | **+628.380,18 TL** | — | **+589.780,73 TL** | — | **+1.043.637,19 TL** | — | — |
| **Ağırlıklı Yıllık Ortalama** | **8.760** | **71,73 TL/MWh** | **71,7329 TL** | **67,33 TL/MWh** | **67,3266 TL** | **119,14 TL/MWh** | **119,1367 TL** | **2.186,39 TL** | **2.198,25 TL** |

---

## 8. Keşifsel Gözlem Penceresi (Ekim 2026 / 12 Gün / 288 Saat)

Modellerin gelişim sürecinde incelenen 01–12 Ekim 2026 tarihleri arasındaki 288 saatlik dönem, **keşifsel (exploratory) bir vaka incelemesi** olarak ele alınmıştır. Kanonik model ağırlıkları 15 Ekim 2026 21:10 itibarıyla diske kaydedilmiş; modeller, veri kütüğü ve değerlendirme kodları 20 Ekim 2026 tarihinde açıklamalı `protocol-v1` etiketiyle dondurulmuştur. Değişmezlik güvencesi bu etiketin uzak depoya itilmesi ve commit hash'i ile `MANIFEST_SHA256.txt` özetinin kayıt e-postasıyla danışman öğretim üyesine iletilmesiyle tamamlanacaktır. Tarafsız, veri sızıntısından arındırılmış bağımsız ileriye dönük değerlendirme penceresi, harici zaman damgasını sabitleyen kayıt e-postasının gönderildiği günü takip eden ilk tam gün olan **21 Ekim 2026 00:00:00+03:00 – 31 Mart 2027 23:00:00+03:00** tarihleri arasını (**162 gün / 3.888 saat**; 23 tam hafta + 1 artık gün) kapsamak üzere protokole bağlanmıştır (`reports/prospective_evaluation_protocol_2026_2027.md`):

#### Tablo 16: Keşifsel Gözlem Penceresi Sonuçları (288 Saat)

| Strateji | 12 Günlük Ortalama (TL/MWh) | %95 Günlük Bootstrap GA | Eylem Dağılımı (%) |
| :--- | :---: | :---: | :--- |
| **Oracle (Teorik Üst Sınır)** | **876,36** | [725,69; 1.039,18] | — |
| **Her Zaman Fazla** | **+397,52** | [183,90; 647,89] | %100,00 Kısa |
| **Aşama 2 (GÖP Sonrası Yayılım - Teorik)** | **+25,25** | [-169,14; 218,68] | %44,44 Uzun, %55,56 Kısa |
| **Pasif / Sıfır Hareket** | **0,00** | [0,00; 0,00] | %100,00 Pasif |
| **Aşama 1 (GÖP Öncesi Kalibre)** | **-91,50** | [-246,07; 61,49] | **%33,68 Uzun, %4,17 Kısa, %62,15 Pasif** |
| **Her Zaman Açık (Naif Çoğunluk)** | **-405,76** | [-644,57; -179,99] | %100,00 Uzun |

*Ekonometrik Değerlendirme:*
1. **Rejim Değişimi:** Ekim 2026'nın ilk 12 gününde piyasa belirgin bir Enerji Fazlası rejimine girmiştir (%55,9 Fazla, %38,9 Açık). Sabit "Her Zaman Açık" kuralı **-405,76 TL/MWh** seviyesinde ciddi bir asimetrik kayba uğramıştır.
2. **Modelin Kaybı:** Aşama 1 modeli eylemlerinin %62,15'inde pasif kalarak naif kuralın kaybını törpülemiş olsa da, **-91,50 TL/MWh ile mutlak bazda zarar etmiş ve tamamen pasif kalma stratejisinin ($0,00$ TL) gerisinde kalmıştır.** Bu bulgu, modelin bir sermaye koruma garantisi sunmadığını göstermektedir.

---

## 9. Lisansüstü Tez Bulguları ve Ekonometrik Tartışma Sentezi

Tez kapsamında yürütülen ampirik deneyler, ekonometrik testler ve duyarlılık analizleri aşağıdaki 7 temel bilimsel sonucu ortaya koymuştur:

1. **PTF, SMF'nin Baskın Bilgi Çapasıdır ($L_1$ vs $L_2$ Ayrımı):**  
   Türkiye elektrik piyasasında SMF tahmininde en kritik bilgi girdisi PTF'dir. Saatlerin %21,56'sında $PTF=SMF$ gerçekleşmekte olup medyan yayılım sıfırdır. Bu nedenle modelsiz $SMF=PTF$ naif kuralı, mutlak hata kaybında ($L_1$ / MAE) tüm makine öğrenmesi modellerini geride bırakmaktadır (682,75 TL vs 748,14 TL). Karesel hatada ($L_2$ / RMSE) yayılım modelinin sağladığı 12,48 TL'lik fark ise Harvey-Leybourne-Newbold testi ile doğrulandığı üzere **istatistiki bir üstünlük göstermemektedir ($p = 0,2702$)**.
2. **Sınıflandırma Becerisi İstatistiki Olarak Anlamlı Fakat Sınırlıdır:**  
   GÖP öncesinde makine öğrenmesi modeli sistem yönünü %59,66 doğrulukla tahmin ederek çoğunluk sınıfına (%55,74) göre +3,92 puanlık mütevazı bir yönsel fark sağlamıştır (Holm-düzeltmeli $p_{\text{adj}} \approx 0,034$). Ancak saat dilimi analizi, bu başarının yalnızca sabah (+11,87 puan) ve gündüz (+7,76 puan) saatlerinde üretildiğini; akşam zirvesinde çoğunluğun gerisinde kalındığını (-1,01 puan), gece saatlerinde ise çoğunluğun 2,97 puan altında kalındığını (%46,21 vs %49,18) ortaya koymaktadır.
3. **P&L Spread Büyüklüğüne Bağlıdır, Yön Etiketine Değil; Sınıflandırıcı Hedefle Uyumsuzdur:**  
   GÖP sonrası dönemde PTF bilgisinin dahil edilmesiyle yön doğruluğu %59,66'dan %62,21'e yükselmekte, BSS skoru ise %6,43'ten %12,33'e çıkmaktadır. Ancak bu olasılık kalitesi sınıflandırıcı tabanlı kararlarda ekonomik getiriye dönüşmemekte; marjinal beklenen değer kuralı $\tau=70$ TL altında **-20,81 TL/MWh** (doğrulamada optimize edilmiş $\tau=140$ TL ile **-7,46 TL/MWh**, argmax kuralı ile **-33,18 TL/MWh**) zarar üretmektedir. Aşama 2'nin raporlanan **+119,14 TL/MWh** getirisi sınıflandırıcıya değil, $L_2$ koşullu ortalama yayılım regresyonunun işaret kuralına ($\hat{S}_{L2} > 0$) bağlıdır.  
   - **Hedef Uyumsuzluğu ve Mekanizma Ayrımı (Spread vs Ceza Marjı):** Uzun pozisyon getirisi Enerji Açığı saatinde $\approx S^+ + k \cdot \max(PTF, SMF)$, Enerji Fazlası saatinde ise $\approx -S^- - l \cdot \min(PTF, SMF)$ biçimindedir ($S = SMF - PTF$). Burada kritik ekonometrik ayrım şudur:  
     1. Pozisyonunu yayılım tahminine göre değiştiren modeller ($L_2$ regresörü ve PTF eşik kuralları) için **getiri doğrudan $SMF - PTF$ spread'inin işaretine ve büyüklüğüne bağlıdır** (Tablo 4D'nin kanıtladığı üzere $L_2$'nin +119,14 TL getirisinin +150,50 TL'si yayılımdan gelir). Yön etiketini ($D_t$) tahmin eden sınıflandırıcı ise büyüklük duyarsızlığı nedeniyle bu hedefle yapısal olarak uyumsuzdur; nitekim argmax kuralında yön doğruluğu %62,21 olan sınıflandırıcı **-33,18 TL/MWh** (beklenen değer kuralında $\tau=70$ ile **-20,81 TL/MWh**) kaybettirirken, yön doğruluğu yalnızca %41,7 olan $L_2$ regresörü **+119,14 TL/MWh** kazandırmaktadır.  
     2. Buna karşılık naif referans kural ('Her Zaman Açık') için yayılım belirleyici değildir; ortalama yayılım (+11,86 TL) naif getirinin yalnızca %16,5'ini açıklamakta, getirinin asıl gövdesi (+95,69 TL) yasal dengesizlik ceza katsayısı ($k=0,06$) asimetrisinden türemektedir.
   - **Normal Saatler ve Ekstrem Dilim Dekompozisyonu:** Tablo 4B'nin gösterdiği üzere iki model arasındaki **+139,95 TL/MWh**'lik farkın ana taşıyıcısı, **normal piyasa saatlerindeki büyüklük ağırlıklı koşullu ortalama kararlarıdır (+167,71 TL net fark katkısı, %119,8 pay)**. Ekstrem sıçrama saatleri (88 saat) ise +42,40 TL (%30,3) katkı sağlamaktadır (kalan tavan ve anomali saatlerinin -70,16 TL'lik dengeleyici etkisiyle net 139,95 TL).
   - **Ortalamaya Dönüş ve PTF Seviye Etkisi:** $L_2$ regresörünün genel yön isabeti (%41,7) çoğunluk sınıfının (%55,7) ve şans düzeyinin altındadır. Tablo 4C bulgularının gösterdiği üzere bu getiri karmaşık bir yön tahmininden değil; sıfır tabanı ve fiyat tavanlarının dayattığı **PTF seviyesi ve yapısal ortalamaya dönüş (mean-reversion)** dinamiğinden kaynaklanmaktadır. Nitekim iki değişkenli doğrusal regresyon ($PTF$ + Saat) tek başına **+108,51 TL** (L2 ile fark: -10,63 TL [%95 GA: -68,71; +41,21]), tek değişkenli doğrusal regresyon (yalnızca $PTF$) ise **+94,77 TL** (L2 ile fark: -24,36 TL [%95 GA: -87,65; +29,16]) üretmektedir. Eğitim kümesi PTF medyanına dayanan deterministik *"PTF < 2.500 TL ise Uzun"* kuralı nokta tahmininde **+140,57 TL** üretse de, L2 ile bloklu bootstrap farkı (+21,43 TL [%95 GA: -14,85; +62,71]) sıfırı kapsadığından bu üstünlük istatistiksel olarak kesin bir üstünlük değil, nokta tahmini düzeyindedir. PTF öznitelikleri çıkarıldığında ise model getirisi doğrudan naif kural düzeyine (**+70,73 TL**) inmekte ve $L_2$'ye göre istatistiki olarak anlamlı düzeyde kaybetmektedir (-48,40 TL [%95 GA: -84,83; -13,95], $p < 0,01$).
4. **Ekonomik Üstünlük (Alfa) Gösterilememiştir:**  
   Yönsel tahmin başarısı ekonomik bir avantaja dönüşmemektedir. Yıllık toplam P&L'de Aşama 1 yön modeli (**67,33 TL/MWh**), naif "Her Zaman Açık" kuralının (**71,73 TL/MWh**) altında kalmıştır. Eşleştirilmiş blok bootstrap farkı (-4,41 TL/MWh [%95 GA: -59,28; +44,81], $p > 0,75$) iki stratejinin getirisinin istatistiksel olarak başa baş olduğunu göstermektedir. Bu durum EPİAŞ'ın -106 / -103 TL'lik ampirik veri kalıntılarından bağımsız olup kuramsal yasal formül duyarlılık analizinde de aynen geçerlidir.
5. **Sonuçlar Rejim ve Tek Ay Bağımlılığı Taşımaktadır:**  
   Her iki stratejinin yıllık nakit akışının %60'ından fazlası tek bir aya (Haziran 2026) aittir. Haziran ayı dışlandığında model ile naif kuralın getirisi eşitlenmekte (27,97 TL vs 26,64 TL; fark +1,32 TL), permütasyon testi zamanlama anlamlılığını tamamen kaybetmektedir ($p = 0,1840$, iki yönlü $p = 0,3680$).
6. **Algoritmik Başarı Değil, Mekanik Maruziyet Kısıtı:**  
   Aşama 1 modelinin oynaklığı %28, kümülatif maksimum çekilmeyi ise %18,3 düşürmesi algoritmik bir başarı değildir. Maruziyet-eşleşmiş permütasyon sınamasında modelin CVaR ($p_{\text{adj}} = 0,3760$), en kötü ay ($p = 0,6360 \rightarrow p_{\text{adj}} = 1,000$) ve MDD ($p = 0,8150 \rightarrow p_{\text{adj}} = 1,000$) değerleri rastgele zamanlama boş dağılımından farksız çıkmıştır. **Aşama 1 maruziyeti yaklaşık yarıya indirir; getiri ve çekilmedeki azalma bununla tutarlıdır. Akıllı bir risk filtresi olarak etkinliği gösterilememiştir.**
7. **Tezin Savunulabilir Temel Çıkarımı:**  
   GÖP öncesi makine öğrenmesi modelleri piyasada spekülatif bir alfa makinesi değildir. $PTF$ fiyatının açıklanmadığı karar anında üretilen modeller, piyasanın yapısal ceza marjı asimetrisi ve tek yönlü çoğunluk kuralları karşısında net bir ekonomik üstünlük kuramamaktadır. Makine öğrenmesinin katkısı, yönsel olasılık kalibrasyonu sağlayarak belirsiz saatlerde piyasa maruziyetini mekanik olarak kesmekle sınırlı kalmaktadır.
