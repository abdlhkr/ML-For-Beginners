# Confusion Matrix (Karmaşıklık Matrisi) ve Sınıflandırma Metrikleri

## Senaryo: E-posta Spam (İstenmeyen) Filtresi

Bir e-posta servis sağlayıcısında çalıştığınızı ve yeni bir yapay zeka modeli geliştirdiğinizi hayal edin. Amacınız gelen e-postaların **"Spam"** olup olmadığını tahmin etmek.

*   **Pozitif Sınıf (Hedefimiz):** Spam E-posta
*   **Negatif Sınıf:** Normal (Önemli/Gerçek) E-posta

Sistemi test etmek için elimizde ne olduğunu bildiğimiz **1000 adet** e-posta var:
*   Gerçekte Normal olanlar: **850 adet**
*   Gerçekte Spam olanlar: **150 adet**

Modelimizi çalıştırıyoruz ve sonuçları aşağıdaki tabloya, yani **Karmaşıklık Matrisine (Confusion Matrix)** yerleştiriyoruz:

| Gerçek \ Tahmin | Tahmin: Normal | Tahmin: Spam (Pozitif) | Toplam (Support) |
| :--- | :--- | :--- | :--- |
| **Gerçek: Normal** | **TN = 800** | **FP = 50** | 850 |
| **Gerçek: Spam** | **FN = 30** | **TP = 120** | 150 |

---

### 1. Temel Terimlerin (Hücrelerin) Anlamı

*   **True Negative (TN - Doğru Negatif) = 800:**
    E-posta gerçekte Normaldi, modelimiz de "Normal" dedi. (Kullanıcının gelen kutusuna doğru bir şekilde düştü).
*   **False Positive (FP - Yanlış Pozitif) = 50:**
    E-posta gerçekte Normaldi (belki patronun attığı bir mail), ama modelimiz yanlışlıkla "Spam" dedi. *(En tehlikeli hata türlerinden biri!)*
*   **False Negative (FN - Yanlış Negatif) = 30:**
    E-posta gerçekte Spamdi (reklam veya virüs), ama modelimiz bunu kaçırdı ve "Normal" dedi.
*   **True Positive (TP - Doğru Pozitif) = 120:**
    E-posta gerçekte Spamdi, modelimiz de başarıyla "Spam" diyerek engelledi.

---

### 2. Metriklerin Hesaplanması (Spam Sınıfı İçin)

Modelimizin "Spam" yakalama performansını ölçelim.

#### Accuracy (Doğruluk)
Model, 1000 mailin toplam kaç tanesini doğru tahmin etti?
`Accuracy = (TP + TN) / Toplam Veri = (120 + 800) / 1000 = 0.92 (%92)`

> **Yorum:** "Modelim %92 oranında doğru çalışıyor." Kulağa harika geliyor değil mi? Ama veri setimiz dengesiz (850'si zaten normal). Bu yüzden Accuracy tek başına yanıltıcıdır, diğer metrikleri de görmeliyiz.

#### Precision (Hassasiyet)
Model bir e-postaya **"Spam" dediğinde**, bu tahminin yüzde kaçı GERÇEKTEN Spam'di?
`Precision = TP / (TP + FP) = 120 / (120 + 50) = 120 / 170 ≈ 0.70 (%70)`

> **Yorum:** Model "Spam" diyerek çöpe attığı her 100 e-postanın 30'unda hata yapıyor (FP=50 yüzünden). Yani hassasiyetimiz %70. Önemli mailleri çöpe atma riskimiz var.

#### Recall (Duyarlılık / Kapsayıcılık)
Sistemdeki **gerçek Spam e-postaların** yüzde kaçını bulabildik/yakalayabildik?
`Recall = TP / (TP + FN) = 120 / (120 + 30) = 120 / 150 = 0.80 (%80)`

> **Yorum:** Bize gelen gerçek spam maillerin %80'ini başarıyla yakalayıp engelledik. Geri kalan %20'si (FN=30) maalesef kullanıcıların gelen kutusuna sızdı.

#### F1-Score
Precision ve Recall arasında bir denge (harmonik ortalama) kurar. Eğer biri çok yüksek diğeri çok düşükse, F1 skoru bu dengesizliği cezalandırır.
`F1 = (2 * Precision * Recall) / (Precision + Recall) = (2 * 0.70 * 0.80) / (0.70 + 0.80) ≈ 0.74 (%74)`

> **Yorum:** Spam sınıfı için genel performans puanımız %74.

#### Support (Destek / Veri Adedi)
Değerlendirdiğimiz sınıfın gerçek veri setinde kaç kez geçtiğidir.
> **Yorum:** Spam sınıfı için Support = 150. Normal sınıfı için Support = 850.

---

### 3. Ortalamalar (Macro vs. Weighted Avg)

Modelin sadece "Spam" için değil, "Normal" mailler için de (TN ve FN kullanılarak) ayrı bir Precision, Recall ve F1 skoru vardır. 

*   *Normal sınıfı için hesapladığımızda F1 Skoru yaklaşık **0.95** çıkar (çünkü veri çoktur ve TN çok yüksektir).*
*   *Spam sınıfının F1 Skoru ise **0.74**'tü.*

Modelin **Genel Başarısını (Ortalamasını)** nasıl raporlayacağız? İki farklı yöntem vardır:

#### Macro Avg (Makro Ortalama)
Sınıfların veri sayısını (150'ye 850 olmasını) **hiç umursamaz.** Her sınıf eşit ağırlıktaymış gibi iki F1 skorunu toplayıp ikiye böler. Azınlık sınıfının performansına ne kadar değer verdiğinizi gösterir.
`Macro F1 = (0.74 + 0.95) / 2 = 0.845 (%84.5)`

#### Weighted Avg (Ağırlıklı Ortalama)
Sınıfların büyüklüğünü (Support) hesaba katar. "Normal maillerden 850 tane var, o halde onun skoru ortalamaya daha çok etki etmeli" der. 
`Weighted F1 = ((0.74 * 150) + (0.95 * 850)) / 1000 ≈ 0.918 (%91.8)`

**Kısa Özet:** 
Makro ortalama %84 iken, ağırlıklı ortalama %91 çıktı. Eğer raporunuzda "Ağırlıklı F1 skorumuz %91" derseniz çok başarılı görünür, ancak "Makro skorumuz %84" derseniz yöneticileriniz azınlık olan Spam sınıfında (asıl çözmek istediğiniz problemde) zorlandığınızı daha net görür.
