# Analisis Sentimen Multi-Aspek Ulasan MySiloam Menggunakan BERTopic dan SVM

## 📌 Deskripsi Proyek

Repository ini merupakan implementasi penelitian skripsi yang berfokus pada **analisis sentimen multi-aspek** terhadap ulasan pengguna aplikasi **MySiloam** yang dikumpulkan dari Google Play Store dan App Store periode 2019–2025.

Penelitian ini mengombinasikan **BERTopic** untuk ekstraksi aspek layanan dan **Support Vector Machine (SVM)** untuk klasifikasi sentimen guna memahami pengalaman pengguna secara lebih mendalam.

## 🎯 Tujuan Penelitian

* Mengidentifikasi aspek layanan utama yang dibahas pengguna MySiloam.
* Membandingkan performa beberapa algoritma clustering dalam BERTopic.
* Mengklasifikasikan sentimen ulasan berdasarkan aspek layanan.
* Memberikan insight yang dapat digunakan untuk evaluasi dan pengembangan aplikasi kesehatan digital.

## 📊 Dataset

* Sumber: Google Play Store & Apple App Store
* Periode: 2019–2025
* Total ulasan terkumpul: 2.657
* Total ulasan setelah preprocessing: 1.699

## 🔍 Metodologi

### 1. Data Collection

* Scraping ulasan aplikasi MySiloam.
* Penggabungan data dari Google Play Store dan App Store.

### 2. Data Preprocessing

* Case folding
* Cleaning text
* Tokenization
* Stopword removal
* Stemming

### 3. Topic Modeling

Menggunakan BERTopic dengan tiga algoritma clustering:

* HDBSCAN
* BIRCH
* K-Means

Evaluasi topic modeling dilakukan menggunakan:

* C_v Coherence
* UMass Coherence
* NPMI
* Topic Diversity

### 4. Sentiment Classification

Menggunakan:

* Support Vector Machine (SVM)
* One-vs-One (OvO)

Dua skenario klasifikasi:

* Pendekatan dua tahap
* Pendekatan klasifikasi gabungan

## 🏆 Hasil Penelitian

### Topic Modeling Terbaik

* BERTopic + K-Means + Stemming
* C_v Score: 0.4113

### Aspek yang Ditemukan

1. Fitur Medis Aplikasi
2. Pelayanan dan Kepuasan
3. Teknis Aplikasi

### Konsistensi Pelabelan

* Krippendorff's Alpha: 0.8816

### Performa Klasifikasi Sentimen

| Metode               | F1-Score |
| -------------------- | -------- |
| Pendekatan Dua Tahap | 89.53%   |
| Pendekatan Gabungan  | 82.74%   |

Hasil menunjukkan bahwa pendekatan dua tahap menghasilkan performa yang lebih baik dibandingkan klasifikasi gabungan.

## 📂 Struktur Repository

```text
├── scrapping.ipynb
├── data_prepocessing.ipynb
├── topic_modeling_skenario1.ipynb
├── topic_modeling_skenario2_&_evaluasi_topic_modeling.ipynb
├── pelatihan_model_&e_evaluasi_model.ipynb
└── mysiloam_project/
```

## 🛠️ Tools & Libraries

* Python
* BERTopic
* Scikit-Learn
* Pandas
* NumPy
* NLTK
* Sastrawi
* Matplotlib
* Seaborn
* Jupyter Notebook

## 📈 Kontribusi

Penelitian ini menunjukkan bahwa kombinasi **BERTopic berbasis K-Means** dan **SVM One-vs-One** efektif untuk melakukan analisis sentimen multi-aspek pada ulasan aplikasi kesehatan digital berbahasa Indonesia.

## 👩‍💻 Author

**Jihan Hasna Iftinan**
Information Systems Graduate
Data Analyst | Machine Learning Enthusiast | Business Intelligence
