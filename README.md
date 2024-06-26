# Laporan Proyek Machine Learning - Filbert Wijaya

## Domain Proyek

Domain dari proyek ini adalah ekonomi dan bisnis. Ketika memiliki data rumah dengan harga rumah, sulit untuk mengetahui karakteristik rumah yang memengaruhi harga rumah dan sulit untuk mengetahui harga rumah secara cepat dari beberapa fitur yang ada. Proyek ini bertujuan untuk mempermudah prediksi harga rumah dengan fokus terhadap fitur yang penting dan prediksi dengan bantuan machine learning. Proyek ini menggunakan <i>regression</i> yang diimplementasikan ke dalam deep learning untuk memprediksi harga rumah dan proyek ini juga melibatkan Exploratory Data Analysis untuk mencari hubungan fitur dari dataset dengan harga rumah.

## Business Understanding

Perusahaan yang bergerak di bidang perumahan dapat mempertimbangkan harga jual rumah dengan mudah dengan bantuan machine learning. Perusahaan juga dapat memperoleh informasi tambahan seperti karakteristik yang memengaruhi harga rumah.

### Problem Statements

- Apakah fitur yang memengaruhi harga rumah?
- Berapakah harga rumah berdasarkan data rumah yang tersedia?

### Goals

- Menemukan fitur yang memengaruhi harga rumah.
- Membuat model machine learning yang dapat memprediksi harga rumah.

## Data Understanding
Proyek ini menggunakan dataset harga rumah dari Kaggle. [Housing Price Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

### Variabel-variabel pada Housing Prices Dataset adalah sebagai berikut:
- price : merupakan harga rumah.
- area : merupakan luas rumah.
- bedrooms: merupakan jumlah kamar tidur dalam rumah. Jumlah kamar tidur berkisar dari 1 hingga 6.
- bathrooms: merupakan jumlah kamar mandi dalam rumah. Jumlah kamar mandi berkisar dari 1 hingga 4.
- stories: merupakan jumlah lantai dalam rumah. Jumlah lantai berkisar dari 1 hingga 4.
- mainroad: merupakan jawaban apakah rumah terhubung ke <i>mainroad</i>. Jawaban adalah yes atau no.
- guestroom: merupakan jawaban apakah rumah memiliki <i>guestroom</i>. Jawaban adalah yes atau no.
- basement: merupakan jawaban apakah rumah memiliki <i>basement</i>. Jawaban adalah yes atau no.
- hotwaterheating: merupakan jawaban apakah rumah memiliki <i>hot water heater</i>. Jawaban adalah yes atau no.
- airconditioning: merupakan jawaban apakah rumah memiliki AC. Jawaban adalah yes atau no.
- parking: merupakan jumlah tempat parkir di rumah. Jumlah tempat parkir berkisar dari 0 hingga 3.
- prefarea: merupakan jawaban apakah rumah berlokasi di tempat yang strategis. Jawaban adalah yes atau no.
- furnishingstatus: merupakan status kelengkapan perabotan rumah. Status kelengkapan terdiri dari furnished yang berarti lengkap, semi-furnished yang berarti cukup, dan unfurnished berarti tidak lengkap.

Berikut adalah informasi tentang kelengkapan data dan tipe data. Semua data lengkap ditandai dengan 545 non-null pada setiap kolom.

| #  | Column           | Non-Null Count | Dtype  |
|----|------------------|----------------|--------|
| 0  | price            | 545 non-null   | int64  |
| 1  | area             | 545 non-null   | int64  |
| 2  | bedrooms         | 545 non-null   | int64  |
| 3  | bathrooms        | 545 non-null   | int64  |
| 4  | stories          | 545 non-null   | int64  |
| 5  | mainroad         | 545 non-null   | object |
| 6  | guestroom        | 545 non-null   | object |
| 7  | basement         | 545 non-null   | object |
| 8  | hotwaterheating  | 545 non-null   | object |
| 9  | airconditioning  | 545 non-null   | object |
| 10 | parking          | 545 non-null   | int64  |
| 11 | prefarea         | 545 non-null   | object |
| 12 | furnishingstatus | 545 non-null   | object |

Berikut adalah informasi statistik untuk dataset rumah:

|       |        price |         area |   bedrooms |  bathrooms |    stories | parking    |
|------:|-------------:|-------------:|-----------:|-----------:|-----------:|------------|
| count | 5.450000e+02 |   545.000000 | 545.000000 | 545.000000 | 545.000000 | 545.000000 |
|  mean | 4.766729e+06 |  5150.541284 |   2.965138 |   1.286239 |   1.805505 |   0.693578 |
|  std  | 1.870440e+06 |  2170.141023 |   0.738064 |   0.502470 |   0.867492 |   0.861586 |
|  min  | 1.750000e+06 |  1650.000000 |   1.000000 |   1.000000 |   1.000000 |   0.000000 |
|  25%  | 3.430000e+06 |  3600.000000 |   2.000000 |   1.000000 |   1.000000 |   0.000000 |
|  50%  | 4.340000e+06 |  4600.000000 |   3.000000 |   1.000000 |   2.000000 |   0.000000 |
|  75%  | 5.740000e+06 |  6360.000000 |   3.000000 |   2.000000 |   2.000000 |   1.000000 |
|  max  | 1.330000e+07 | 16200.000000 |   6.000000 |   4.000000 |   4.000000 |   3.000000 |

Jumlah data adalah sebanyak 545 data.
Kondisi data adalah bersih dan lengkap ditandai dengan non-null pada semua kolom.
Price, area, bedrooms, bathrooms, stories, parking adalah data numerik.
Mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus adalah data kategori.

Berikut adalah persentase untuk masing-masing data kategori:

![Mainroad](https://github.com/filbert-w/ProyekPertama/assets/114629987/58532f14-e0b1-4112-bdcf-a85c86142442)

![Guestroom](https://github.com/filbert-w/ProyekPertama/assets/114629987/0c79d748-7a22-431f-ba74-dc856fdd1686)

![Basement](https://github.com/filbert-w/ProyekPertama/assets/114629987/3e6ee070-fc6f-4115-992c-feb6bacdeb74)

![HotWaterHeating](https://github.com/filbert-w/ProyekPertama/assets/114629987/08519d12-a8f1-4bbd-a7d7-d49a5f72b0cb)

![AirConditioning](https://github.com/filbert-w/ProyekPertama/assets/114629987/58d73a06-4f77-4ec5-a780-d7354c009286)

![PrefArea](https://github.com/filbert-w/ProyekPertama/assets/114629987/fd750143-407b-4878-99c5-d95a07774e6c)

![FurnishingStatus](https://github.com/filbert-w/ProyekPertama/assets/114629987/623918cf-ba38-4342-a482-3209097a12d1)

Berikut adalah histogram dataset:

![Histogram](https://github.com/filbert-w/ProyekPertama/assets/114629987/f4786a2e-31ef-4e31-b56e-35ef6cc05cef)

Berikut adalah hubungan data kategori dengan harga rumah:

![CategoryAndPrice](https://github.com/filbert-w/ProyekPertama/assets/114629987/b9da8ad4-6a65-4a4d-ad20-dbf156704f1f)

Dapat dilihat bahwa fitur kategori memengaruhi harga rumah.

Berikut adalah pairplot dataset:

![PairPlot](https://github.com/filbert-w/ProyekPertama/assets/114629987/cd161321-a061-4fb2-a74e-5fcb7b5cc897)

Berikut adalah correlation matrix untuk fitur numerik:

![CorrelationMatrix](https://github.com/filbert-w/ProyekPertama/assets/114629987/5aa33b1d-cd75-4282-9fa0-5a51549ae1dd)

Dapat dilihat bahwa fitur area dan bathrooms memengaruhi harga rumah.

Berdasarkan pengamatan pada EDA, semua fitur kategori memengaruhi harga rumah sedangkan pada fitur numerik terdapat beberapa fitur yang kurang memengaruhi harga rumah seperti bedrooms, stories, dan parking sehingga dilakukan drop untuk fitur tersebut.

## Data Preparation

### Encoding fitur kategori
Encoding pada fitur kategori adalah teknik mengubah data kategori menjadi bentuk lain (numerik). Kegunaannya adalah agar bisa dijadikan input untuk model. Teknik-teknik yang dilakukan pada encoding pada proyek ini adalah:
- One hot encoding pada kolom mainroad.
- One hot encoding pada kolom guestroom.
- One hot encoding pada kolom basement.
- One hot encoding pada kolom hotwaterheating.
- One hot encoding pada kolom airconditioning.
- One hot encoding pada kolom prefarea.
- One hot encoding pada kolom furnishingstatus.
- Drop pada kolom mainroad.
- Drop pada kolom guestroom.
- Drop pada kolom basement.
- Drop pada kolom hotwaterheating.
- Drop pada kolom airconditioning.
- Drop pada kolom prefarea.
- Drop pada kolom furnishingstatus.

### Pembagian dataset dengan fungsi train_test_split dari library sklearn
Fungsi train_test_split adalah fungsi untuk membagi dataset yang kita miliki menjadi dataset latih dan dataset uji. Kegunaannya adalah agar kita dapat melakukan training pada model dengan dataset latih dan kemudian melakukan evaluasi dengan dataset uji. Teknik-teknik yang dilakukan pada pembagian dataset pada proyek ini adalah:
- Membagi dataset menjadi 80% untuk data latih dan 20% untuk data uji. Dataset dibagi menjadi 80:20 karena jumlah dataset yang tersedia adalah 545 sehingga data training menjadi 436 dan data test menjadi 109 agar jumlah data test tidak terlalu sedikit.

### Standarisasi
Standarisasi adalah proses mengubah distribusi data agar memiliki rata-rata 0 dan standar deviasi 1. Kegunaannya adalah mempercepat proses training. StandardScaler diperlukan untuk mempermudah standarisasi dataset. Hal ini dapat dilakukan dengan meng-import library StandardScaler dari sklearn, membuat instansi StandardScaler, melakukan fit pada dataset, kemudian melakukan standarisasi pada dataset dengan scaler. Teknik-teknik yang dilakukan pada standarisasi adalah:
- Melakukan standarisasi pada dataset latih.
- Melakukan standarisasi pada dataset uji.

### Konversi data ke bentuk tensor
Konversi data ke bentuk tensor adalah proses konversi data ke bentuk tensor agar dapat menjadi input untuk model machine learning. Teknik-teknik yang dilakukan pada konversi data ke bentuk tensor adalah:
- Melakukan konversi data latih ke bentuk tensor.
- Melakukan konversi data uji ke bentuk tensor.

## Modeling

Proyek ini menggunakan deep learning. Model dimulai dengan Dense layer dengan 512 unit yang menerima input dengan shape [None, 17] (17 fitur) dan diikuti relu activation function. Output dari layer pertama akan diteruskan ke layer ke-dua dengan Dropout layer yang melakukan drop pada 20% nilai layer sebelumnya, layer ke-tiga Dense layer 256 unit dengan relu activation function, Dropout layer dengan drop 20%, layer ke-lima dengan Dense layer 128 unit dengan relu activation function, Dropout layer dengan drop 10%, layer ke-tujuh dengan Dense layer 64 unit dengan relu activation function, layer ke-delapan dengan Dense layer 32 unit dengan relu activation function. Kemudian output layer model ini adalah Dense layer 1 unit. Output dari model adalah harga prediksi rumah. Model ini menggunakan optimizer Adam, loss mean squared error, dan metrik mean absolute error.

## Evaluation

Pada kasus regresi ini, metrik yang digunakan adalah mean absolute error. Mean absolute error adalah metrik untuk melihat perbedaan prediksi dan nilai sebenarnya dengan menjumlahkan nilai mutlak dari selisih prediksi dan nilai sebenarnya yang kemudian dirata-ratakan. Berdasarkan hasil evaluasi, mean absolute loss yang diperoleh adalah 944786,5; mendekati mean absolute error yang diperoleh saat training (827071,25). Model berhasil memprediksi sebagian besar harga rumah dengan selisih harga di bawah 1000000.
