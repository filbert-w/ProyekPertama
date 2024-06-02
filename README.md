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

Jumlah data adalah sebanyak 545 data.
Kondisi data adalah bersih dan lengkap ditandai dengan non-null pada semua kolom.
Price, area, bedrooms, bathrooms, stories, parking adalah data numerik.
Mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus adalah data kategori.

![CorrelationMatrix](https://github.com/filbert-w/ProyekPertamaMachineLearningTerapan/blob/main/correlation_matrix.png)

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
- Membagi dataset menjadi 80% untuk data latih dan 20% untuk data uji.

### Standarisasi
Standarisasi adalah proses mengubah distribusi data agar memiliki rata-rata 0 dan standar deviasi 1. Kegunaannya adalah mempercepat proses training. Teknik-teknik yang dilakukan pada standarisasi adalah:
- Melakukan standarisasi pada dataset latih.
- Melakukan standarisasi pada dataset uji.

### Konversi data ke bentuk tensor
Konversi data ke bentuk tensor adalah proses konversi data ke bentuk tensor agar dapat menjadi input untuk model machine learning. Teknik-teknik yang dilakukan pada konversi data ke bentuk tensor adalah:
- Melakukan konversi data latih ke bentuk tensor.
- Melakukan konversi data uji ke bentuk tensor.

## Modeling

Proyek ini menggunakan deep learning. Model dimulai dengan Dense layer dengan 512 unit yang menerima input dengan shape [None, 17] (17 fitur) dan diikuti relu activation function. Output dari layer pertama akan diteruskan ke layer ke-dua dengan Dropout layer yang melakukan drop pada 20% nilai layer sebelumnya, layer ke-tiga Dense layer 256 unit dengan relu activation function, Dropout layer dengan drop 20%, layer ke-lima dengan Dense layer 128 unit dengan relu activation function, Dropout layer dengan drop 10%, layer ke-tujuh dengan Dense layer 64 unit dengan relu activation function, layer ke-delapan dengan Dense layer 32 unit dengan relu activation function. Kemudian output layer model ini adalah Dense layer 1 unit. Output dari model adalah harga prediksi rumah. Model ini menggunakan optimizer Adam, loss mean squared error, dan metrik mean absolute error.

## Evaluation

Pada kasus regresi ini, metrik yang digunakan adalah mean absolute error. Mean absolute error adalah metrik untuk melihat perbedaan prediksi dan nilai sebenarnya dengan menjumlahkan nilai mutlak dari selisih prediksi dan nilai sebenarnya yang kemudian dirata-ratakan. Berdasarkan hasil evaluasi, mean absolute loss yang diperoleh adalah 9,10499095916748. Hasil prediksi cukup mendekati harga rumah sebenarnya.
