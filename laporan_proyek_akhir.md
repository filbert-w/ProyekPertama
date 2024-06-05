# Laporan Proyek Machine Learning - Filbert Wijaya

## Project Overview

Pada zaman sekarang, sudah banyak film yang tersedia. Hal tersebut menyebabkan perlunya langkah tambahan untuk mencari dan menyaring film-film yang sangat banyak untuk mencari film yang sesuai. Proyek ini bermaksud mempermudah pencarian film yang diinginkan dengan sistem rekomendasi.

## Business Understanding

Aplikasi layanan streaming dengan rekomendasi film yang sesuai akan meningkatkan kepuasan pengguna dan meningkatkan waktu pemakaian aplikasi.

### Problem Statements

Menjelaskan pernyataan masalah:
- Berdasarkan data film, bagaimana membuat sistem rekomendasi film yang sesuai?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Menghasilkan beberapa rekomendasi film berdasarkan karakteristik film dengan pendekatan content-based filtering.

## Data Understanding
Proyek ini menggunakan dataset Netflix Movies and TV Shows dari Kaggle. [Netlix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows).

|   | show_id |    type |                 title |        director |                                              cast |       country |         date_added | release_year | rating |  duration |                                         listed_in |                                       description |
|--:|--------:|--------:|----------------------:|----------------:|--------------------------------------------------:|--------------:|-------------------:|-------------:|-------:|----------:|--------------------------------------------------:|--------------------------------------------------:|
| 0 |      s1 |   Movie |  Dick Johnson Is Dead | Kirsten Johnson |                                               NaN | United States | September 25, 2021 |         2020 |  PG-13 |    90 min |                                     Documentaries | As her father nears the end of his life, filmm... |
| 1 |      s2 | TV Show |     Blood &amp; Water |             NaN | Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban... |  South Africa | September 24, 2021 |         2021 |  TV-MA | 2 Seasons |   International TV Shows, TV Dramas, TV Mysteries | After crossing paths at a party, a Cape Town t... |
| 2 |      s3 | TV Show |             Ganglands | Julien Leclercq | Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi... |           NaN | September 24, 2021 |         2021 |  TV-MA |  1 Season | Crime TV Shows, International TV Shows, TV Act... | To protect his family from a powerful drug lor... |
| 3 |      s4 | TV Show | Jailbirds New Orleans |             NaN |                                               NaN |           NaN | September 24, 2021 |         2021 |  TV-MA |  1 Season |                            Docuseries, Reality TV | Feuds, flirtations and toilet talk go down amo... |
| 4 |      s5 | TV Show |          Kota Factory |             NaN | Mayur More, Jitendra Kumar, Ranjan Raj, Alam K... |         India | September 24, 2021 |         2021 |  TV-MA | 2 Seasons | International TV Shows, Romantic TV Shows, TV ... | In a city of coaching centers known to train I... |

Variabel-variabel pada Neflix Movies and TV Shows dataset adalah sebagai berikut:

* show_id : merupakan id dari show di Netflix.
* type : merupakan jenis show di Netflix, yaitu **Movie** atau **TV Show**.
* title : merupakan judul show di Netflix.
* director : merupakan sutradara dari show di Netflix.
* cast : merupakan cast dari show di Netflix.
* country : merupakan negara dari show di Netflix.
* date_added : merupakan tanggal ditambahnya show di Netflix.
* release_year : merupakan tahun rilis dari show di Netflix.
* rating : merupakan rating usia dari show di Netflix.
* duration : merupakan durasi show di Netflix.
* listed_in : merupakan genre dari show di Netflix.
* description: merupakan deskripsi dari show di Netflix.

|  #  |       Column | Non-Null Count |  Dtype |
|:---:|-------------:|---------------:|-------:|
|  0  |      show_id |  8807 non-null | object |
|  1  |         type |  8807 non-null | object |
|  2  |        title |  8807 non-null | object |
|  3  |     director |  6173 non-null | object |
|  4  |         cast |  7982 non-null | object |
|  5  |      country |  7976 non-null | object |
|  6  |   date_added |  8797 non-null | object |
|  7  | release_year |  8807 non-null |  int64 |
|  8  |       rating |  8803 non-null | object |
|  9  |     duration |  8804 non-null | object |
|  10 |    listed_in |  8807 non-null | object |
|  11 |  description |  8807 non-null | object |

Jumlah data adalah 8807 data. Terdapat beberapa kolom dengan data null yaitu kolom director, cast, country, date_added, rating, dan duration. Sedangkan show_id, type, title, release_year, listed_in, dan description sudah lengkap ditandai dengan **8807 non-null**.

## Data Preparation
### Penghapusan stopword
Stopword adalah karakter yang tidak menambah makna semantik dalam kata atau kalimat. Pada kasus ini akan dilakukan penghapusan stopword pada kolom listed_in. Hal ini akan mempermudah dilakukannya vektorisasi saat menggunakan metode TF-IDF nantinya.

Sebagian besar value dari kolom listed_in berisi genre yang dipisah tanda koma sehingga digunakan split untuk memisahkan genre tersebut berdasarkan tanda koma, kemudian genre tersebut digabungkan kembali menjadi satu string menggunakan join sehingga siap untuk dilakukan fit dengan TF-IDF.

|    0 |                Documentaries |
|-----:|-----------------------------:|
|   1  |       International TV Shows |
|   2  |               Crime TV Shows |
|   3  |                   Docuseries |
|   4  |       International TV Shows |
|  ... |                              |
| 8802 | Cult Movies                  |
| 8803 | Kids' TV                     |
| 8804 | Comedies                     |
| 8805 | Children &amp; Family Movies |
| 8806 | Dramas                       |

## Modeling
Proyek ini menggunakan TfidfVectorizer untuk melakukan vektorisasi terhadap data di kolom listed_in. Pertama akan dilakukan dipanggil TfidfVectorizer, kemudian melakukan fit pada data di kolom listed_in, kemudian melakukan transform pada data di kolom listed_in untuk mengubahnya menjadi matriks.

|                                            title | music | lgbtq | faith |  reality | stand | mysteries |    anime | talk | thrillers |   dramas | ... | adventure | international | musicals | independent |  fi | sci | documentaries |   korean | sports | teen |
|-------------------------------------------------:|------:|------:|------:|---------:|------:|----------:|---------:|-----:|----------:|---------:|----:|----------:|--------------:|---------:|------------:|----:|----:|--------------:|---------:|-------:|-----:|
| Inuyasha the Movie - L'isola del fuoco scarlatto |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.509902 |  0.0 |  0.000000 | 0.000000 | ... |  0.351247 |      0.196787 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                     Sabotage                     |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.412392 | ... |  0.644179 |      0.000000 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                  Slobby's World                  |   0.0 |   0.0 |   0.0 | 0.897424 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.000000 | ... |  0.000000 |      0.000000 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                    Love Alarm                    |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.000000 | ... |  0.000000 |      0.146923 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.421478 |    0.0 |  0.0 |
|                Everyday I Love You               |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.377451 | ... |  0.000000 |      0.330324 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                Legends of Strength               |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.000000 | ... |  0.000000 |      0.000000 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                       Troy                       |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.000000 | ... |  0.000000 |      0.149503 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|         Money Heist: From Tokyo to Berlin        |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.000000 | ... |  0.000000 |      0.164402 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                   House Arrest                   |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.000000 | 0.000000 | ... |  0.000000 |      0.312817 |      0.0 |    0.612626 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |
|                    Aapla Manus                   |   0.0 |   0.0 |   0.0 | 0.000000 |   0.0 |       0.0 | 0.000000 |  0.0 |  0.754579 | 0.418956 | ... |  0.000000 |      0.366646 |      0.0 |    0.000000 | 0.0 | 0.0 |           0.0 | 0.000000 |    0.0 |  0.0 |

Setelah itu akan digunakan cosine similarity untuk menghitung jarak sudut cosinus antara vektor dalam matriks.

|                                                title | The Devil's Mistress | ... | Henry Danger | Hiroshima: The Real History | Guillermo Vilas: Settling the Score |
|-----------------------------------------------------:|---------------------:|----:|-------------:|----------------------------:|------------------------------------:|
|                Jim Gaffigan: King Baby               |             0.000000 | ... |     0.000000 |                    0.000000 |                            0.000000 |
|                     Centaurworld                     |             0.000000 | ... |     1.000000 |                    0.000000 |                            0.000000 |
|                     Breaking Free                    |             0.371555 | ... |     0.000000 |                    0.447165 |                            0.495791 |
|                   Girl from Nowhere                  |             0.241111 | ... |     0.493435 |                    0.000000 |                            0.048015 |
|                  The Legend of Korra                 |             0.000000 | ... |     0.610035 |                    0.000000 |                            0.000000 |
| Club de Cuervos Presents: The Ballad of Hugo Sánchez |             0.086854 | ... |     0.486401 |                    0.000000 |                            0.039880 |
|                    Ordinary World                    |             0.193646 | ... |     0.130745 |                    0.000000 |                            0.000000 |
|                   Alive and Kicking                  |             0.000000 | ... |     0.000000 |                    1.000000 |                            0.482193 |
|                   High Flying Bird                   |             0.404239 | ... |     0.000000 |                    0.000000 |                            0.746888 |
|                    Shadow and Bone                   |             0.115594 | ... |     0.417743 |                    0.000000 |                            0.000000 |

Setelah itu, model akan dibangun dengan menerima input judul movie, matriks cosinus similarity, kolom data movie, dan k sebagai jumlah rekomendasi. Kemudian model akan memilih k rekomendasi dengan similarity terbesar.

```
def movie_recommendations(show_title, similarity_data=cosine_sim_df, items=data[['show_id', 'show_type', 'title', 'listed_in', 'description']], k=5):
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,show_title].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop show_title agar nama show yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(show_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
```

Model dapat mendapatkan rekomendasi dengan memasukkan judul movie sebagai parameter.

```
movie_recommendations("Inuyasha the Movie - L'isola del fuoco scarlatto")
```

Berikut adalah rekomendasi movie yang mirip dengan Inuyasha the Movie - L'isola del fuoco scarlatto menurut model:

|   |                                             title | show_id | show_type |                                           listed_in |                                       description |
|--:|--------------------------------------------------:|--------:|----------:|----------------------------------------------------:|--------------------------------------------------:|
| 0 | InuYasha: The Movie 2: The Castle Beyond the L... |   s7091 |     Movie | Action &amp; Adventure Anime Features Internatio... | With their biggest foe seemingly defeated, Inu... |
| 1 | InuYasha the Movie 2: The Castle Beyond the Lo... |     s52 |     Movie | Action &amp; Adventure Anime Features Internatio... | With their biggest foe seemingly defeated, Inu... |
| 2 | InuYasha the Movie 3: Swords of an Honorable R... |     s53 |     Movie | Action &amp; Adventure Anime Features Internatio... | The Great Dog Demon beaqueathed one of the Thr... |
| 3 | InuYasha the Movie: Affections Touching Across... |     s55 |     Movie | Action &amp; Adventure Anime Features Internatio... | A powerful demon has been sealed away for 200 ... |
| 4 |      Naruto Shippûden the Movie: The Will of Fire |     s59 |     Movie | Action &amp; Adventure Anime Features Internatio... | When four out of five ninja villages are destr... |

## Evaluation
Proyek ini menggunakan metrik cosine similarity. Cosine similarity adalah metrik untuk melihat kemiripan antara dua objek dengan menghitung sudut cosinus antara dua vektor. Semakin kecil jaraknya maka dua objek tersebut semakin mirip. Berdasarkan hasil prediksi model, model berhasil merekomendasikan movie dengan genre yang serupa.
