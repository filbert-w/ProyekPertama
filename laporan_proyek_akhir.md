# Laporan Proyek Machine Learning - Filbert Wijaya

## Project Overview

Domain dari proyek ini adalah rekomendasi film dengan pendekatan content-based filtering. Saat membuka aplikasi menonton film, jika melihat list film yang tersedia, terdapat sangat banyak film. Hal ini akan sangat menghabiskan waktu untuk melihat satu per satu film dari list film yang jumlahnya terlalu banyak, sehingga menyebabkan perlunya langkah tambahan untuk mencari dan menyaring film-film yang sangat banyak untuk mencari film yang sesuai dengan preferensi. Dengan diterapkannya model content-based filtering dalam menghasilkan rekomendasi film, proyek ini bermaksud mempermudah pencarian film yang diinginkan dengan merekomendasikan film-film serupa yang dengan satu film yang dipilih user. Dataset yang digunakan dalam proyek ini adalah dataset yang berisi film yang ditayangkan di Netflix. Dengan sistem rekomendasi ini, akan dihasilkan beberapa rekomendasi film yang dianggap serupa dengan film yang dipilih user, sehingga mempermudah user menemukan film yang serupa, dan membantu perusahaan perfilman dalam menyajikan film yang sesuai untuk target pasar.

## Business Understanding

Aplikasi menonton film dengan rekomendasi film yang tepat akan meningkatkan kepuasan pengguna karena kemudahan dalam menemukan film yang disukai dan meningkatkan waktu pemakaian aplikasi oleh pengguna yang akan menguntungkan perusahaan.

### Problem Statements

- Berdasarkan satu film, bagaimana merekomendasikan beberapa film yang serupa dari seluruh dataset film?

### Goals

- Membuat model machine learning dengan pendekatan content-based filtering yang dapat menemukan film yang serupa.

## Data Understanding
Proyek ini menggunakan dataset Netflix Movies and TV Shows dari Kaggle. [Netlix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows).

Tabel 1.

|   | show_id |    type |                 title |        director |                                              cast |       country |         date_added | release_year | rating |  duration |                                         listed_in |                                       description |
|--:|--------:|--------:|----------------------:|----------------:|--------------------------------------------------:|--------------:|-------------------:|-------------:|-------:|----------:|--------------------------------------------------:|--------------------------------------------------:|
| 0 |      s1 |   Movie |  Dick Johnson Is Dead | Kirsten Johnson |                                               NaN | United States | September 25, 2021 |         2020 |  PG-13 |    90 min |                                     Documentaries | As her father nears the end of his life, filmm... |
| 1 |      s2 | TV Show |     Blood &amp; Water |             NaN | Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban... |  South Africa | September 24, 2021 |         2021 |  TV-MA | 2 Seasons |   International TV Shows, TV Dramas, TV Mysteries | After crossing paths at a party, a Cape Town t... |
| 2 |      s3 | TV Show |             Ganglands | Julien Leclercq | Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi... |           NaN | September 24, 2021 |         2021 |  TV-MA |  1 Season | Crime TV Shows, International TV Shows, TV Act... | To protect his family from a powerful drug lor... |
| 3 |      s4 | TV Show | Jailbirds New Orleans |             NaN |                                               NaN |           NaN | September 24, 2021 |         2021 |  TV-MA |  1 Season |                            Docuseries, Reality TV | Feuds, flirtations and toilet talk go down amo... |
| 4 |      s5 | TV Show |          Kota Factory |             NaN | Mayur More, Jitendra Kumar, Ranjan Raj, Alam K... |         India | September 24, 2021 |         2021 |  TV-MA | 2 Seasons | International TV Shows, Romantic TV Shows, TV ... | In a city of coaching centers known to train I... |

Variabel-variabel pada Neflix Movies and TV Shows dataset adalah sebagai berikut:

* show_id : merupakan id dari show di Netflix.
* type : merupakan jenis show di Netflix.
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

Tabel 2.

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

Data unik untuk masing-masing kolom adalah:
* Jumlah show dalam dataset:  8807
* Jumlah tipe show dalam dataset:  2
* Jumlah judul show dalam dataset:  8807
* Jumlah sutradara dalam dataset:  4529
* Jumlah cast dalam dataset:  7693
* Jumlah negara dalam dataset:  749
* Jumlah tanggal dalam dataset:  1768
* Jumlah tahun rilis dalam dataset:  74
* Jumlah rating dalam dataset:  18
* Jumlah durasi dalam dataset:  221
* Jumlah genre dalam dataset:  514
* Jumlah deskripsi dalam dataset:  8775

Tipe show dalam dataset adalah **Movie** dan **TV Show**.

Tahun rilis show dalam dataset adalah 1925, 1942, 1943, 1944, 1945, 1946, 1947, 1954, 1955, 1956, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, dan 2021.

Rating usia dalam dataset adalah PG-13, TV-MA, PG, TV-14, TV-PG, TV-Y, TV-Y7, R, TV-G, G, NC-17, 74 min, 84 min, 66 min, NR, nan, TV-Y7-FV, dan UR.

Pada kolom rating, dapat dilihat terdapat data yang tidak sesuai dengan rating usia.

Durasi dalam dataset terdiri dari yang paling pendek adalah 3 menit dan paling panjang adalah 312 menit, dan season paling sedikit adalah 1 season dan season paling banyak adalah 17 season.

Genre dalam dataset yaitu Documentaries, International TV Shows, TV Dramas, TV Mysteries, Crime TV Shows, TV Action & Adventure, Docuseries, Reality TV, Romantic TV Shows, TV Comedies, TV Horror, Children & Family Movies, Dramas, Independent Movies, International Movies, British TV Shows, Comedies, Spanish-Language TV Shows, Thrillers, Romantic Movies, Music & Musicals, Horror Movies, Sci-Fi & Fantasy, TV Thrillers, Kids' TV, Action & Adventure, TV Sci-Fi & Fantasy, Classic Movies, Anime Features, Sports Movies, Anime Series, Korean TV Shows, Science & Nature TV, Teen TV Shows, Cult Movies, TV Shows, Faith & Spirituality, LGBTQ Movies, Stand-Up Comedy, Movies, Stand-Up Comedy & Talk Shows, Classic & Cult TV.

Genre yang paling banyak dalam dataset adalah Dramas dan Comedies dengan masing-masing berjumlah 79.

## Data Preparation
### Penanganan missing value
Missing value adalah nilai yang tidak terisi pada kolom dalam dataset. Penanganan missing value biasanya dilakukan dengan drop atau diisi dengan rata-rata. Penanganan ini penting agar data yang digunakan dalam pembangunan model bersifat lengkap. Kolom director, cast, country, date_added, rating, dan duration memiliki missing value. Pada proyek ini, kolom yang akan digunakan adalah show_id, type, title, release_year listed_in, description yang tidak memiliki data null.

Tabel 3.

|              |      |
|--------------|------|
| show_id      | 0    |
| type         | 0    |
| title        | 0    |
| director     | 2634 |
| cast         | 825  |
| country      | 831  |
| date_added   | 10   |
| release_year | 0    |
| rating       | 4    |
| duration     | 3    |
| listed_in    | 0    |
| description  | 0    |

### Penghapusan stopword
Stopword adalah karakter yang tidak menambah makna semantik dalam kata atau kalimat. Pada kasus ini akan dilakukan penghapusan stopword pada kolom listed_in. Hal ini akan mempermudah dilakukannya vektorisasi saat menggunakan metode TF-IDF nantinya.

Sebagian besar value dari kolom listed_in berisi genre yang dipisah tanda koma sehingga digunakan replace untuk mengubah tanda koma menjadi spasi, tetapi sebelum itu spasi pada kolom listed_in diubah menjadi underscore terlebih dahulu agar masing-masing genre terpisah dengan baik. Setelah penghapusan tanda koma, kolom listed_in siap untuk dilakukan fit dengan TF-IDF.

Tabel 4.

|      |                                                   |
|------|---------------------------------------------------|
| 0    | Documentaries                                     |
| 1    | International_TV_Shows TV_Dramas TV_Mysteries     |
| 2    | Crime_TV_Shows International_TV_Shows TV_Actio... |
| 3    | Docuseries Reality_TV                             |
| 4    | International_TV_Shows Romantic_TV_Shows TV_Co... |
| ...  | Cult_Movies Dramas Thrillers                      |
| 8802 | Cult_Movies Dramas Thrillers                      |
| 8803 | Kids'_TV Korean_TV_Shows TV_Comedies              |
| 8804 | Comedies Horror_Movies                            |
| 8805 | Children_&amp;_Family_Movies Comedies             |
| 8806 | Dramas International_Movies Music_&amp;_Musicals  |

### Drop pada data duplikat
Data duplikat adalah data yang sama persis dan berulang. Hal ini bersifat tidak efisien sehingga data duplikat perlu dihapus. Pada kasus ini tidak ditemukan data duplikat, dapat dilihat bahwa data masih tetap berjumlah 8807.

Tabel 5.

|      | show_id |    type |                 title | release_year |                                         listed_in |                                       description |
|------|--------:|--------:|----------------------:|-------------:|--------------------------------------------------:|--------------------------------------------------:|
|   0  |      s1 |   Movie |  Dick Johnson Is Dead |         2020 |                                     Documentaries | As her father nears the end of his life, filmm... |
|   1  |      s2 | TV Show |     Blood &amp; Water |         2021 |     International_TV_Shows TV_Dramas TV_Mysteries | After crossing paths at a party, a Cape Town t... |
|   2  |      s3 | TV Show |             Ganglands |         2021 | Crime_TV_Shows International_TV_Shows TV_Actio... | To protect his family from a powerful drug lor... |
|   3  |      s4 | TV Show | Jailbirds New Orleans |         2021 |                             Docuseries Reality_TV | Feuds, flirtations and toilet talk go down amo... |
|   4  |      s5 | TV Show |          Kota Factory |         2021 | International_TV_Shows Romantic_TV_Shows TV_Co... | In a city of coaching centers known to train I... |
|  ... |     ... |     ... |                   ... |          ... |                                               ... |                                               ... |
| 8802 |   s8803 |   Movie |                Zodiac |         2007 |                      Cult_Movies Dramas Thrillers | A political cartoonist, a crime reporter and a... |
| 8803 |   s8804 | TV Show |           Zombie Dumb |         2018 |              Kids'_TV Korean_TV_Shows TV_Comedies | While living alone in a spooky town, a young g... |
| 8804 |   s8805 |   Movie |            Zombieland |         2009 |                            Comedies Horror_Movies | Looking to survive in a world taken over by zo... |
| 8805 |   s8806 |   Movie |                  Zoom |         2006 |             Children_&amp;_Family_Movies Comedies | Dragged from civilian life, a former superhero... |
| 8806 |   s8807 |   Movie |                Zubaan |         2015 |  Dramas International_Movies Music_&amp;_Musicals | A scrappy but poor boy worms his way into a ty... |

## Modeling
Proyek ini menggunakan TfidfVectorizer untuk melakukan vektorisasi terhadap data di kolom listed_in. Pertama akan dilakukan dipanggil TfidfVectorizer, kemudian melakukan fit pada data di kolom listed_in, kemudian melakukan transform pada data di kolom listed_in untuk mengubahnya menjadi matriks.

Tabel 6.

|                                                title | docuseries |  action_ | up_comedy | lgbtq_movies | _family_movies | up_comedy_ | anime_features | _nature_tv | tv_action_ | comedies | ... | crime_tv_shows | sci | international_tv_shows | independent_movies |   tv_sci |     fi_ | faith_ | romantic_movies | spanish | _cult_tv |
|-----------------------------------------------------:|-----------:|---------:|----------:|-------------:|---------------:|-----------:|---------------:|-----------:|-----------:|---------:|----:|---------------:|----:|-----------------------:|-------------------:|---------:|--------:|-------:|----------------:|--------:|---------:|
|                     Pulp Fiction                     |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
|                     Wu Assassins                     |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.487893 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.555583 | 0.42258 |    0.0 |        0.000000 |     0.0 |      0.0 |
|                        Audible                       |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
|        Little Baby Bum: Nursery Rhyme Friends        |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
|                      Be Somebody                     |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.524701 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.721712 |     0.0 |      0.0 |
|       The Most Assassinated Woman in the World       |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
|         Zipi &amp; Zape y la Isla del Capitan        |        0.0 | 0.000000 |       0.0 |          0.0 |       0.627428 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.461161 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
|                        Unroyal                       |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
| Minimalism: A Documentary About the Important Things |        0.0 | 0.000000 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |
|                     Patriot Games                    |        0.0 | 0.726319 |       0.0 |          0.0 |       0.000000 |        0.0 |            0.0 |        0.0 |   0.000000 | 0.000000 | ... |            0.0 | 0.0 |                    0.0 |                0.0 | 0.000000 | 0.00000 |    0.0 |        0.000000 |     0.0 |      0.0 |

Setelah itu akan digunakan cosine similarity untuk menghitung jarak sudut cosinus antara vektor dalam matriks.

Tabel 7.

|                               title | Bulbul Can Sing | Ice Fantasy | Where the Money Is | Errementari: The Blacksmith and the Devil | Colin Quinn: The New York Story |
|------------------------------------:|----------------:|------------:|-------------------:|------------------------------------------:|--------------------------------:|
|               Puerta 7              |           0.000 |    0.092207 |           0.000000 |                                       0.0 |                        0.000000 |
|               Spice Up              |           0.000 |    0.127848 |           0.000000 |                                       0.0 |                        0.000000 |
|     Ari Shaffir: Double Negative    |           0.000 |    0.000000 |           0.000000 |                                       0.0 |                        0.299901 |
|       Chronically Metropolitan      |           0.746 |    0.000000 |           0.433418 |                                       0.0 |                        0.000000 |
|             My Holo Love            |           0.000 |    0.109378 |           0.000000 |                                       0.0 |                        0.000000 |
|              Good Girls             |           0.000 |    0.000000 |           0.000000 |                                       0.0 |                        0.000000 |
|               42 Grams              |           0.000 |    0.000000 |           0.000000 |                                       0.0 |                        0.000000 |
| 100 Things to do Before High School |           0.000 |    0.000000 |           0.000000 |                                       0.0 |                        0.000000 |
|          The Legend of 420          |           0.000 |    0.000000 |           0.000000 |                                       0.0 |                        0.000000 |
|           Surviving Death           |           0.000 |    0.000000 |           0.000000 |                                       0.0 |                        0.000000 |

Setelah itu, model akan dibangun dengan menerima input judul movie, matriks cosinus similarity, kolom data movie, dan k sebagai jumlah rekomendasi. Kemudian model akan memilih k rekomendasi dengan similarity terbesar.

Model dapat mendapatkan rekomendasi dengan memasukkan judul movie sebagai parameter.

Tabel 8. Contoh film untuk menjadi input model.

|      | show_id |  type |                                            title | release_year |                                             listed_in |                                       description |
|------|--------:|------:|-------------------------------------------------:|-------------:|------------------------------------------------------:|--------------------------------------------------:|
| 7088 |   s7089 | Movie | Inuyasha the Movie - L'isola del fuoco scarlatto |         2004 | Action_&amp;_Adventure Anime_Features Internationa... | Ai, a young half-demon who has escaped from Ho... |

Contohnya jika model menerima input Inuyasha the Movie - L'isola del fuoco scarlatto, maka rekomendasi menurut model yaitu:

Tabel 9. Rekomendasi film menurut model.

|   |                                             title | show_id |  type | release_year |                                             listed_in |                                       description |
|---|--------------------------------------------------:|--------:|------:|-------------:|------------------------------------------------------:|--------------------------------------------------:|
| 0 | Inuyasha the Movie - La spada del dominatore d... |   s7090 | Movie |         2003 | Action_&amp;_Adventure Anime_Features Internationa... | The Great Dog Demon beaqueathed one of the Thr... |
| 1 |   InuYasha the Movie 4: Fire on the Mystic Island |     s54 | Movie |         2004 | Action_&amp;_Adventure Anime_Features Internationa... | Ai, a young half-demon who has escaped from Ho... |
| 2 | InuYasha the Movie 3: Swords of an Honorable R... |     s53 | Movie |         2003 | Action_&amp;_Adventure Anime_Features Internationa... | The Great Dog Demon beaqueathed one of the Thr... |
| 3 | InuYasha the Movie: Affections Touching Across... |     s55 | Movie |         2001 | Action_&amp;_Adventure Anime_Features Internationa... | A powerful demon has been sealed away for 200 ... |
| 4 |          Naruto Shippuden the Movie: Blood Prison |     s57 | Movie |         2011 | Action_&amp;_Adventure Anime_Features Internationa... | Mistakenly accused of an attack on the Fourth ... |

## Evaluation
Proyek ini menggunakan metrik precision. Precision mengukur proporsi antara genre dari film rekomendasi yang sama dengan genre dari film input dengan jumlah rekomendasi. Misal untuk hasil prediksi pada Tabel 9, dapat dilihat bahwa semua rekomendasi memiliki genre yang sama dengan input model, sehingga memiliki presisi 5/5 atau 100%. Model berhasil memberikan rekomendasi film yang serupa berdasarkan satu film dengan akurasi yang tinggi.
