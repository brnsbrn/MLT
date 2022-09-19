# Laporan Sistem Rekomendasi Buku Muhammad Sabran

## Project Overview
Buku merupakan salah satu benda yang tak bisa lepas dari hidup manusia. Buku sangat memiliki manfaat yang besar bagi kita, salah satunya dapat menambah wawasan serta pengetahuan kita, makanya sarana pembelajaran kita banyak melalui buku. Selain untuk belajar, buku juga dapat berfungsi sebagai sarana hiburan, contohnya seperti novel dan cerpen. Berdasarkan banyaknya tipe buku tadi, tentunya tiap orang memiliki selera dan ketertarikan mereka sendiri terhadap buku yang mereka suka. Untuk itu melalui proyek ini, dibuatlah sistem rekomendasi buku berdasarkan minat buku dari pembaca. Adapun referensi penelitian dengan topik yang sama dapat dilihat melalui link berikut [SISTEM REKOMENDASI: BUKU ONLINE DENGAN METODE COLLABORATIVE FILTERING](https://ejournal.akprind.ac.id/index.php/technoscientia/article/view/612).

Melihat pentingnya dampak buku bagi kehidupan kita, kita perlu banyak membaca buku. Ketika kita membaca buku, kita pasti memiliki ketertarikan kepada satu atau beberapa bidang. Dikarenakan banyaknya buku yang telah dan akan terbit, kita membutuhkan sistem rekomendasi yang akan menyaring buku - buku sesuai dengan selera dan ketertarikan kita. Dengan adanya sistem rekomendasi ini, kita tidak perlu lama - lama dalam mencari buku sesuai ketertarikan kita.

Kita akan menyelesaikan masalah ini dengan content filtered based yang berasal dari rating para pengguna dan content filtered based recommendation system yang berasal dari penulis buku yang sama. 

## Business Understanding
Tentunya tiap orang akan memiliki selera, kertertarikan, dan kesukaannya masing-masing tak terkecuali dalam membaca buku. Seseorang mungkin saja tertarik dengan genre buku, ataupun penulisnya, serta covernya. Untuk itu dengan adanya sistem rekomendasi buku ini, dapat memudahkan pembaca dalam menemukan rekomendasi buku-buku berdasarkan selera dan kesukaan mereka.

### Problem Statement
- Bagaimana cara mendapatkan rekomendasi buku berdasarkan penulisnya?
- Bagaimana cara membuat sistem rekomendasi buku berdasarkan rating tertinggi?

### Goals
- Mengetahui cara mendapatkan rekomendasi buku berdasarkan penulisnya.
- Mengetahui cara membuat sistem rekomendasi buku berdasarkan rating tertinggi.

### Solution Approach
Dalam melaksanakan proyek ini, saya menggunakan 2 algoritma Machine Learning yaitu Content Based Filtering dan Collaborative Based Filtering

**Content Based Filtering**
Content Based Filtering adalah sistem rekomendasi yang merekomendasikan item sesuai dengan item yang disukai oleh pengguna di masa lalu.

Content-based filtering mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai pengguna. Algoritma ini bekerja dengan menyarankan item serupa yang pernah disukai di masa lalu atau sedang dilihat di masa kini kepada pengguna. Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.

Untuk membuat profil pengguna, dua informasi ini penting bagi sistem dengan pendekatan content-based filtering: 

- Model preferensi pengguna.
- Riwayat interaksi pengguna dengan sistem rekomendasi. 

**Collaborative Filtering**
Collaborative filtering bergantung pada pendapat komunitas pengguna. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten.
Collaborative filtering dibagi lagi menjadi dua kategori, yaitu: model based (metode berbasis model machine learning) dan memory based (metode berbasis memori). 

Pada Content Based Filtering, saya akan membuat sistem rekomendasi berdasarkan penulis dari buku yang telah dibaca sebelumnya. Sedangkan untuk Collaborative Filtering saya akan membuat sistem rekomendasi berdasarkan buku-buku yang mendapatkan rating tinggi dari pengguna

## Data Understanding
Adapun dataset yang saya gunakan dapat dilihat pada [tautan](https://www.kaggle.com/arashnic/book-recommendation-dataset) berikut. Untuk data yang akan digunakan ialah Books.csv dan Ratings.csv. Adapun untuk detail penjelasan tiap fiturnya sebagai berikut.

**Book.csv** merupakan dataset yang menampung detail buku mulai dari judul, cover, penulis dan tahun terbit. Adapun fitur yang ada pada dataset ini yaitu.
- ISBN  : Menunjukkan ISBN (International Standard Book Number) dari buku
- Book-Title   :   Menunjukkan judul dari buku
- Book-Author  : Menunjukkan penulis dari buku
- Year-of-Publication   : Menunjukkan tahun terbit buku
- Publisher :    Menunjukkan penerbit dari buku
- Image-URL-S   :  Menunjukkan tautan untuk gambar sampul berukuran kecil
- Image-URL-M   :  Menunjukkan tautan untuk gambar sampul berukuran sedang
- Image-ULR-L   :   Menunjukkan tautan untuk gambar sampul berukuran besar

**Ratings.csv** merupakan dataset yang menampung hasil penilaian rating pengguna terhadap buku-buku. Adapun fitur yang ada pada dataset ini yaitu.
- User-ID: Menunjukkan ID dari pengguna
- ISBN  : Menunjukkan ISBN (International Standard Book Number) dari buku
- Book-Rating   : Menunjukkan rating penilaian dari pengguna


### Univariate EDA

**Book**

![image](https://user-images.githubusercontent.com/113587270/190958296-8cf455ee-8064-4357-89e1-a065d3a0b846.png)

Berdasarkan data di atas, Book memiliki 271360 data entri, data yang sangat banyak tentunya. Selain itu pada Book terdapat 8 variabel antara lai ISBN, judul, penulis, tahun terbit, penerbit, link cover kecil, besar, dan sedang.

![image](https://user-images.githubusercontent.com/113587270/190958919-6de23488-835f-4cae-92db-e3dab59773f3.png)

Terdapat 9553 judul buku dengan tahun terbit yang berbeda, yang paling lama diterbitkan pada tahun 1937 dan terbaru diterbitkan pada tahun 2002

**Rating**

![image](https://user-images.githubusercontent.com/113587270/190959306-376f1d60-9167-4091-a56c-ab72f422848b.png)

Pada dataset Rating, terdapat 1149780 data entri yang tentunya sangat banyak. Serta terdapat 3 variabel yaitu UserID (ID dari pengguna), ISBN, dan Book-Rating (Rating yang diberikan pengguna terhadap buku).

![image](https://user-images.githubusercontent.com/113587270/190959877-74fd930b-01b4-4c65-85f3-7c1bdf9c73d0.png)

Dapat dilihat terdapat 679 data dengan UserID yang berbeda dengan Rating dari rentang 1-10.

## Data Preparation
Dikarenakan data yang sangat banyak, maka saya akan mengambil beberapa data saja. Pada data Book saya mengambil 10000 row, sedangkan pada data Rating saya mengambil 5000 row.

![image](https://user-images.githubusercontent.com/113587270/190960371-f562538d-41a1-419d-b858-9df876e77413.png)

**Content Based Filtering**
Pada content Based Filtering, data preparation yang diperlukan ada 4, yaitu:
- **Drop kolom Na**

    ![image](https://user-images.githubusercontent.com/113587270/190960871-7c08f312-818b-408a-9c06-5d1add44891c.png)
    
    Drop semua kolom yang mengandung Na/Null pada data book dan rating.
- **Drop row duplikat**

  ![image](https://user-images.githubusercontent.com/113587270/190961457-f5b7c3d6-d40d-4149-ad32-2e675a74aaae.png)
  
  Drop baris yang menduplikasi baris yang lain agar data tidak tumpang tindih dan tidak berulang-ulang.

- **Mengubah Dataframe menjadi List**
  
  ![image](https://user-images.githubusercontent.com/113587270/190961801-561a0ee7-97b7-45a0-9e77-c4b4faf7e7ab.png)

  Kita mengubah dataframe book menjadi sebuah list. Dalam hal ini, kita menggunakan fungsi tolist() dari library numpy.
  
- **Membuat Dictionary**

  ![image](https://user-images.githubusercontent.com/113587270/190962440-ca889371-3857-4318-a28b-be244ba4c49f.png)
  
  Buat dictionary untuk menentukan key value dari list yang telah kita buat.

**Collaborative Based Filtering**
- **Melakukan encoding**
  Lakukan encoding pada kolom user_id dan ISBN agar menjadi berurutan dan dalam bentuk integer
  
  ![image](https://user-images.githubusercontent.com/113587270/190962997-18e403f8-7d0b-4e10-91b7-a2b379947345.png)
  ![image](https://user-images.githubusercontent.com/113587270/190963633-4c73d71b-dbfb-4b5b-b43e-30b57934c54d.png)

- **Mapping**
  
  ![image](https://user-images.githubusercontent.com/113587270/190963970-4e74d9e0-1cc2-4f9c-bafd-b139583c804c.png)

  Berikutnya, petakan userID dan ISBN ke dataframe yang berkaitan.

- **Cek Data**
  Cek beberapa hal dalam data seperti jumlah user, jumlah resto, dan mengubah nilai rating menjadi float.
  
  ![image](https://user-images.githubusercontent.com/113587270/190964257-c5b4a316-e81a-4522-bac8-20a9957417e9.png)


- **Train-Test-Split**
  Pertama acak terlebih dahulu dataset.
  
  ![image](https://user-images.githubusercontent.com/113587270/190964758-173b0e0a-bba0-4966-ac4f-fa54dbca7c6e.png)
  
  Selanjutnya, bagi data train dan test dengan komposisi 70:30. Namun sebelumnya, kita perlu memetakan (mapping) data user dan buku menjadi satu value terlebih dahulu. 
  
  ![image](https://user-images.githubusercontent.com/113587270/190965030-416167ef-6521-4f37-9757-edcdfb9cd3f6.png)

## Modelling
### Content Based Filtering
Modelling menggunakan fungsi tfidfvectorizer() dari library sklearn.

![image](https://user-images.githubusercontent.com/113587270/190966464-152487a8-4a20-408a-8860-c93cc0e3faef.png)

Lakukan fit transformasi dalam bentuk matriks.

![image](https://user-images.githubusercontent.com/113587270/190967078-9a6cf16e-3b41-49b2-9988-0689814f0409.png)

Untuk menghasilkan vektor tf-idf dalam bentuk matriks, gunakan fungsi todense()

![image](https://user-images.githubusercontent.com/113587270/190967219-ca3695cd-c28b-4058-8b58-c2936315b8a8.png)

Selanjutnya, mari kita lihat matriks tf-idf untuk beberapa buku dan penulis. 

![image](https://user-images.githubusercontent.com/113587270/190968170-2b2ac1d7-94a2-4ce5-b0c9-64b0403078f8.png)

Sayangnya, berdasarkan matriks di atas tidak ada yang saling berkorelasi dikarenakan penulisnya terbatas.

Selanjutnya, itung derajat kesamaan (similarity degree) antar buku dengan teknik cosine similarity. Di sini, kita menggunakan fungsi cosine_similarity dari library sklearn

![image](https://user-images.githubusercontent.com/113587270/190968955-9189e46d-f532-41ad-b04f-7ee44b248a4d.png)

Buat dataframe cosine dengan kolom dan baris merupakan judul buku.

![image](https://user-images.githubusercontent.com/113587270/190969223-c4867395-8ab0-4a46-a092-377be0ace305.png)

Buat fungsi untuk menampilkan 5 rekomendasi teratas.

![image](https://user-images.githubusercontent.com/113587270/190969720-30a17b5e-baaa-4bd1-96a1-3b574cdf9b42.png)

Misal item buku yang sudah dibaca adalah 'The Star Rover', maka hasil rekomendasi berdasarkan buku tersebut adalah.

![image](https://user-images.githubusercontent.com/113587270/190970239-3736c3d9-f391-41bb-8df6-cd3457563639.png)


![image](https://user-images.githubusercontent.com/113587270/190969973-f8bb3674-16b6-4a9b-8881-a1097f7ff5a0.png)



### Collaborative Filtering
Di sini, saya membuat model dengan class RecommenderNet dengan keras Model class.

![image](https://user-images.githubusercontent.com/113587270/190971528-1a8c43cf-4562-40bd-99a0-3ec0eb94e50f.png)

Lakukan kompilasi model. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 

![image](https://user-images.githubusercontent.com/113587270/190972404-26f9c9d0-ee07-492f-8779-3defb982d539.png)

Kemudian latih model, hingga RMSE nya turun.

![image](https://user-images.githubusercontent.com/113587270/190972655-191630ce-8d0a-4ba0-b0a9-ca7d4452e121.png)

Didapat nilai akhir dari RMSE pada data train sebesar 0.2322 dan nilai RMSE pada data test sebesar 0.3484.

Adapun hasil rekomendasinya sebagai berikut.
![image](https://user-images.githubusercontent.com/113587270/190973039-636d2df5-1910-47a8-9f50-3480ff66f38c.png)



## Evaluation

**Content Based Filtering**
Pada Content Based Filtering, saya menggunakan akurasi sebagai metrik evaluasi.
Akurasi pada Content Based Filtering didapat dari:
Jumlah rekomendasi buku yang sesuai dengan penulis / Jumlah buku yang direkomendasikan.
Pada hasil rekomendasi Content Based, dari 5 buku yang direkomendasikan hanya 3 buku yang memiliki penulis yang sesuai dengan buku yang telah dibaca user.

![Accuracy](https://raw.githubusercontent.com/farelarden/Dicoding-SIB/main/27.JPG)

Kelebihan metriks akurasi ini yaitu mudah digunakan karena hanya cukup membagi jumlah data rekomendasi yang benar dengan seluruh data rekomendasi, sedangkan untuk kekurangannya karena metriks ini sangat sederhana sehingga tidak dapat digunakan untuk aspek yang lebih kompleks

**Collaborative Based Filtering**
Pada Collaborative Based Filtering, metriks yang saya gunakan adalah RMSE. Root Mean Square Error (RMSE) adalah  metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar.

Berikut adalah rumus RMSE:

![image](https://user-images.githubusercontent.com/113587270/190975143-1962abd6-0ce8-4948-a334-4f8257dcbd2d.png)

Keterangan:
At = Nilai data Aktual
Ft = Nilai hasil peramalan
N= banyaknya data
∑ = Summation (Jumlahkan keseluruhan  nilai)

Kelebihan RMSE ialah memiliki nilai eror yang lebih kecil dibandingkan dengan MSE yang membuat model memiliki akurasi yang lebih baik, namun kekurangannya sering menyebabkan model mengalami overfitting atau underfitting.

Berikut hasil nilai RMSE pada Collaborative Filtering.

![image](https://user-images.githubusercontent.com/113587270/190972655-191630ce-8d0a-4ba0-b0a9-ca7d4452e121.png)

Bisa dilihat bahwa nilai error baik di data train dan data test mengalami penurunan yang menandakan model yang kita buat cukup bagus.

