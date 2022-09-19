# Laporan Sistem Rekomendasi Buku Muhammad Sabran

## Project Overview
Buku merupakan salah satu benda yang tak bisa lepas dari hidup manusia. Buku sangat memiliki manfaat yang besar bagi kita, salah satunya dapat menambah wawasan serta pengetahuan kita, Buku merupakan informasi segala    kebutuhan     yang    diperlukan, dimulai dari iptek, seni budaya, ekonomi, politik,  sosial  dan  pertahanan  keamanan dan   lain-lain.   Upaya   membaca   buku membuka wawasan dunia intelek sehingga  dapat  mengubah  masa  depan serta   mencerdaskan   akal,   pikiran   dan iman [1] . Selain untuk belajar, buku juga dapat berfungsi sebagai sarana hiburan, contohnya seperti novel dan cerpen. Berdasarkan banyaknya tipe buku tadi, tentunya tiap orang memiliki selera dan ketertarikan mereka sendiri terhadap buku yang mereka suka. Untuk itu melalui proyek ini, dibuatlah sistem rekomendasi buku berdasarkan minat buku dari pembaca. 

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

![image](https://user-images.githubusercontent.com/113587270/190984426-84603566-50e1-4bcd-a4af-3da75294aadc.png)

Terdapat 242135 judul buku dengan tahun terbit yang berbeda serta beberapa nama penulisnya.

**Rating**

![image](https://user-images.githubusercontent.com/113587270/190959306-376f1d60-9167-4091-a56c-ab72f422848b.png)

Pada dataset Rating, terdapat 1149780 data entri yang tentunya sangat banyak. Serta terdapat 3 variabel yaitu UserID (ID dari pengguna), ISBN, dan Book-Rating (Rating yang diberikan pengguna terhadap buku).

![image](https://user-images.githubusercontent.com/113587270/190959877-74fd930b-01b4-4c65-85f3-7c1bdf9c73d0.png)

Dapat dilihat terdapat 679 data dengan UserID yang berbeda dengan Rating dari rentang 0-10.

**Visualisasi Data**

![image](https://user-images.githubusercontent.com/113587270/190987464-3ce1df0f-ef76-4d1f-a774-4e62e4313837.png)

Bisa dilihat berdasarkan grafik di atas, banyak pengguna yang memberikan rating 0.

![image](https://user-images.githubusercontent.com/113587270/190988052-046bcae4-8826-4ce0-84c1-73c7d7c23530.png)

Berdasarkan grafik di atas, mayoritas buku diterbitkan di tahun 2002.

## Data Preparation
Dikarenakan data yang sangat banyak, maka saya akan mengambil beberapa data saja. Pada data Book saya mengambil 10000 row, sedangkan pada data Rating saya mengambil 5000 row.

![image](https://user-images.githubusercontent.com/113587270/190960371-f562538d-41a1-419d-b858-9df876e77413.png)

**Content Based Filtering**
Pada content Based Filtering, data preparation yang diperlukan ada 4, yaitu:
- **Drop kolom Na** dengan menggunakan method dropna() agar tidak ada kolom pada book dan rating yang memiliki nilai Na/Null.
- **Drop row duplikat** agar tidak ada row yang sama/duplikat sehingga dapat menyebabkan terjadinya tumpang tindih data dan data yang berulang dengan nilai yang sama.
- **Mengubah Dataframe Book menjadi List**, selanjutnya koversikan beberapa fitur di dataframe book menjadi sebuah list. Disini fitur ISBN, title, author, serta year saya konversi menjadi sebuah list string.

- **Membuat Dictionary** dengan menambahkan key value untuk memanggil 4 fitur tadi yang telah diubah menjadi list.
  

**Collaborative Based Filtering**
- **Melakukan encoding**
  Lakukan encoding pada kolom user_id dan ISBN agar yang sebelumnya cuma berupa angka userid dan ISBN berupa angka acak, diubah menjadi angka integer yang berurutan. Hasilnya dapat dilihat di bawah ini.
![image](https://user-images.githubusercontent.com/113587270/190989127-3c0b5c11-de9c-40a4-a754-aaad9ab6dc38.png)
![image](https://user-images.githubusercontent.com/113587270/190989186-3dcb227e-f2f3-428e-a480-4c79160d5985.png)


 
- **Mapping**
  
  ![image](https://user-images.githubusercontent.com/113587270/190963970-4e74d9e0-1cc2-4f9c-bafd-b139583c804c.png)

  Berikutnya, petakan userID dan ISBN ke dataframe yang berkaitan.

- **Cek Data**
  Cek beberapa hal dalam data seperti jumlah user, jumlah resto, dan mengubah nilai rating menjadi float.
  
- **Train-Test-Split**
  Pertama acak terlebih dahulu dataset.
  
  ![image](https://user-images.githubusercontent.com/113587270/190964758-173b0e0a-bba0-4966-ac4f-fa54dbca7c6e.png)
  
  Selanjutnya, bagi data train dan test dengan komposisi 70:30. Namun sebelumnya, kita perlu memetakan (mapping) data user dan buku menjadi satu value terlebih dahulu. 

## Modelling dan Result
### Content Based Filtering
-   Modelling menggunakan fungsi tfidfvectorizer() dari library sklearn. Disini kita akan mengambil beberapa kata penting dari book_author untuk mengidentifikasi sistem rekomendasi berdasarkan penulis yang sama.

-   Kemudian lakukan fit transformasi dari list book_author tadi ke dalam bentuk matriks. Sehingga tercipta matriks seperti di bawah ini,
    ![image](https://user-images.githubusercontent.com/113587270/190990430-0fbdc862-52d6-47d6-9c6d-6dc517973e6c.png)

-   Selanjutnya, itung derajat kesamaan (similarity degree) antar buku dengan teknik cosine similarity. Di sini, kita menggunakan fungsi cosine_similarity dari library sklearn sehingga didapat output.
    ![image](https://user-images.githubusercontent.com/113587270/190990894-da3f700c-17fc-4b9a-b21e-6e559acc7c02.png)

-   Kemudian, dengan menggunakan argpartition, saya akan mengambil 5 rekomendasi buku dengan penulis yang sama dengan buku yang telah dibaca sebelumnya.
-   Saya uji hal tersebut dengan menggunakan judul buku 'The Star Rover' karangan Jack London.
 
    ![image](https://user-images.githubusercontent.com/113587270/190991454-2f54d265-9a29-4747-b97c-2428e3473a67.png)
    
-   Hasil rekomendasinya yang diberikan oleh sistem berdasarkan buku di atas yaitu.
    ![image](https://user-images.githubusercontent.com/113587270/190991877-d5129d5f-7ab8-465c-9d78-8fd7d532bd87.png)
    
Berdasarkan hasil rekomendasi, sistem mengambil kata kunci Jack dan London sebagai bahan untuk merekomendasikan kepada user. Dari hasil di atas hanay 3 buku yang memiliki penulis yang sama dengan buku yang telah dibaca user, 2 lainnya masuk ke dalam rekomendasi dikarenakan mengandung kata Jack dan London pada penulisya.





### Collaborative Filtering
-   Pertama, kita melakukan proses embedding terhadap data user dan buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan buku. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

-   Lakukan kompilasi model. Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. Setelah itu lakukan train pada model.

Adapun hasil rekomendasinya sebagai berikut.

![image](https://user-images.githubusercontent.com/113587270/190993983-928e0721-a141-4ca8-827b-2f0fce032d95.png)




## Evaluation

**Content Based Filtering**
Pada Content Based Filtering, saya menggunakan metriks Precision.
Precision Adalah sebuah metrics yang digunakan untuk mengukur berapa jumlah prediksi benar yang telah dibuat.Berikut adalah rumusnya :

![image](https://user-images.githubusercontent.com/113587270/190994341-e57168a8-ff31-4680-89c7-81eaf0d9ed1a.png)

kelebihan:
-   Sangat baik untuk klasifikasi
-   Dokumen yang dipilih secara acak dari kumpulan dokumen yang diambil adalah relevan.
-   Precision bagus untuk kasus di mana kelasnya seimbang
Kekurangan
-   Tidak baik untuk data yang Imbalance
-   Hanya hasil teratas yang dikembalikan oleh sistem

Pada hasil rekomendasi Content Based, dari 5 buku yang direkomendasikan hanya 3 buku yang memiliki penulis yang sesuai dengan buku yang telah dibaca user.

Sehingga bisa dihitung presisinya adalah 3/5 = 0.6 (60%)

![image](https://user-images.githubusercontent.com/113587270/190994855-038241ab-c558-43e4-80ce-dcd9b14dfa10.png)


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

#   Referensi

[1] M. Irfan, A. Dwi, and F. Hastarita, “SISTEM REKOMENDASI: BUKU ONLINE DENGAN METODE COLLABORATIVE FILTERING,” vol. 7, no. c, pp. 76–84, 2014.
