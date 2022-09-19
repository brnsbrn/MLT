# Laporan Sistem Rekomendasi Buku Muhammad Sabran

## Project Overview
Berdasarkan salah satu artikel dari Universitas Hasanuddin [tautan](https://journal.unhas.ac.id/index.php/jupiter/article/view/1672), membaca buku merupakan hal yang penting untuk dilakukan. Orang - orang yang memilih untuk sering membaca buku memiliki wawasan yang luas. Lewat membaca, kita juga dapat mengetahui, mengenal banyak hal yang sebelumnya belum dikenal dan kita pelajari dan pahami lewat membaca buku.

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


## Evaluation

**Content Based Filtering**
Pada Content Based Filtering, saya menggunakan akurasi sebagai metrik evaluasi.
Akurasi pada Content Based Filtering adalah:
Jumlah buku yang direkomendasikan sesuai dengan penulis buku / Jumlah buku yang direkomendasikan

Dalam mengaplikasi metrik akurasi pada kode adalah dengan membuat variabel books_that_have_been_read_row yang akan mengambil satu row dari buku yang pernah dibaca sebelumnya, dan juga membuat variabel books_that_have_been_read_author adalah penulis buku dari buku yang pernah dibaca sebelumnya.

Saya juga membuat variabel  book_recommendation_authors merupakan sebuah list yang terdiri dari penulis - penulis dari buku - buku yang direkomendasikan oleh sistem.
Kemudian saya membuat looping yang merupakan proses manual di mana setiap penulis dari buku yang direkomendasikan akan dicek, apabila sama, maka variabel real_author akan bertambah 1.

Kemudian di bawah ini adalah hasil dari akurasi dari model sistem rekomendasi, di mana jumlah buku yang direkomendasikan sesuai dengan penulis buku (Variabel real_author) / Jumlah buku yang direkomendasikan (5).

![Accuracy](https://raw.githubusercontent.com/farelarden/Dicoding-SIB/main/27.JPG)

Kelebihan pada evaluasi metrik akurasi terletak pada kesederhanaannya untuk dipahami, karena metrik akurasi hanyalah jumlah yang benar dibandingkan dengan keseluruhan jawaban.

Sedangkan kekurangan pada evaluasi metrik adalah terletak pada kesederhanannya pula, karena metrik akurasi hanya menghitung jumlah yang benar dibandingkan dengan keseluruhan jawaban dan tidak memperhitungkan aspek lainnya.

**Collaborative Based Filtering**
Pada Collaborative Based Filtering, kita memiliki evaluasi metrik RMSE atau root-mean-square error.RMSE adalah ukuran yang sering digunakan untuk perbedaan antara nilai (nilai sampel atau populasi) yang diprediksi oleh model atau penduga dan nilai yang diamati.Berbeda dengan MSE, RMSE adalah hasil dari akar MSE membuat RMSE memiliki nilai yang lebih kecil dibandingkan dengan MSE.

Berikut adalah rumus RMSE:
![RMSE](https://raw.githubusercontent.com/farelarden/Dicoding-SIB/main/22.JPG)
Dimana,
At = Nilai data Aktual
Ft = Nilai hasil peramalan
N= banyaknya data
∑ = Summation (Jumlahkan keseluruhan  nilai)

RMSE memiliki nilai yang lebih kecil daripada MSE, dengan adanya nilai kecil ini dapat menjadi kelebihan dan kekurangan tersendiri. Kelebihan dari nilai kecil adalah kita tidak perlu takut karena nilai error nya kecil dan kita dapat langsung masuk ke tahap selanjutnya, namun dengan nilai error yang kecil, kita dapat menjadi terlalu percaya diri dengan modelnya tanpa melihat posibilitas overfitting atau undefitting.

RMSE terlebih dahulu kita definisikan pada bagian metrik dalam model, kemudian kita visualisasikan lewat grafik.
![Grafik](https://raw.githubusercontent.com/farelarden/Dicoding-SIB/main/24.JPG))


