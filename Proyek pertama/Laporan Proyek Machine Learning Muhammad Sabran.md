# Laporan Proyek Machine Learning Muhammad Sabran

## Domain Proyek

Titanic merupakan sebuah kapal pesiar mewah dengan panjang 254 meter dan bagian lambungnya dibagi menjadi 16 kompartemen. Hal ini membuat Titanic sangat dielu-elukan pada zamannya sebagai kapal pesiar terbesar dan termewah pada zaman itu. Namun nahas, di tanggal 15 April 1912, Kapal yang membawa 2.200 orang penumpang dan awak tersebut tenggelam di Samudera Atlantik setelah kapal tersebut tidak sengaja menabrak sebuah gunung es.

Dilansir dari situs Kompas melalui [link berikut](https://www.kompas.com/tren/read/2021/04/15/073927465/hari-ini-dalam-sejarah-tenggelamnya-kapal-titanic?page=all), akibat kejadian tersebut 1.500 orang meregang nyawa dan hanya 700 orang yang selamat. Tentunya selain faktor keberuntungan, terdapat faktor lain yang mempengaruhi penumpang tersebut bisa selamat. Oleh karena itu, proyek Machine Learning kali ini akan menganalisis faktor apa yang mempengaruhi penumpang selamat serta dapat memprediksi penumpang mana yang selamat dan tidak berdasarkan data.

## Business Understanding

Berdasarkan film Titanic yang mengangkat kisah tragis tenggelamnya kapal, para perempuan, lansia, dan anak-anak didahulukan untuk diselamatkan. Hal ini tentunya mempengaruhi faktor keselamatan para penumpang titanic. Yang artinya umur serta gender mempengaruhi keselamatan penumpang. Selain itu juga, digambarkan bahwa penumpang VIP atau yang memiliki tiket First Class lebih diprioritaskan sehingga tentunya akan mempengaruhi faktor keselematan penumpang.

### Problem Statement:
Berdasarkan permasalahan di atas, maka permasalahan yang ditemukan yaitu.
- Apakah faktor yang mempengaruhi tingkat keselamatan penumpang Titanic?
- Bagaimana cara memprediksi penumpang Titanic bisa selamat atau tidak?

### Goals:
Maka berdasarkan permasalahan di atas, adapun tujuan dari proyek ini yaitu.
-   Mengetahui faktor yang mempengaruhi keselamatan penumpang Titanic.
-	Memprediksi Keselamatan penumpang - penumpang yang ada di kapal Titanic.

### Solution Statements:
-	Melakukan proses EDA untuk mengetahui korelasi dari setiap fiturnya.
-	Membuat Model Machine Learning dengan menggunakan beberapa algoritma antara lain:
	1.	Logistic Regression
	2.	Random Forest
	3.	LGBM Classifier

## Data Understanding

Adapun dataset yang digunakan diperoleh melalui situs kaggle yang dapat diunduh melalui [tautan](https://www.kaggle.com/competitions/titanic) berikut. Informasi yang diperoleh dari dataset tersebut yaitu di dalam data set tersebut terdapat 891 input, 11 fitur dan 1 label. 
### Penjelasan fitur-fitur
Adapun penjelasan detail mengenai fitur-fitur yang ada di dataset tersebut yaitu.
-	PasssengerId : Fitur nomor ID penumpang
- 	Survived : Fitur yang menandakan apakah penumpang selamat atau tidak (0=tidak, 1= selamat)
-	Pclass: Fitur untuk mengetahui tingkatan kelas yang dipilih (1 = 1st, 2 = 2nd, 3 = 3rd)
-	Sex : Fitur jenis kelamin penumpang
-	Age : Fitur umur penumpang
-	Sibsp : Fitur untuk memberitahu berapa banyak saudara atau pasangan dari penumpang tersebut
-	Parch : Fitur untuk memberitahu berapa banyak orangtua atau anak dari penumpang tersebut
-	Ticket : Fitur untuk mengetahui nomor tiket penumpang
-	Fare : Fitur untuk mengetahui harga tiket penumpang
-	Cabin : Fitur untuk memberi tahu nomor kabin penumpang.
-	Embarked :Fitur untuk mengetahui darimana penumpang berangkat (C = Cherbourg, Q = Queenstown, S = Southampton)

Tampilkan Dataframe yang telah diread dan cek banyak input serta fiturnya
![satu](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/1.PNG)

Tampilkan deskripsi dataset
![dua](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/2.PNG)

Lalu tampilkan tipe data

![tiga](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/3.PNG)

### EDA Univariate

![empat](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/4.png)

Berdasarkan grafik di atas mayoritas penumpang kapal Titanic tidak selamat




![lima](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/5.png)

Berdasarkan grafik tersebut, mayoritas penumpang berada di kelas tiga



![enam](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/6.png)

Berdasarkan grafik di atas, mayoritas penumpang berjenis kelamin laki-laki



![tujuh](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/7.png)

Berdasarkan grafik diatas, mayoritas penumpang tidak membawa saudara atau pasangannya


![delapan](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/8.png)

Berdasarkan grafik diatas, mayoritas penumpang tidak membawa orangtua atau anaknya


![9](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/9.png)

Berdasarkan grafik diatas, mayoritas penumpang berangkat dari pelabuhan Southampton


![10](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/10.png)

Berdasarkan grafik diatas, mayoritas usia penumpang berada di rentang 18-33 tahun


![11](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/11.png)

Berdasarkan grafik diatas, mayoritas penumpang membeli tiket dengan harga yang murah


### EDA Multivariate

![12](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/12.png)

Berdasarkan grafik diatas, mayoritas penumpang yang tidak selamat ialah laki-laki, sedangkan mayoritas yang selamat ialah perempuan


![13](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/13.png)

Berdasarkan grafik diatas, mayoritas penumpang yang tidak selamat berasal dari kelas ketiga sedangkan yang selamat berasal dari kelas pertama


![14](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/14.png)

Berdasarkan grafik diatas, mayoritas penumpang yang selamat maupun tidak selamat berangkat dari pelabuhan Southampton. Artinya mayoritas penumpang berangkat dari pelabuhan tersebut.

**Korelasi Matriks**


![16](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/16.PNG)

Berdasarkan korelasi di atas, yang mempengaruhi label Survived ialah fitur Fare dan Pclass.


## Data Preparation

Dalam data preparation saya melakukan 3 hal sebelum memasukkan data ke model latih:

-	Encoding Fitur Kategori
	Encoding Fitur Kategori dilaksanakan di beberapa fitur bertipe objek, yaitu sex dan embark. Hal ini dilakukan karena model machine learning hanya dapat menerima data dalam bentuk numerik. Saya melaksanakan encoding fitur kategori dengan menggunakan fitur get_dummies.
-	Train-Test-Split
	Membagi dataset menjadi data latih dan data validasi adalah hal yang harus kita lakukan sebelum melatih model. Hal ini dilakukan supaya kita dapat melakukan validasi dengan benar tanpa bias dari model.
-	Standarisasi
	Algoritma machine learning akan lebih baik dan lebih optimal apabila dilatih pada model yang memiliki data dengan skala relative yang sama. Scaling ini dilaksanakan untuk membantu model machine learning yang akan dipakai lebih mudah diolah. Saya melaksanakan standarisasi pada kolom Age, Sibsp, dan Fare dengan StandardScaler sebagai fitur standarisasi.

## Modeling

Model – model yang saya pakai dalam projek ini adalah:
1.	**Logistic Regression**
    Logistic Regression adalah sebuah algoritma klasifikasi di mana algoritma ini mencari hubungan antar fitur diskrit/kontinu dengan probabilitias hasi loutput diskrit tertentu.
2.	**XGBClassifier**
   Merupakan salah satu dari gradient boosting algorithm yang sangat efisien dan fleksibel. XGBClassifier juga memiliki parallel tree boosting. Gradient Boosting algorithm pada dasarnya optimal karena kesalahan diminimalkan dengan menggunakan algoritma penurunan gradien(Berbeda dengan loss function pada umumnya). XGBClassifier ini digunakan untuk masalah klasifikasi.


Berikut adalah tabel evaluasi dari keempat model:

![Evaluasi](https://raw.githubusercontent.com/farelarden/Dicoding-SIB/main/14.JPG))

Berikut XGBRegressor memiliki Accuracy, Precision, Recall, dan F1 Score terbaik dari 3 model lainnya, membuatnya menjadi model terbaik dari keempat model yang saya pakai.
## Evaluasi

Sebelum ke metrik evaluasi, terlebih dahulu kita harus mengerti tentang confusion matrix.
 
Di dalam confusion matrix, terdapat 4 kesimpulan yang dapat kita ambil:
-	True Positive (TP): Jumlah prediksi positif yang benar terhadap jumlah positif yang sebenarnya.
-	False Positive (FP): Jumlah prediksi positif yang salah.
-	True Negative (TN): Jumlah prediksi negatif yang benar terhadap jumlah negatif yang sebenarnya.
-	False Negative (FN): Jumlah prediksi negatif yang salah.

Saya menggunakan 4 metrik evaluasi dalam projek ini. 4 metrik evaluasi tersebut adalah:
-	**Accuracy**: Ratio dari True Positives dan True Negative terhadap seluruh positif dan negative di seluruh observasi.
	Rumus Accuracy Score = (TP + TN)/ (TP + FN + TN + FP) 
-	**Precision**: Kemampuan model untuk memprediksi nilai positif terhadap seluruh jumlah positif yang diprediksi oleh model.
	Rumus Precision Score = TP / (FP + TP)
-	**Recall**: Kemampuan model untuk memprediksi nilai positif terhadap seluruh jumlah positif yang sesungguhnya.
	Rumus Recall Score = TP / (FN + TP)
-	**F1**: Metrik yang menimbang kemampuan model untuk memberikan Precision dan Recall.
	Rumus F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)

Berikut adalah tabel evaluasi dari keempat model:

![Evaluasi](https://raw.githubusercontent.com/farelarden/Dicoding-SIB/main/14.JPG))
 
