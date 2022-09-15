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

Langkah pertama yang saya lakukan ialah menggabungkan dataframe Training dan Test

![17](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/17.PNG)

Lalu cek apakah terdapat missing values

![18](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/18.PNG)

Lakukan langkah untuk mengatasi missing values

![19](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/19.PNG)

Lakukan Log Transformation untuk Uniform Data Distribution

![20](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/20.PNG)

Lakukan Encoding pada fitur kategorikal

![21](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/21.PNG)

Lakukan Train Data Split untuk membagi dataset

![22](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/22.PNG)

## Modeling

Model – model yang saya pakai dalam projek ini adalah:
1.	**Logistic Regression**
    Regresi logistik (kadang disebut model logistik atau model logit), dalam statistika digunakan untuk prediksi probabilitas kejadian suatu peristiwa dengan mencocokkan data pada fungsi logit kurva logistik.
    Berikut pembuatan modelnya.
    
    ![23](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/23.PNG)

    
2.	**Random Forest**
   Random Forest adalah algoritma dalam machine learning yang digunakan untuk pengklasifikasian data set dalam jumlah besar. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang Anda miliki. Berikut pembuatan modelnya.
   
   ![24](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/24.PNG)
   
   
   
3.	**LightGBM Classifier**
   LightGBM adalah algoritma berbasis histogram yang menempatkan nilai kontinu ke dalam tong diskrit, yang mengarah pada pelatihan yang lebih cepat dan penggunaan memori yang lebih efisien. Pada bagian ini, kita akan menjelajahi LightGBM secara mendalam.
   
   
   ![25](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/25.PNG)


Dikarenakan dari ketiga model tersebut, model LGBM Classifier memiliki tingkat Accuracy dan CV Score yang tinggi dibanding yang lain, maka kita akan menggunakan model LGBM Classifier


## Evaluasi

Matriks evaluasi yang digunakan ialah Accuracy dan Cross Validatin Score.

**Accuracy**
Merupakan rasio prediksi Benar (positif dan negatif) dengan keseluruhan data. 

![acc](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/WhatsApp%20Image%202022-09-15%20at%2018.47.24.jpeg)


**Cross Validation**
Cross validation adalah suatu metode tambahan dari teknik data mining yang bertujuan untuk memperoleh hasil akurasi yang maksimal. Metode ini sering juga disebut dengan k-fold cross validation. Berikut cara kerja Cross Validation.

![cv](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/croos%20validation.jpg)

Berdasarkan pelatihan dengan model LGBMClassifier. Model dapat memprediksi kemungkinan penumpang apakah selamat atau tidak seperti di bawah ini.


![test](https://github.com/brnsbrn/MLT/blob/main/Proyek%20pertama/Ss/26.PNG)

Adapun fitur yang mempengaruhi Keselamatan Penumpang ialah Fare atau harga tiker dan Pclass. Dikarenakan kebanyakan yang membeli tiket dengan harga murah akan tidak selamat, dapat disimpulkan bahwa mereka membeli tiket kelas ketiga. Sehingga penumpang dengan tiket kelas pertama leboh memiliki kemungkinan selamat yang lebih besar.
