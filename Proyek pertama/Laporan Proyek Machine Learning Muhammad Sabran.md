# Laporan Proyek Machine Learning Muhammad Sabran

## Domain Proyek

Titanic merupakan sebuah kapal pesiar mewah dengan panjang 254 meter dan bagian lambungnya dibagi menjadi 16 kompartemen. Hal ini membuat Titanic sangat dielu-elukan pada zamannya sebagai kapal pesiar terbesar dan termewah pada zaman itu. Namun nahas, di tanggal 15 April 1912, Kapal yang membawa 2.200 orang penumpang dan awak tersebut tenggelam di Samudera Atlantik setelah kapal tersebut tidak sengaja menabrak sebuah gunung es.

Dilansir dari situs Kompas melalui [link berikut](https://www.kompas.com/tren/read/2021/04/15/073927465/hari-ini-dalam-sejarah-tenggelamnya-kapal-titanic?page=all), akibat kejadian tersebut 1.500 orang meregang nyawa dan hanya 700 orang yang selamat. Tentunya selain faktor keberuntungan, terdapat faktor lain yang memengaruhi penumpang tersebut bisa selamat. Seperti yang dibahas oleh **Yogesh Kakde** dalam penelitiannya yang berjudul [Predicting Survival on Titanic by Applying Exploratory Data Analytics and Machine Learning Techniques](https://www.researchgate.net/profile/Yogesh-Kakde/publication/325228831_Predicting_Survival_on_Titanic_by_Applying_Exploratory_Data_Analytics_and_Machine_Learning_Techniques/links/5c068f63a6fdcc315f9c0bb9/Predicting-Survival-on-Titanic-by-Applying-Exploratory-Data-Analytics-and-Machine-Learning-Techniques.pdf) [1], di mana mungkin saja faktor-faktor seperti umur, jenis kelamin, dan tingkatan kelas dapat memengaruhi peluang selamat atau tidaknya penumpang kapal Titanic.

## Business Understanding

Berdasarkan film Titanic yang mengangkat kisah tragis tenggelamnya kapal, para perempuan, lansia, dan anak-anak didahulukan untuk diselamatkan. Hal ini tentunya mempengaruhi faktor keselamatan para penumpang titanic. Yang artinya umur serta jenis kelamin memengaruhi keselamatan penumpang. Selain itu juga, digambarkan bahwa penumpang _VIP_ atau yang memiliki tiket kelas pertama lebih diprioritaskan sehingga tentunya akan memengaruhi faktor keselamatan penumpang.

### Problem Statement:
Berdasarkan permasalahan di atas, maka permasalahan yang ditemukan yaitu.
- Apakah faktor yang memengaruhi tingkat keselamatan penumpang Titanic?
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

Adapun dataset yang digunakan diperoleh melalui situs kaggle yang dapat diunduh melalui [tautan](https://www.kaggle.com/datasets/mirichoi0218/insurance). Dataset ini memiliki 1339 data dan 7 fitur. Fitur yang terdiri dari fitur numerikal (age, bmi, children, dan charges) serta fitur kategorikal (sex, smoker, dan region).
Adapun penjelasan detail mengenai fitur-fitur yang ada di dataset tersebut yaitu.
- age : Umur pasien
- sex : Jenis kelamin pasien
- bmi : Body Mass Index (berat badan normal/sehat)
- children : Jumlah anak yang ditanggung oleh asuransi
- smoker : Merokok atau tidak
- region : Tempat asal (northeast, southeast, southwest, northwest)
- charges : Biaya medis individu (per orang) 

### EDA Univariate


![download](https://user-images.githubusercontent.com/113587270/190840585-0a68e9d9-03ff-4ca8-80ab-dcfb41e45e02.png)


Berdasarkan grafik di atas jumlah laki-laki dan perempuan tidak terpaut jauh, meskipun didominasi oleh laki-laki.



![image](https://user-images.githubusercontent.com/113587270/190840797-3693aae9-892f-45aa-b01c-e57af08d932d.png)


Berdasarkan grafik di atas, sebesar 79.5% pasien bukan merupakan perokok.


![image](https://user-images.githubusercontent.com/113587270/190840840-51cf5bfb-c5dc-49e4-b238-b73b39407cf3.png)

Berdasarkan grafik di atas, mayoritas pasien berasal dari southeast.



### EDA Multivariate

![image](https://user-images.githubusercontent.com/113587270/190840953-4d3fee47-0380-4065-9757-9eb94220f11b.png)

![image](https://user-images.githubusercontent.com/113587270/190840960-217d6ef1-6d61-4a53-b6fd-e5fd06ace887.png)

![image](https://user-images.githubusercontent.com/113587270/190840910-7ca87982-5635-4afb-af2b-c153dd64dd49.png)

Berdasarkan ketiga grafik di atas, dapat disimpulkan bahwa yang sangat menentukan besarnya charges ialah status merokok atau tidak. Di mana apabila status pasien adalah seorang perokok, maka biaya pengobatannya akan lebih besar diandingkan dengan pasien yang tidak merokok.

**Korelasi Matriks**


![image](https://user-images.githubusercontent.com/113587270/190841074-9dc3e922-ab7a-4834-a2dc-d054c3ebb5f8.png)

Berdasarkan matriks di atas, fitur numerikal memiliki korelasi yang rendah terhadap charges. Maka kemungkinan fitur yang berkorelasi kuat ada di fitur kategorikal.


## Data Preparation

### Melakukan Encoding
Langkah pertama yaitu melakukan one-hot-encoding pada fitur kategorikal (sex, smoker, dan region) menggunakan get_dummies.

![image](https://user-images.githubusercontent.com/113587270/190841224-6de0a419-812a-4513-b3ba-980f8190450c.png)

Selanjutnya buat kembali korelasi matriks seluruh fitur yang ada.

![image](https://user-images.githubusercontent.com/113587270/190841288-1606ef3c-2db4-43cd-8855-39ae0c062203.png)

Maka didapatkan fitur yang berkorelasi kuat terhadap charges ialah fitur smoker (yes dan no).

### Melabeli Data
Yang pertama membuat dataframe X yang menampung variabel independen, caranya cukup dengan drop variabel dependen (charges).

![image](https://user-images.githubusercontent.com/113587270/190841382-ab2d2c03-58ce-46fb-b5c1-14619baa1539.png)

Selanjutnya buat dataframe y untuk menampung variabel dependen (charges)

![image](https://user-images.githubusercontent.com/113587270/190841406-677d74e1-b613-456a-bf9f-1a7105aaf715.png)

### Train-Test-Split
Selanjutnya membagi data sampel menjadi data train dan data test, dengan porsi 85% data train dan 15% data tes.

![image](https://user-images.githubusercontent.com/113587270/190841552-0196cf84-804c-4f00-bcbf-aa8c680868c2.png)

![image](https://user-images.githubusercontent.com/113587270/190841575-c2343b26-f8ac-4098-8ffc-f9ad4bef0ece.png)

### Standarisasi
Selanjutnya melakukan standarisasi menggunakan StandardScaler dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Sehingga nilai standar deviasi sama dengan 1 dan mean sama dengan 0. 

![image](https://user-images.githubusercontent.com/113587270/190841698-6f137525-a6a4-40a1-8b50-124d100effe7.png)

![image](https://user-images.githubusercontent.com/113587270/190841834-55150fa6-9631-4d1c-a9df-94fdd16f5e4e.png)





## Modeling

Model – model yang saya pakai dalam projek ini adalah:
1.	_**KNN**_
    merupakan algoritma yang berfungsi untuk melakukan klasifikasi suatu data berdasarkan data pembelajaran (train data sets), yang diambil dari k tetangga terdekatnya (nearest neighbors). Kelebihan dari algoritma ini yaitu mudah diimplementasi untuk menjadi sebuah model, namun kekurangannya model ini kurang efektif untuk data dalam jumlah besar.
    
    ![image](https://user-images.githubusercontent.com/113587270/190842113-e5203a0b-6dd5-4516-a112-eeb16274eee5.png)
   
    Disini saya menggunakan parameter jumlah tetangga sebanyak 5, artinya ia akan mengambil 5 tetangga dengan jarak terdekat (Menggunakan Euclidean Distance). Selanjutnya dia akan mengambil data mayoritas yang ada di 5 sampel tetangga tersebut untuk dimasukkan menjadi data baru.

    
2.	_**Random Forest**_
   Adalah algoritma dalam _machine learning_ yang digunakan untuk pengklasifikasian dataset dalam jumlah besar. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang Anda miliki. Nantinya ia akan menggabungkan beberapa decision tree. Nantinya random forest akan mencari fitur terbaik secara acak, fitur terbaik inilah yang akan berperan penting dalam meprBerikut pembuatan modelnya.
   ![image](https://user-images.githubusercontent.com/113587270/190842430-45c2cffc-f859-46e8-a5f9-61491c16de88.png)
Di sini saya menggunakan parameter n estimator sebanyak 100 yang artinya ia akan membuat sebanyak 100 cabang pohon,

   
3.	_**LightGBM Classifier**_
   Adalah algoritma berbasis histogram yang menempatkan nilai kontinu ke dalam tong diskrit, yang mengarah pada pelatihan yang lebih cepat dan penggunaan memori yang lebih efisien. Kurang lebih mirip dengan _Random Forest_, namun bedanya LGBM lebih berfokus menumbuhkan _Leaf_ pada _Decision Tree_ yang ia buat. Berikut modelnya.
![25](https://user-images.githubusercontent.com/113587270/190399574-85c74256-9aa2-46d7-bbb1-4549f70bafe4.PNG)


Dikarenakan dari ketiga model tersebut, model _LGBM Classifier_ memiliki tingkat _Accuracy_ dan _CV Score_ yang tinggi dibanding yang lain, maka kita akan menggunakan model _LGBM Classifier_.


## Evaluasi

Matriks evaluasi yang digunakan ialah Accuracy dan Cross Validatin Score.

_**Accuracy**_
Merupakan rasio prediksi benar (positif dan negatif) dengan keseluruhan data. 


![WhatsApp Image 2022-09-15 at 18 47 24](https://user-images.githubusercontent.com/113587270/190408220-313dea42-f9de-4e0b-bf7e-10d9b1f3ab89.jpeg)


_**Cross Validation**_
Suatu metode tambahan dari teknik _data mining_ yang bertujuan untuk memperoleh hasil akurasi yang maksimal. Metode ini sering juga disebut dengan _k-fold cross validation_. Berikut cara kerja _Cross Validation_.

![croos validation](https://user-images.githubusercontent.com/113587270/190408315-2b4f4840-b486-48da-ba41-ebca4a70dcdd.jpg)


## Kesimpulan

-	Berdasarkan pelatihan dengan model _LGBMClassifier_. Model dapat memprediksi kemungkinan penumpang apakah selamat atau tidak seperti di bawah ini.

	![26](https://user-images.githubusercontent.com/113587270/190408546-a445c804-8a92-4979-a2d7-33587de517f8.PNG)


-	Adapun fitur yang mempengaruhi keselamatan penumpang ialah _Fare_ atau harga tiket dan _Pclass_. Dikarenakan kebanyakan yang membeli tiket dengan harga murah akan tidak selamat, dapat disimpulkan bahwa mereka membeli tiket kelas ketiga. Sehingga penumpang dengan tiket kelas pertama lebih memiliki kemungkinan selamat yang lebih besar.

## Referensi

[1]	Y. Kakde and S. Agrawal, “Predicting Survival on Titanic by Applying Exploratory Data Analytics and Machine Learning Techniques,” Int. J. Comput. Appl., vol. 179, no. 44, pp. 32–38, 2018, doi: 10.5120/ijca2018917094.
