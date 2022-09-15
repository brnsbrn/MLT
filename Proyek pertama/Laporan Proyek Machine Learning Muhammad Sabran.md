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

### EDA Univariate


![4](https://user-images.githubusercontent.com/113587270/190392570-93d79a34-0431-471e-bfea-8be95ca67548.png)

Berdasarkan grafik di atas mayoritas penumpang kapal Titanic tidak selamat





![5](https://user-images.githubusercontent.com/113587270/190392609-88f90cd8-fecf-490b-afe9-37ab79e6c628.png)

Berdasarkan grafik tersebut, mayoritas penumpang berada di kelas tiga




![6](https://user-images.githubusercontent.com/113587270/190392640-3075d0d3-5bfc-4f00-a8fb-475b4dfeed32.png)

Berdasarkan grafik di atas, mayoritas penumpang berjenis kelamin laki-laki



![tujuh](https://user-images.githubusercontent.com/113587270/190392650-f6e6e2cd-f9a4-4080-8db4-9224f03db8ea.png)

Berdasarkan grafik diatas, mayoritas penumpang tidak membawa saudara atau pasangannya



![8](https://user-images.githubusercontent.com/113587270/190392744-234090af-dbfc-415c-90dc-d2ec5ae63ed9.png)

Berdasarkan grafik diatas, mayoritas penumpang tidak membawa orangtua atau anaknya



![9](https://user-images.githubusercontent.com/113587270/190392772-f415ad22-f999-48a3-9ab9-89ee5b4db3ad.png)

Berdasarkan grafik diatas, mayoritas penumpang berangkat dari pelabuhan Southampton


![10](https://user-images.githubusercontent.com/113587270/190392804-8c4413c0-e578-48ab-9db4-9aecb61c0dd0.png)

Berdasarkan grafik diatas, mayoritas usia penumpang berada di rentang 18-33 tahun



![11](https://user-images.githubusercontent.com/113587270/190392838-9bcb0a9c-a5a4-4e0e-b977-d03f52097b1d.png)

Berdasarkan grafik diatas, mayoritas penumpang membeli tiket dengan harga yang murah


### EDA Multivariate


![12](https://user-images.githubusercontent.com/113587270/190392929-51b206d3-af84-4dd8-98db-98f8c57e0470.png)

Berdasarkan grafik diatas, mayoritas penumpang yang tidak selamat ialah laki-laki, sedangkan mayoritas yang selamat ialah perempuan



![13](https://user-images.githubusercontent.com/113587270/190392946-3723044b-1509-429b-b8e9-64b64455e140.png)

Berdasarkan grafik diatas, mayoritas penumpang yang tidak selamat berasal dari kelas ketiga sedangkan yang selamat berasal dari kelas pertama



![14](https://user-images.githubusercontent.com/113587270/190393469-f7902fef-9add-49f2-8cc5-967097f16aa7.png)

Berdasarkan grafik diatas, mayoritas penumpang yang selamat maupun tidak selamat berangkat dari pelabuhan Southampton. Artinya mayoritas penumpang berangkat dari pelabuhan tersebut.

**Korelasi Matriks**



![16](https://user-images.githubusercontent.com/113587270/190393539-9caf0021-e9a4-4134-a89f-72c0ba4c8170.PNG)

Berdasarkan korelasi di atas, yang mempengaruhi label Survived ialah fitur Fare dan Pclass.


## Data Preparation

-	Langkah pertama yang saya lakukan ialah menggabungkan dataframe Training dan Test kemudian lakukan pengecekan apakah terdapat data yang hilang pada dataframe tersebut. Karena kolom Cabin terdapat banyak nilai yang hilang, maka kolom Cabin akan dihapus. Selanjutnya, karena kolom Age dan Fare merupakan numerikal, maka isi data yang hilang dengan nilai mean dari kolom tersebut. Sedangkan karena Embarked merupakan kategorikal, maka data yang hilang diisi dengan modus dari kolom tersebut.
-	Kemudian lakukan **_Encoding_** untuk fitur kategorikal yaitu Sex dan Embarked. Untuk kolom Sex,  0 mewakili wanita dan 1 mewakili pria. Sedangkan untuk kolom Embarked, 0 mewakili C, 1 mewakili G, dan 2 mewakili S.
-	Kemudian lakukan **_Train_Data_Split_** atau membagi dataset menjadi data train dan data test. Disini saya membagi ukuran data train menjadi 75% dan data test menjadi 25%.


## Modeling

Model – model yang saya pakai dalam projek ini adalah:
1.	**Logistic Regression**
    Regresi logistik (kadang disebut model logistik atau model logit), dalam statistika digunakan untuk prediksi probabilitas kejadian suatu peristiwa dengan mencocokkan data pada fungsi logit kurva logistik.
    Berikut pembuatan modelnya.
    
    ![23](https://user-images.githubusercontent.com/113587270/190399360-1e9e430f-f1b0-44fa-a379-3f72fd9a087b.PNG)

    
2.	**Random Forest**
   Random Forest adalah algoritma dalam machine learning yang digunakan untuk pengklasifikasian data set dalam jumlah besar. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang Anda miliki. Berikut pembuatan modelnya.
   
   ![24](https://user-images.githubusercontent.com/113587270/190399407-919c5844-8601-4817-ab2a-1c833c83f159.PNG)

   
3.	**LightGBM Classifier**
   LightGBM adalah algoritma berbasis histogram yang menempatkan nilai kontinu ke dalam tong diskrit, yang mengarah pada pelatihan yang lebih cepat dan penggunaan memori yang lebih efisien. Pada bagian ini, kita akan menjelajahi LightGBM secara mendalam.
   
   ![25](https://user-images.githubusercontent.com/113587270/190399574-85c74256-9aa2-46d7-bbb1-4549f70bafe4.PNG)


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
