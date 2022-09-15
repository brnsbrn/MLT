# Laporan Proyek Machine Learning Muhammad Sabran

## Domain Proyek

Titanic merupakan sebuah kapal pesiar mewah dengan panjang 254 meter dan bagian lambungnya dibagi menjadi 16 kompartemen. Hal ini membuat Titanic sangat dielu-elukan pada zamannya sebagai kapal pesiar terbesar dan termewah pada zaman itu. Namun nahas, di tanggal 15 April 1912, Kapal yang membawa 2.200 orang penumpang dan awak tersebut tenggelam di Samudera Atlantik setelah kapal tersebut tidak sengaja menabrak sebuah gunung es.

Dilansir dari situs Kompas melalui [link berikut](https://www.kompas.com/tren/read/2021/04/15/073927465/hari-ini-dalam-sejarah-tenggelamnya-kapal-titanic?page=all), akibat kejadian tersebut 1.500 orang meregang nyawa dan hanya 700 orang yang selamat. Tentunya selain faktor keberuntungan, terdapat faktor lain yang memengaruhi penumpang tersebut bisa selamat. Seperti yang dibahas oleh **Yogesh Kakde** dalam penelitiannya yang berjudul [Predicting Survival on Titanic by Applying Exploratory
Data Analytics and Machine Learning Techniques](https://www.researchgate.net/profile/Yogesh-Kakde/publication/325228831_Predicting_Survival_on_Titanic_by_Applying_Exploratory_Data_Analytics_and_Machine_Learning_Techniques/links/5c068f63a6fdcc315f9c0bb9/Predicting-Survival-on-Titanic-by-Applying-Exploratory-Data-Analytics-and-Machine-Learning-Techniques.pdf)[1], di mana mungkin saja faktor-faktor seperti umur, jenis kelamin, dan tingkatan kelas dapat memengaruhi peluang selamat atau tidaknya penumpang kapal Titanic.

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

Berdasarkan grafik di atas mayoritas penumpang kapal Titanic tidak selamat.





![5](https://user-images.githubusercontent.com/113587270/190392609-88f90cd8-fecf-490b-afe9-37ab79e6c628.png)

Berdasarkan grafik tersebut, mayoritas penumpang berada di kelas tiga.




![6](https://user-images.githubusercontent.com/113587270/190392640-3075d0d3-5bfc-4f00-a8fb-475b4dfeed32.png)

Berdasarkan grafik di atas, mayoritas penumpang berjenis kelamin laki-laki.



![tujuh](https://user-images.githubusercontent.com/113587270/190392650-f6e6e2cd-f9a4-4080-8db4-9224f03db8ea.png)

Berdasarkan grafik diatas, mayoritas penumpang tidak membawa saudara atau pasangannya.



![8](https://user-images.githubusercontent.com/113587270/190392744-234090af-dbfc-415c-90dc-d2ec5ae63ed9.png)

Berdasarkan grafik diatas, mayoritas penumpang tidak membawa orangtua atau anaknya.



![9](https://user-images.githubusercontent.com/113587270/190392772-f415ad22-f999-48a3-9ab9-89ee5b4db3ad.png)

Berdasarkan grafik diatas, mayoritas penumpang berangkat dari pelabuhan Southampton.


![10](https://user-images.githubusercontent.com/113587270/190392804-8c4413c0-e578-48ab-9db4-9aecb61c0dd0.png)

Berdasarkan grafik diatas, mayoritas usia penumpang berada di rentang 18-33 tahun.



![11](https://user-images.githubusercontent.com/113587270/190392838-9bcb0a9c-a5a4-4e0e-b977-d03f52097b1d.png)

Berdasarkan grafik diatas, mayoritas penumpang membeli tiket dengan harga yang murah.


### EDA Multivariate


![12](https://user-images.githubusercontent.com/113587270/190392929-51b206d3-af84-4dd8-98db-98f8c57e0470.png)

Berdasarkan grafik diatas, mayoritas penumpang yang tidak selamat ialah laki-laki, sedangkan mayoritas yang selamat ialah perempuan.



![13](https://user-images.githubusercontent.com/113587270/190392946-3723044b-1509-429b-b8e9-64b64455e140.png)

Berdasarkan grafik diatas, mayoritas penumpang yang tidak selamat berasal dari kelas ketiga sedangkan yang selamat berasal dari kelas pertama.



![14](https://user-images.githubusercontent.com/113587270/190393469-f7902fef-9add-49f2-8cc5-967097f16aa7.png)

Berdasarkan grafik diatas, mayoritas penumpang yang selamat maupun tidak selamat berangkat dari pelabuhan Southampton. Artinya mayoritas penumpang berangkat dari pelabuhan tersebut.

**Korelasi Matriks**



![16](https://user-images.githubusercontent.com/113587270/190393539-9caf0021-e9a4-4134-a89f-72c0ba4c8170.PNG)

Berdasarkan korelasi di atas, yang mempengaruhi label Survived ialah fitur Fare dan Pclass.


## Data Preparation

-	Langkah pertama yang saya lakukan ialah menggabungkan _dataframe Training_ dan _Test_ kemudian lakukan pengecekan apakah terdapat data yang hilang pada _dataframe_ tersebut. Karena kolom _Cabin_ terdapat banyak nilai yang hilang, maka kolom _Cabin_ akan dihapus. Selanjutnya, karena kolom _Age_ dan _Fare _merupakan numerikal, maka isi data yang hilang dengan nilai mean dari kolom tersebut. Sedangkan karena _Embarked_ merupakan kategorikal, maka data yang hilang diisi dengan modus dari kolom tersebut.
-	Kemudian lakukan **_Encoding_** untuk fitur kategorikal yaitu _Sex_ dan _Embarked_. Untuk kolom _Sex_,  0 mewakili wanita dan 1 mewakili pria. Sedangkan untuk kolom _Embarked_, 0 mewakili C, 1 mewakili G, dan 2 mewakili S.
-	Kemudian lakukan **_Train_Data_Split_** atau membagi dataset menjadi _data train_ dan _data test_. Disini saya membagi ukuran _data train_ menjadi 75% dan_ data test_ menjadi 25%.


## Modeling

Model – model yang saya pakai dalam projek ini adalah:
1.	_**Logistic Regression**_
    Dalam statistika digunakan untuk prediksi probabilitas kejadian suatu peristiwa dengan mencocokkan data pada fungsi logit kurva logistik. Nantinya dengan algoritma _Logistic Regression_, akan memprediksi Y (selamat atau tidak) berdasarkan X (beberapa fitur numerikal) pada _data test_. Kemudian model akan dilatih menggunakan data train untuk meningkatkan akurasinya. Berikut pembuatan modelnya.
    
    ![23](https://user-images.githubusercontent.com/113587270/190399360-1e9e430f-f1b0-44fa-a379-3f72fd9a087b.PNG)

    
2.	_**Random Forest**_
   Adalah algoritma dalam _machine learning_ yang digunakan untuk pengklasifikasian dataset dalam jumlah besar. Karena fungsinya bisa digunakan untuk banyak dimensi dengan berbagai skala dan performa yang tinggi. Klasifikasi ini dilakukan melalui penggabungan tree dalam decision tree dengan cara training dataset yang Anda miliki. Nantinya ia akan menggabungkan beberapa decision tree. Decision tree ini nantinya akan berisi percabangan dalam menganalisa fitur X pada _data train_ dan _data test_, sehingga diakhir nanti ia dapat memprediksi nilai Y (selamat atau tidak). Berikut pembuatan modelnya.
   ![24](https://user-images.githubusercontent.com/113587270/190399407-919c5844-8601-4817-ab2a-1c833c83f159.PNG)

   
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
