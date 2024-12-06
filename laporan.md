# Laporan Proyek Machine Learning - M. Raditya Adhirajasa

## Domain Proyek

Polusi udara dianggap sebagai masalah lingkungan yang mempengaruhi kesehatan manusia, baik dalam jangka panjang maupun jangka pendek. Sekitar 7 juta kematian per tahun disebabkan oleh paparan polusi udara dan bencana alam atmosfer lainnya. Salah satu polutan udara yang paling berbahaya adalah partikel halus yang dikenal sebagai PM2.5, yaitu partikel dengan diameter kurang dari 2,5 mikrometer. PM2.5 dapat dengan mudah terhirup dan menembus jauh ke dalam saluran pernapasan, sehingga dapat menyebabkan berbagai penyakit pernapasan dan kardiovaskular, bahkan meningkatkan risiko kematian dini.

Kota-kota besar di seluruh dunia seperti Beijing menghadapi tingkat polusi udara yang sangat tinggi akibat urbanisasi, industrialisasi, dan aktivitas transportasi. Oleh karena itu, kemampuan untuk memprediksi konsentrasi PM2.5 sangat penting untuk memberikan peringatan dini kepada masyarakat, membantu para pembuat kebijakan dalam menerapkan langkah-langkah mitigasi, dan meningkatkan kesadaran masyarakat akan bahaya polusi udara.

Dataset yang digunakan dalam proyek ini mencatat data harian dan cuaca dari Kedutaan Besar AS di Beijing selama lima tahun. Informasi yang tersedia meliputi konsentrasi PM2.5, suhu, tekanan, kecepatan angin, dan kondisi hujan atau salju. Dengan memanfaatkan data tersebut dan pendekatan machine learning berbasis deret waktu, penelitian ini bertujuan untuk membangun model prediktif yang dapat memproyeksikan tingkat PM2.5 berdasarkan polusi dan kondisi cuaca beberapa jam sebelumnya.

**Mengapa masalah ini harus diselesaikan?**
- Dampak Kesehatan yang Signifikan

  PM2.5 adalah salah satu polutan udara yang paling berbahaya karena ukurannya yang kecil memungkinkannya masuk ke saluran pernapasan dan bahkan aliran darah. Paparan PM2.5 dalam jangka panjang telah dikaitkan dengan penyakit seperti asma, bronkitis, penyakit jantung, stroke, dan kanker paru-paru. Dengan meningkatnya jumlah orang yang tinggal di daerah perkotaan risiko kesehatan dari PM2.5 terus meningkat. Memiliki sistem prediksi yang dapat diandalkan dapat membantu orang untuk menghindari paparan berlebih dan mengurangi dampak kesehatan.
- Manajemen Risiko dan Mitigasi Lingkungan

  Kemampuan untuk memprediksi konsentrasi PM2.5 secara akurat memberikan peluang bagi pemerintah dan lembaga terkait untuk menerapkan langkah-langkah mitigasi seperti membatasi aktivitas industri atau kendaraan bermotor pada waktu-waktu tertentu. Hal ini juga memungkinkan pengambilan keputusan berdasarkan data untuk merancang kebijakan yang lebih efektif untuk mengurangi polusi udara.

**Bagaimana masalah ini diselesaikan?**
- Penggunaan Metode Time Series Forecasting

  Karena konsentrasi PM2.5 memiliki sifat berulang dan bergantung pada data historis, maka pendekatan deret waktu menjadi pilihan pertama. Model seperti Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU) dapat digunakan untuk menangkap pola temporal yang kompleks dan menghasilkan prediksi yang lebih akurat.

**Referensi**

[Air Pollution Forecasting Using Deep Learning](https://www.researchgate.net/publication/357043273_Air_Pollution_Forecasting_Using_Deep_Learning) 

## Business Understanding

Polusi udara, khususnya partikel PM2.5, telah menjadi masalah global dengan dampak yang luas terhadap kesehatan manusia dan lingkungan. Di kota-kota besar seperti Beijing, konsentrasi PM2.5 sering kali melebihi ambang batas aman yang ditetapkan oleh WHO. Mengingat kompleksitas faktor yang memengaruhi kualitas udara, seperti kondisi cuaca, aktivitas manusia, dan pola angin sulit untuk memprediksi tingkat polusi secara manual. Oleh karena itu, pendekatan berbasis data yang memanfaatkan teknologi pembelajaran mesin diperlukan untuk membuat prediksi yang akurat. Pada proyek ini, proses klarifikasi masalah dilakukan dengan mengidentifikasi hubungan antara variabel cuaca dan tingkat polusi udara untuk membangun model prediksi berdasarkan data historis. Dengan memanfaatkan pendekatan forecasting, model diharapkan dapat memberikan prediksi tingkat PM2.5 dalam satu jam ke depan, sehingga dapat membantu masyarakat dan pemangku kepentingan dalam mengambil keputusan yang lebih tepat.

### Problem Statements

- Apakah model LSTM dan GRU cocok pada dataset polusi udara?
- Berapa tingkat konsentrasi PM2.5 di waktu tertentu?

### Goals

- Menilai error yang dihasilkan dari model LSTM dan GRU.
- Membuat Model yang memprediksi tingkat konsentrasi PM2.5.

    ### Solution statements
    - Membangun Model Prediksi Time Series dengan LSTM dan GRU.

        Menggunakan dua algoritma model deep learning yaitu Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU). Keduanya dirancang untuk menangani data time series dengan dependensi jangka panjang. LSTM dan GRU akan digunakan untuk mempelajari pola historis dalam data polusi udara dan cuaca untuk memprediksi tingkat PM2.5 pada jam mendatang. Keunggulan dari LSTM dan GRU adalah kemampuannya untuk menangkap pola temporal yang kompleks serta mampu mengatasi masalah vanishing gradient.
    - Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE)

      MSE mengukur seberapa besar kesalahan prediksi model dibandingkan dengan nilai aktual sedangkan RMSE memberikan gambaran yang lebih intuitif mengenai seberapa jauh prediksi model dari nilai sebenarnya dengan memberikan penalti lebih besar pada kesalahan yang lebih besar.

## Data Understanding
Data ini dikumpulkan di US Embassy, Beijing, yang mencatat kualitas udara dan parameter cuaca terkait. Dataset yang digunakan berjumlah 43,800 untuk data latih dan 346 untuk data uji, yang mencakup periode dari awal tahun 2010 bulan januari hingga akhir tahun 2014 bulan desember. Data dicatat pada interval waktu setiap jam, memberikan informasi terperinci tentang kondisi polusi udara dan cuaca setiap jam. Setiap baris data menggambarkan kondisi polusi dan cuaca pada waktu tertentu (tahun, bulan, hari, jam) dan mencakup berbagai variabel yang relevan. 

Link Dataset : [Air Pollution Forecasting](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate).

### Variabel-variabel pada dataset adalah sebagai berikut:
- date : merupakan keterangan waktu.
- pollution : merupakan konsentrasi polusi dalam PM2.5.
- dew : merupakan titik embun.
- temp : merupakan suhu udara.
- press : merupakan tekanan atmosfer.
- wnd_dir : merupakan arah pergerakan angin.
- wnd_spd : merupakan kecepatan angin.
- snow : merupakan jumlah jam terkumulasi salju.
- rain : merupakan jumlah jam terkumulasi hujan.

**Exploratory Data Analytic**:
- Deskripsi Variable

  ![image](https://github.com/user-attachments/assets/19f2d1ed-39ec-42eb-8d74-9f15950d7729)

  Melihat deskripsi dataset dengan menggunakan fungsi info(). Hal ini diperlukan agar dapat memahami data lebih lanjut seperti jumlah data, jumlah kolom, tipe data tiap kolom, dll. Berdasarkan deskripsi data tersebut bisa diapatkan gambaran singkat tentang struktur dataset, termasuk jumlah data yang valid di setiap kolom dan tipe data yang perlu diproses lebih lanjut. Data tersebut memiliki 2 tipe object, 4 tipe float, 3 tipe integer.

  ```
  print("train data\n", df_train.isnull().sum())
  ```
  ![image](https://github.com/user-attachments/assets/91cc3652-848e-43a7-b2da-9d47548ec1a1)

  Selanjutnya perlu dilihat apakah data memiliki missing value atau tidak. Dapat dilihat bahwa dataset tidak memiliki missing value.

  ![image](https://github.com/user-attachments/assets/ecf0967a-2c5c-4dfc-9151-348ffd7f1f05)

  Menggunakan fungsi describe() dapat membantu untuk melihat gambaran umum tentang distribusi data numerik, apakah terdapat outliers atau nilai ekstrim, serta variasi data pada masing-masing kolom. Ini membantu dalam memahami seberapa tersebar data dan apakah perlu ada penyesuaian lebih lanjut.

- Visualisasi Data

  ![image](https://github.com/user-attachments/assets/f01f9d31-64c5-479d-b30b-9f62afc1a7cd)

  Dengan menampilkan boxplot pada data seperti yang dilakukan pada kode di atas digunakan untuk menganalisis distribusi dan karakteristik data dari berbagai fitur (kolom) numerik dalam dataset. Boxplot dapat membantu untuk memahami distribusi dan variabilitas data serta mendekteksi adanya outlier.

  ![image](https://github.com/user-attachments/assets/eac83840-9cfc-4a90-957f-f892306e69ea)

  Menampilkan histogram seperti diatas dapat membantu untuk memahami jumlah data dari distribusi nya secara jelas dari setiap kolom.

## Data Preparation

- Encode Data

  Pada tahap ini saya menerapkan label encoding pada data wnd_dir bertipe data object/kategorikal. Encode data adalah proses mengubah data ke dalam format yang dapat dipahami oleh komputer, terutama model machine learning. Encoding biasanya dilakukan pada data yang tidak berbentuk numerik, seperti data kategorikal atau teks, agar dapat diolah oleh algoritma yang hanya menerima input berupa angka/numerik.
  
- Normalisasi data

  Pada tahap ini saya menggunakan Min-Max Scalar. Min-Max Scaler adalah metode normalisasi yang digunakan untuk mengubah nilai fitur dalam dataset ke rentang tertentu, antara 0 dan 1. Teknik ini memastikan bahwa semua fitur memiliki skala yang sama, tanpa mengubah distribusi datanya. Normalisasi data membantu algoritma machine learning, terutama neural networks seperti LSTM dan GRU, untuk bekerja lebih efisien dengan skala data yang seragam. Ini mengurangi risiko bias terhadap fitur dengan nilai yang lebih besar.

- Transform input dan target data

  Pada tahap ini menjadi input (X) dan target (y) berdasarkan jumlah langkah waktu (n_past) dan jumlah prediksi di masa depan (n_future).
Proses ini memastikan bahwa data siap untuk pelatihan dan evaluasi dengan model seperti LSTM atau GRU.


## Modeling

**LSTM**

LSTM (Long Short-Term Memory) adalah jenis recurrent neural network (RNN) yang dirancang untuk mengatasi masalah vanishing gradient dan long-term dependencies dalam data time series. LSTM memiliki kemampuan untuk "mengingat" informasi dari waktu yang lama dengan menggunakan gates untuk mengatur aliran informasi ke dalam dan keluar dari memori.

![image](https://github.com/user-attachments/assets/affc6032-792f-4858-9b64-5b12793605d0)


- Tahapan LSTM
  -  Arsitektur Model: Menentukan struktur lapisan LSTM dan Dense sesuai dengan kompleksitas masalah.
  -  Kompilasi Model: Memilih fungsi loss, optimizer, dan metrik evaluasi yang sesuai.
  -  Pelatihan Model: Menggunakan data latih untuk melatih model selama beberapa epoch.
  -  Evaluasi Model: Menggunakan data uji untuk mengevaluasi performa model dengan MSE dan RMSE.

- Parameter LSTM:
  -  units: Jumlah neuron dalam setiap lapisan LSTM, dipilih berdasarkan kompleksitas pola dalam data.
  -  dropout: Nilai regulasi untuk mencegah overfitting (0.3 = 30%).
  -  learning_rate: Tingkat pembelajaran untuk mengontrol kecepatan optimasi.
  -  return_sequences: Mengatur apakah lapisan LSTM akan mengembalikan seluruh urutan atau hanya output terakhir.

- Kelebihan LSTM:
  -  LSTM mampu menangkap pola jangka panjang karena LSTM dirancang untuk mengatasi masalah long-term dependencies dalam data time series.
  -  Lapisan LSTM mampu menangani data yang memiliki noise moderat seperti pada time series.

- Kekurangan LSTM:
  -  Pelatihan LSTM membutuhkan waktu lebih lama dibandingkan model yang lebih sederhana.
  -  LSTM tergolong sulit untuk hyperparameter tuning seperti jumlah neuron, learning rate, dan dropout membutuhkan eksperimen yang ekstensif.

**GRU**

GRU (Gated Recurrent Unit) adalah jenis Recurrent Neural Network (RNN) yang dirancang untuk menangani masalah pada data berurutan (time series) dengan mengatasi keterbatasan pada RNN tradisional, seperti masalah vanishing gradient. GRU memiliki arsitektur yang lebih sederhana dibandingkan LSTM (Long Short-Term Memory), karena hanya menggunakan dua gate: update gate dan reset gate.

![image](https://github.com/user-attachments/assets/4f82df0c-3e44-4e6a-a940-102b06fb00de)


- Tahapan GRU
  -  Arsitektur Model: Menentukan struktur lapisan GRU dan Dense sesuai dengan karakteristik data time series.
  -  Kompilasi Model: Memilih fungsi loss, optimizer, dan metrik evaluasi yang sesuai.
  -  Pelatihan Model: Melatih model dengan data latih untuk beberapa epoch.
  -  Evaluasi Model: Menggunakan data uji untuk mengevaluasi performa model berdasarkan metrik MSE dan RMSE.

- Parameter GRU:
  -  units: Jumlah neuron dalam setiap lapisan GRU. Parameter ini dipilih berdasarkan kompleksitas data yang akan dipelajari.
  -  kernel_regularizer: Regularisasi L2 untuk mencegah overfitting dan mengontrol besar bobot.
  -  dropout: Untuk mengurangi overfitting dengan mengabaikan sebagian neuron.
  -  batch normalization: Membantu mempercepat pelatihan dengan menstabilkan distribusi input di dalam jaringan.
  -  learning_rate: Tingkat pembelajaran untuk mengontrol laju pembaruan bobot selama pelatihan.

- Kelebihan GRU:
  -  Struktur yang lebih sederhana dari LSTM karena hanya memiliki dua jenis gate (update dan reset) dibandingkan LSTM yang memiliki tiga jenis gate, menjadikannya lebih efisien dalam hal komputasi dan memori.
  -  GRU sering kali lebih cepat dilatih dibandingkan LSTM Karena memiliki lebih sedikit parameter.

- Kekurangan GRU:
  -  Meskipun lebih efisien GRU tidak selalu sekuat LSTM dalam menangani masalah jangka panjang yang sangat kompleks.
  -  Untuk beberapa jenis masalah spesifik yang memerlukan kontrol lebih rinci terhadap memori, LSTM bisa lebih unggul.

**Penerapan Callback dan Compile Model**

Callback adalah fungsi atau objek yang dipanggil secara otomatis selama proses pelatihan model pada titik tertentu. Callback digunakan untuk meningkatkan efisiensi pelatihan, mencegah overfitting, dan memastikan model yang dihasilkan adalah yang terbaik. Disini saya menggunakan 2 callback yaitu:
-  EarlyStopping yang berfungsi untuk menghentikan pelatihan ketika tidak ada peningkatan dalam performa model untuk menghindari overfitting dan pemborosan waktu komputasi.
-  ModelCheckpoint yang menyimpan model terbaik yang ditemukan selama pelatihan untuk memastikan bahwa model yang diekspor adalah model dengan performa terbaik berdasarkan val_loss.

Setelah persiapan selesai langkah yang harus dilakukan adalah melatih model, saya menggunakan 30 epoch, 10% validation, dan 32 batch size.

**Model Terbaik**

Berdasarkan proyek ini, saya mendapat bahwa LSTM lebih baik daripada GRU pada dataset prediksi polusi udara. Meskipun GRU lebih cepat dalam pelatihan dan lebih efisien secara komputasi, LSTM lebih baik dalam menangani data time series dengan ketergantungan jangka panjang yang lebih rumit. Oleh karena itu, pada kasus ini, dimana data menunjukkan ketergantungan temporal yang kuat, LSTM menghasilkan performa yang lebih baik dengan loss, MSE, dan RMSE yang lebih kecil, karena kemampuannya untuk mempertahankan dan mengelola informasi dalam jangka panjang dengan lebih baik daripada GRU. 


## Evaluation

**MSE (Mean Squared Error)**

MSE adalah metrik yang mengukur rata-rata kuadrat selisih antara nilai yang diprediksi dan nilai yang sebenarnya. Metrik ini memberikan penalti yang lebih besar untuk kesalahan yang lebih besar, sehingga model yang memiliki kesalahan yang lebih besar akan memberikan nilai MSE yang lebih tinggi.

Formula MSE:

![image](https://github.com/user-attachments/assets/ad16f7b5-3756-4fe6-9516-42b736f284a9)

Tahapan MSE:
-  Menghitung selisih untuk setiap data, MSE menghitung selisih antara nilai aktual (𝑦𝑖) dan nilai yang diprediksi (𝑦^).
-  Mengkuadratkan selisih antara nilai aktual dan nilai prediksi dikuadratkan untuk menghilangkan tanda negatif dan memberi penalti lebih besar pada kesalahan yang lebih besar.
-  Setelah semua selisih dikuadratkan, MSE mengambil rata-rata dari kesalahan kuadrat tersebut untuk memberikan satu nilai yang menggambarkan performa model secara keseluruhan.

Hasil Evaluasi
  - GRU Train MSE: 0.0086
  - GRU Test MSE: 0.0099
  - LSTM Train MSE: 0.0069
  - LSTM Test MSE: 0.0076

**RMSE (Root Mean Squared Error)**

RMSE adalah akar kuadrat dari MSE. Metrik ini memberikan penilaian dalam satuan yang sama dengan data asli, sehingga lebih mudah diinterpretasikan daripada MSE. RMSE juga lebih sensitif terhadap kesalahan besar dan memberikan gambaran yang jelas tentang seberapa besar kesalahan prediksi model.

Formula RMSE:

![image](https://github.com/user-attachments/assets/9314ca8b-1069-45ba-a375-211a7c468d8a)

Tahapan RMSE:
-  Setelah mendapatkan nilai MSE, RMSE dihitung dengan mengambil akar kuadrat dari MSE. Hal ini memberikan ukuran yang lebih intuitif karena RMSE berada pada satuan yang sama dengan data aslinya.

Hasil Evaluasi
  - GRU Train RMSE: 0.0928
  - GRU Test RMSE: 0.0994
  - LSTM Train RMSE: 0.0833
  - LSTM Test RMSE: 0.0869

**Analisis Hasil Evaluasi**
-  MSE pada LSTM lebih rendah daripada GRU baik pada data pelatihan maupun pengujian. Ini menunjukkan bahwa LSTM lebih akurat dalam memprediksi nilai konsentrasi polusi udara pada kedua set data.
-  RMSE yang lebih rendah pada LSTM menunjukkan bahwa kesalahan prediksi LSTM lebih kecil dan lebih konsisten dibandingkan dengan GRU.
-  model LSTM memiliki kinerja yang lebih baik, karena MSE dan RMSE pada LSTM lebih rendah daripada GRU, baik pada data pelatihan maupun data pengujian.
-  LSTM lebih efektif dalam menangani ketergantungan jangka panjang pada data time series. Oleh karena itu, ia dapat mempelajari pola yang lebih rumit dan lebih lama dalam data polusi udara dibandingkan dengan GRU, yang meskipun lebih cepat dan efisien, memiliki kemampuan lebih terbatas dalam mempertahankan informasi jangka panjang.
-  MSE dan RMSE pada LSTM menunjukkan bahwa meskipun kedua model ini memberikan hasil yang cukup baik, LSTM lebih stabil dan memberikan hasil yang lebih baik dalam hal akurasi prediksi.

**Prediksi Model**

LSTM :

![image](https://github.com/user-attachments/assets/27182a4c-5ce0-4d61-94fa-310efc459d94)

GRU :

![image](https://github.com/user-attachments/assets/893bde8c-53bd-4c0f-9106-fc6dbc6ef349)

**Kesimpulan**

Berdasarkan nilai MSE dan RMSE, model LSTM menunjukkan kinerja yang lebih baik dibandingkan GRU dalam memprediksi konsentrasi polusi udara. LSTM memberikan hasil yang lebih rendah dalam hal kesalahan kuadrat rata-rata dan akar kuadrat dari kesalahan tersebut, yang menunjukkan bahwa model ini lebih efektif dalam menangani dependensi jangka panjang dan lebih akurat dalam memprediksi data time series.
