# Laporan Proyek Machine Learning - M. Raditya Adhirajasa

## Domain Proyek

Polusi udara dianggap sebagai masalah lingkungan yang mempengaruhi kesehatan manusia, baik dalam jangka panjang maupun jangka pendek. Sekitar 7 juta kematian per tahun disebabkan oleh paparan polusi udara dan bencana alam atmosfer lainnya (Alghieth et al., 2021). Salah satu polutan udara yang paling berbahaya adalah partikel halus yang dikenal sebagai PM2.5, yaitu partikel dengan diameter kurang dari 2,5 mikrometer. PM2.5 dapat dengan mudah terhirup dan menembus jauh ke dalam saluran pernapasan, sehingga dapat menyebabkan berbagai penyakit pernapasan dan kardiovaskular, bahkan meningkatkan risiko kematian dini.

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

[Alghieth, M., Alawaji, R., Saleh, S.H. & Alharbi, S., 2021. Air Pollution Forecasting Using Deep Learning. International Journal of Online and Biomedical Engineering (iJOE), 17(14), pp. 50–64.](https://www.researchgate.net/publication/357043273_Air_Pollution_Forecasting_Using_Deep_Learning) 

## Business Understanding

Polusi udara, khususnya partikel PM2.5, telah menjadi masalah global dengan dampak yang luas terhadap kesehatan manusia dan lingkungan. Di kota-kota besar seperti Beijing, konsentrasi PM2.5 sering kali melebihi ambang batas aman yang ditetapkan oleh WHO. Mengingat kompleksitas faktor yang memengaruhi kualitas udara, seperti kondisi cuaca, aktivitas manusia, dan pola angin sulit untuk memprediksi tingkat polusi secara manual. Oleh karena itu, pendekatan berbasis data yang memanfaatkan teknologi pembelajaran mesin diperlukan untuk membuat prediksi yang akurat. Pada proyek ini, proses klarifikasi masalah dilakukan dengan mengidentifikasi hubungan antara variabel cuaca dan tingkat polusi udara untuk membangun model prediksi berdasarkan data historis. Dengan memanfaatkan pendekatan forecasting, model diharapkan dapat memberikan prediksi tingkat PM2.5 dalam satu jam ke depan, sehingga dapat membantu masyarakat dan pemangku kepentingan dalam mengambil keputusan yang lebih tepat.

### Problem Statements

- Bagaimana hasil prediksi dapat digunakan untuk membantu pemerintah dan masyarakat dalam mengambil keputusan yang tepat untuk mengurangi paparan polusi udara?
- Berapa tingkat konsentrasi PM2.5 di waktu tertentu?

### Goals

- Menghasilkan informasi yang dapat digunakan untuk mendukung kebijakan pengurangan polusi udara di wilayah yang terkena dampak
- Membuat Model yang memprediksi tingkat konsentrasi PM2.5.

### Solution Statements
- Membangun Model Prediksi Time Series dengan LSTM dan GRU.

  Menggunakan dua algoritma model deep learning yaitu Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU). Keduanya dirancang untuk menangani data time series dengan dependensi jangka panjang. LSTM dan GRU akan digunakan untuk mempelajari pola historis dalam data polusi udara dan cuaca untuk memprediksi tingkat PM2.5 pada jam mendatang. Keunggulan dari LSTM dan GRU adalah kemampuannya untuk menangkap pola temporal yang kompleks serta mampu mengatasi masalah vanishing gradient.

- Evaluasi dengan Mean Squared Error (MSE) dan Root Mean Squared Error (RMSE)

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

  Dengan menampilkan boxplot pada data seperti yang dilakukan pada kode di atas digunakan untuk menganalisis distribusi dan karakteristik data dari berbagai fitur (kolom) numerik dalam dataset. Boxplot dapat membantu untuk memahami distribusi dan variabilitas data serta mendekteksi adanya outlier. Berdasarkan visualiasi tersebut dapat dilihat bahwa pollution, wnd_spd, snow, rain memiliki nilai yang tidak wajar, terutama pada pollution bisa melebihi 900. Namun, pada kasus kali ini data tersebut akan tetap digunakan karena ada kemungkinan memang pada waktu tersebut konsentrasi polusi sedang naik drastis.


  ![image](https://github.com/user-attachments/assets/eac83840-9cfc-4a90-957f-f892306e69ea)

  Menampilkan histogram seperti diatas dapat membantu untuk memahami jumlah data dari distribusi nya secara jelas dari setiap kolom. Berdasarkan visualiasi tersebut menunjukkan bahwa pada pollution umumnya konsentrasi polusi dibawah 50. Kondisi titik embun atau dew cenderung konsisten di kisaran tertentu, kemungkinan menunjukkan lingkungan dengan kelembapan relatif yang stabil. Temperatur dan tekanan menunjukkan perubahan signifikan, mungkin mencerminkan variasi suhu siang dan malam atau musim tertentu. Angin didominasi dari arah NW atau barat laut. Umumnya kecepatan angin tergolong rendah namun ada kemungkinan kecil kecepatan angin menjadi ekstrem

## Data Preparation

- Data Transform

  - Encoding Data

    Tahap ini menerapkan label encoding pada data wnd_dir bertipe data object/kategorikal. Encode data adalah proses mengubah data ke dalam format yang dapat dipahami oleh komputer, terutama model machine learning. Encoding biasanya dilakukan pada data yang tidak berbentuk numerik, seperti data kategorikal atau teks, agar dapat diolah oleh algoritma yang hanya menerima input berupa angka/numerik.

  - Ubah datetime

    Tahap ini mengubah data date menjadi datetime menjadi bentuk datetime agar dapat dikenali sebagai data tanggal dan waktu. Setelah itu mengatur kolom date sebagai index untuk mempermudah analisis berbasis waktu.
  
  - Normalisasi data

    Tahap ini menggunakan Min-Max Scalar. Min-Max Scaler adalah metode normalisasi yang digunakan untuk mengubah nilai fitur dalam dataset ke rentang tertentu, antara 0 dan 1. Teknik ini memastikan bahwa semua fitur memiliki skala yang sama, tanpa mengubah distribusi datanya. Normalisasi data membantu algoritma machine learning, terutama neural networks seperti LSTM dan GRU, untuk bekerja lebih efisien dengan skala data yang seragam. Ini mengurangi risiko bias terhadap fitur dengan nilai yang lebih besar.

- Split Data

  - Transform input dan target data

    Pada tahap ini menjadi input (X) dan target (y) berdasarkan jumlah langkah waktu (n_past) dan jumlah prediksi di masa depan (n_future). lalu membentuk data untuk pelatihan dan pengujian. Proses ini memastikan bahwa data siap untuk pelatihan dan evaluasi dengan model seperti LSTM atau GRU.


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

Callback adalah fungsi atau objek yang dipanggil secara otomatis selama proses pelatihan model pada titik tertentu. Callback digunakan untuk meningkatkan efisiensi pelatihan, mencegah overfitting, dan memastikan model yang dihasilkan adalah yang terbaik. Dengan menggunakan 2 callback yaitu:
-  EarlyStopping yang berfungsi untuk menghentikan pelatihan ketika tidak ada peningkatan dalam performa model untuk menghindari overfitting dan pemborosan waktu komputasi.
-  ModelCheckpoint yang menyimpan model terbaik yang ditemukan selama pelatihan untuk memastikan bahwa model yang diekspor adalah model dengan performa terbaik berdasarkan val_loss.

Setelah persiapan selesai langkah yang harus dilakukan adalah melatih model, dengan menggunakan 30 epoch, 10% validation, dan 32 batch size.

**Model Terbaik**

Berdasarkan proyek ini, LSTM lebih baik daripada GRU pada dataset prediksi polusi udara. Meskipun GRU lebih cepat dalam pelatihan dan lebih efisien secara komputasi, LSTM lebih baik dalam menangani data time series dengan ketergantungan jangka panjang yang lebih rumit. Oleh karena itu, pada kasus ini, dimana data menunjukkan ketergantungan temporal yang kuat, LSTM menghasilkan performa yang lebih baik dengan loss, MSE, dan RMSE yang lebih kecil, karena kemampuannya untuk mempertahankan dan mengelola informasi dalam jangka panjang dengan lebih baik daripada GRU. 


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
  - GRU Train MSE: 0.0091
  - GRU Test MSE: 0.0125
  - LSTM Train MSE: 0.0068
  - LSTM Test MSE: 0.0077

**RMSE (Root Mean Squared Error)**

RMSE adalah akar kuadrat dari MSE. Metrik ini memberikan penilaian dalam satuan yang sama dengan data asli, sehingga lebih mudah diinterpretasikan daripada MSE. RMSE juga lebih sensitif terhadap kesalahan besar dan memberikan gambaran yang jelas tentang seberapa besar kesalahan prediksi model.

Formula RMSE:

![image](https://github.com/user-attachments/assets/9314ca8b-1069-45ba-a375-211a7c468d8a)

Tahapan RMSE:
-  Setelah mendapatkan nilai MSE, RMSE dihitung dengan mengambil akar kuadrat dari MSE. Hal ini memberikan ukuran yang lebih intuitif karena RMSE berada pada satuan yang sama dengan data aslinya.

Hasil Evaluasi
  - GRU Train RMSE: 0.0955
  - GRU Test RMSE: 0.1118
  - LSTM Train RMSE: 0.0829
  - LSTM Test RMSE: 0.0881

**Analisis Hasil Evaluasi**
-  MSE pada LSTM lebih rendah daripada GRU baik pada data pelatihan maupun pengujian. Ini menunjukkan bahwa LSTM lebih akurat dalam memprediksi nilai konsentrasi polusi udara pada kedua set data.
-  RMSE yang lebih rendah pada LSTM menunjukkan bahwa kesalahan prediksi LSTM lebih kecil dan lebih konsisten dibandingkan dengan GRU.
-  model LSTM memiliki kinerja yang lebih baik, karena MSE dan RMSE pada LSTM lebih rendah daripada GRU, baik pada data pelatihan maupun data pengujian.
-  LSTM lebih efektif dalam menangani ketergantungan jangka panjang pada data time series. Oleh karena itu, ia dapat mempelajari pola yang lebih rumit dan lebih lama dalam data polusi udara dibandingkan dengan GRU, yang meskipun lebih cepat dan efisien, memiliki kemampuan lebih terbatas dalam mempertahankan informasi jangka panjang.
-  MSE dan RMSE pada LSTM menunjukkan bahwa meskipun kedua model ini memberikan hasil yang cukup baik, LSTM lebih stabil dan memberikan hasil yang lebih baik dalam hal akurasi prediksi.

**Prediksi Model**

LSTM :

![image](https://github.com/user-attachments/assets/5ae48236-166e-4a3a-8c4b-262fe85c8f10)

GRU :

![image](https://github.com/user-attachments/assets/ed7aa907-a4fe-4260-9e83-864111871d36)

**Kesimpulan**

- Problem Statement

  Hasil prediksi dapat digunakan untuk memberikan informasi yang lebih baik kepada pemerintah dan masyarakat mengenai waktu dan lokasi dengan tingkat polusi udara yang tinggi. Dengan informasi ini, kebijakan untuk mengurangi paparan polusi seperti penutupan jalan, pembatasan aktivitas industri, atau pemberian peringatan dini kepada masyarakat dapat diterapkan. Model prediksi ini juga dapat digunakan untuk memperkirakan konsentrasi PM2.5 pada waktu tertentu, memberikan proyeksi yang lebih tepat tentang kualitas udara berdasarkan data historis dan kondisi cuaca saat itu.

- Goals
  
  Proyek ini memberikan data yang dapat digunakan oleh pengambil kebijakan untuk merencanakan tindakan pengurangan polusi dan meminimalkan dampaknya terhadap kesehatan masyarakat serta membangun model machine learning yang dapat memberikan prediksi yang akurat mengenai tingkat polusi udara PM2.5 berdasarkan data historis dan cuaca yang ada.

- Solution Statement
  
  Dengan membangun model prediksi menggunakan LSTM dan GRU dan mengevaluasinya dengan metrik MSE dan RMSE, proyek ini dapat memberikan prediksi yang lebih akurat mengenai tingkat polusi udara PM2.5 di masa depan, yang sangat berguna dalam pengambilan keputusan oleh pemerintah dan masyarakat untuk mengurangi dampak polusi udara. Evaluasi dengan MSE dan RMSE memastikan bahwa model yang dibangun mampu memberikan prediksi yang lebih dekat dengan nilai aktual dan lebih sensitif terhadap kesalahan besar. Hasil evaluasi juga menunjukkan model LSTM memiliki kinerja lebih baik dibandingkan GRU, hal ini dikarenakan LSTM lebih efektif dalam menangani ketergantungan jangka panjang pada data time series.


