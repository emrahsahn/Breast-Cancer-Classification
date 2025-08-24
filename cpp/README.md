# C++ ile Meme Kanseri Sınıflandırma Projesi

## CMakeLists.txt

Bu dosya, C++ projemizin derlemesi için gerekli olan CMAKE yapılandırmasını içerir.

```cmake
cmake_minimum_required(VERSION 3.10)
project(BreastCancerClassification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(breast_cancer_classification 
    main.cpp 
    data_utils.cpp 
    preprocess_utils.cpp
    ml_utils.cpp
    visualization.cpp
)
```

## Derleme ve Çalıştırma

Projeyi derlemek için aşağıdaki adımları izleyin:

```bash
# CMAKE klasörü oluştur
mkdir build
cd build

# Projeyi yapılandır
cmake ..

# Derle
cmake --build .

# Çalıştır
./breast_cancer_classification
```

## Proje Yapısı

- `main.cpp`: Ana program dosyası
- `data_utils.h/cpp`: Veri okuma fonksiyonları
- `preprocess_utils.h/cpp`: Veri ön işleme fonksiyonları
- `ml_utils.h/cpp`: Makine öğrenimi algoritmaları (KNN)
- `visualization.h/cpp`: Terminal tabanlı veri görselleştirme araçları

## Detaylar

Bu C++ uygulaması, Python'daki meme kanseri sınıflandırma projesinin C++ sürümüdür. Aşağıdaki özellikleri içerir:

1. CSV veri okuma
2. Veri ön işleme (gereksiz sütunları kaldırma, etiketleri dönüştürme)
3. Verileri standartlaştırma
4. K-En Yakın Komşu (KNN) algoritması ile sınıflandırma
5. Grid Search ile en iyi K değerini bulma
6. Karmaşıklık matrisi ve model performans metrikleri
7. Terminal tabanlı veri görselleştirme (histogramlar ve çubuk grafikler)
