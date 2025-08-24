cmake_minimum_required(VERSION 3.10)
project(BreastCancerClassification)

# C++ standart sürümünü belirle
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Derleme optimizasyonları
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Tüm kaynak dosyalarını ekle
set(SOURCES
    main.cpp
    data_utils.cpp
    preprocess_utils.cpp
    ml_utils.cpp
    visualization.cpp
)

# Başlık dosyalarının bulunduğu dizinleri belirt
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Yürütülebilir dosyayı oluştur
add_executable(breast_cancer_classification ${SOURCES})
