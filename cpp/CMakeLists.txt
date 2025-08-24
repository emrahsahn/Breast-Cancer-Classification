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
