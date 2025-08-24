#pragma once
#include <vector>
#include <string>
#include <cmath>

// Veri tipi tanımlamaları
using DataPoint = std::vector<double>;
using DataSet = std::vector<DataPoint>;
using Labels = std::vector<int>;

// String vektörünü sayısal veriye dönüştürme
DataPoint convert_to_numeric(const std::vector<std::string>& row, const std::vector<size_t>& feature_indices);

// Verileri feature ve target olarak ayırma
void split_data(const std::vector<std::vector<std::string>>& data, 
                DataSet& features, Labels& labels,
                const std::vector<size_t>& feature_indices,
                size_t target_index);

// Verileri eğitim ve test setlerine ayırma
void train_test_split(const DataSet& features, const Labels& labels,
                     DataSet& X_train, DataSet& X_test,
                     Labels& y_train, Labels& y_test,
                     double test_size = 0.25);

// Verileri standartlaştırma (z-score normalizasyonu)
void standardize_data(DataSet& features, const DataSet& train_features);

// Karmaşıklık matrisi (Confusion Matrix) hesaplama
std::vector<std::vector<int>> confusion_matrix(const Labels& y_true, const Labels& y_pred);

// Karmaşıklık matrisini yazdırma
void print_confusion_matrix(const std::vector<std::vector<int>>& cm);

// K-en yakın komşu sınıflandırıcısı
class KNNClassifier {
private:
    DataSet training_data;
    Labels training_labels;
    int k;
    
    double euclidean_distance(const DataPoint& a, const DataPoint& b);
    void print_distances(const std::vector<std::pair<double, int>>& distances, int k);

public:
    KNNClassifier(int k = 5);
    
    // Modeli eğitme
    void fit(const DataSet& X_train, const Labels& y_train);
    
    // Tahmin yapma
    int predict(const DataPoint& sample);
    
    // Bir veri seti için tahmin yapma
    Labels predict(const DataSet& samples);
    
    // Model doğruluğunu hesaplama
    double accuracy(const DataSet& X_test, const Labels& y_test);
    
    // K değerini değiştirme
    void set_k(int new_k);
};

// Grid search ile en iyi K değerini bulma
int find_best_k(const DataSet& X_train, const Labels& y_train,
               const DataSet& X_val, const Labels& y_val,
               int k_min, int k_max);
