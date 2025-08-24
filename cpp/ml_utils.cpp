#include "ml_utils.h"
#include <algorithm>
#include <random>
#include <map>
#include <numeric>
#include <iostream>

// String vektörünü sayısal veriye dönüştürme
DataPoint convert_to_numeric(const std::vector<std::string>& row, const std::vector<size_t>& feature_indices) {
    DataPoint numeric_row;
    for (auto idx : feature_indices) {
        if (idx < row.size()) {
            try {
                numeric_row.push_back(std::stod(row[idx]));
            } catch (const std::invalid_argument&) {
                numeric_row.push_back(0.0);  // Dönüşüm hatalıysa 0 ata
            }
        }
    }
    return numeric_row;
}

// Verileri feature ve target olarak ayırma
void split_data(const std::vector<std::vector<std::string>>& data, 
                DataSet& features, Labels& labels,
                const std::vector<size_t>& feature_indices,
                size_t target_index) {
    features.clear();
    labels.clear();
    
    // İlk satır başlık satırı, bu yüzden 1'den başla
    for (size_t i = 1; i < data.size(); ++i) {
        const auto& row = data[i];
        if (row.size() > target_index) {
            // Hedef sütunu sayıya dönüştür
            try {
                int label = std::stoi(row[target_index]);
                labels.push_back(label);
                
                // Özellikleri dönüştür
                features.push_back(convert_to_numeric(row, feature_indices));
            } catch (const std::invalid_argument&) {
                // Dönüşüm hatalıysa atla
            }
        }
    }
}

// Verileri eğitim ve test setlerine ayırma
void train_test_split(const DataSet& features, const Labels& labels,
                     DataSet& X_train, DataSet& X_test,
                     Labels& y_train, Labels& y_test,
                     double test_size) {
    if (features.size() != labels.size() || features.empty()) return;
    
    X_train.clear();
    X_test.clear();
    y_train.clear();
    y_test.clear();
    
    // Verileri karıştırmak için indexler oluştur
    std::vector<size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);  // 0, 1, 2, ...
    
    // İndexleri karıştır
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Test seti boyutunu hesapla
    size_t test_count = static_cast<size_t>(features.size() * test_size);
    
    // Eğitim ve test setlerini ayır
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        if (i < test_count) {
            X_test.push_back(features[idx]);
            y_test.push_back(labels[idx]);
        } else {
            X_train.push_back(features[idx]);
            y_train.push_back(labels[idx]);
        }
    }
}

// Verileri standartlaştırma (z-score normalizasyonu)
void standardize_data(DataSet& features, const DataSet& train_features) {
    if (train_features.empty() || features.empty()) return;
    
    size_t n_features = train_features[0].size();
    
    // Eğitim verilerinin ortalama ve standart sapmasını hesapla
    std::vector<double> mean(n_features, 0.0);
    std::vector<double> std_dev(n_features, 0.0);
    
    // Ortalamaları hesapla
    for (const auto& sample : train_features) {
        for (size_t j = 0; j < n_features; ++j) {
            mean[j] += sample[j];
        }
    }
    
    for (size_t j = 0; j < n_features; ++j) {
        mean[j] /= train_features.size();
    }
    
    // Standart sapmaları hesapla
    for (const auto& sample : train_features) {
        for (size_t j = 0; j < n_features; ++j) {
            std_dev[j] += (sample[j] - mean[j]) * (sample[j] - mean[j]);
        }
    }
    
    for (size_t j = 0; j < n_features; ++j) {
        std_dev[j] = std::sqrt(std_dev[j] / train_features.size());
        // Standart sapma 0 ise, bölme hatası oluşmaması için
        if (std_dev[j] < 1e-10) std_dev[j] = 1.0;
    }
    
    // Verileri standartlaştır
    for (auto& sample : features) {
        for (size_t j = 0; j < n_features && j < sample.size(); ++j) {
            sample[j] = (sample[j] - mean[j]) / std_dev[j];
        }
    }
}

// KNN sınıflandırıcısı
KNNClassifier::KNNClassifier(int k) : k(k) {}

double KNNClassifier::euclidean_distance(const DataPoint& a, const DataPoint& b) {
    double sum = 0.0;
    size_t min_size = std::min(a.size(), b.size());
    for (size_t i = 0; i < min_size; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void KNNClassifier::fit(const DataSet& X_train, const Labels& y_train) {
    training_data = X_train;
    training_labels = y_train;
}

int KNNClassifier::predict(const DataPoint& sample) {
    if (training_data.empty()) return -1;
    
    // Tüm eğitim örnekleri ile mesafeleri hesapla
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < training_data.size(); ++i) {
        double dist = euclidean_distance(sample, training_data[i]);
        distances.push_back({dist, training_labels[i]});
    }
    
    // Mesafelere göre sırala
    std::sort(distances.begin(), distances.end());
    
    // En yakın k komşuyu bul ve çoğunluk oylaması yap
    std::map<int, int> vote_count;
    for (size_t i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        vote_count[distances[i].second]++;
    }
    
    // En çok oy alan etiketi bul
    int max_votes = 0;
    int predicted_label = -1;
    for (const auto& [label, votes] : vote_count) {
        if (votes > max_votes) {
            max_votes = votes;
            predicted_label = label;
        }
    }
    
    return predicted_label;
}

Labels KNNClassifier::predict(const DataSet& samples) {
    Labels predictions;
    for (const auto& sample : samples) {
        predictions.push_back(predict(sample));
    }
    return predictions;
}

double KNNClassifier::accuracy(const DataSet& X_test, const Labels& y_test) {
    if (X_test.size() != y_test.size() || X_test.empty()) return 0.0;
    
    Labels predictions = predict(X_test);
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == y_test[i]) correct++;
    }
    
    return static_cast<double>(correct) / predictions.size();
}

// K değerini değiştirme
void KNNClassifier::set_k(int new_k) {
    k = new_k;
}

// Karmaşıklık matrisi (Confusion Matrix) hesaplama
std::vector<std::vector<int>> confusion_matrix(const Labels& y_true, const Labels& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) 
        return {{0, 0}, {0, 0}};
    
    // Binary sınıflandırma için 2x2 matris
    std::vector<std::vector<int>> cm(2, std::vector<int>(2, 0));
    
    for (size_t i = 0; i < y_true.size(); ++i) {
        int true_class = y_true[i];
        int pred_class = y_pred[i];
        
        if (true_class >= 0 && true_class < 2 && pred_class >= 0 && pred_class < 2) {
            cm[true_class][pred_class]++;
        }
    }
    
    return cm;
}

// Karmaşıklık matrisini yazdırma
void print_confusion_matrix(const std::vector<std::vector<int>>& cm) {
    std::cout << "\nKarmaşıklık Matrisi:" << std::endl;
    std::cout << "    | Tahmin:0 | Tahmin:1 |" << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Gerçek:0 |    " << cm[0][0] << "    |    " << cm[0][1] << "    |" << std::endl;
    std::cout << "Gerçek:1 |    " << cm[1][0] << "    |    " << cm[1][1] << "    |" << std::endl;
    
    // Metrikler hesaplama
    int tp = cm[1][1]; // True Positive
    int tn = cm[0][0]; // True Negative
    int fp = cm[0][1]; // False Positive
    int fn = cm[1][0]; // False Negative
    
    double accuracy = static_cast<double>(tp + tn) / (tp + tn + fp + fn);
    double precision = tp > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = tp > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double f1_score = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
    
    std::cout << "\nMetrikler:" << std::endl;
    std::cout << "Doğruluk (Accuracy): " << accuracy * 100.0 << "%" << std::endl;
    std::cout << "Kesinlik (Precision): " << precision * 100.0 << "%" << std::endl;
    std::cout << "Duyarlılık (Recall): " << recall * 100.0 << "%" << std::endl;
    std::cout << "F1-Score: " << f1_score * 100.0 << "%" << std::endl;
}

// Grid search ile en iyi K değerini bulma
int find_best_k(const DataSet& X_train, const Labels& y_train,
               const DataSet& X_val, const Labels& y_val,
               int k_min, int k_max) {
    double best_accuracy = 0.0;
    int best_k = k_min;
    
    std::cout << "\nEn iyi K değerini arıyorum..." << std::endl;
    std::cout << "K\tDoğruluk" << std::endl;
    std::cout << "-------------------" << std::endl;
    
    for (int k = k_min; k <= k_max; ++k) {
        KNNClassifier knn(k);
        knn.fit(X_train, y_train);
        double accuracy = knn.accuracy(X_val, y_val);
        
        std::cout << k << "\t" << accuracy * 100.0 << "%" << std::endl;
        
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_k = k;
        }
    }
    
    std::cout << "\nEn iyi K değeri: " << best_k << " (Doğruluk: " << best_accuracy * 100.0 << "%)" << std::endl;
    return best_k;
}
