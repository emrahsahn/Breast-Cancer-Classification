#include "ml_utils.h"
#include <algorithm>
#include <random>
#include <map>
#include <numeric>
#include <iostream>

DataPoint convert_to_numeric(const std::vector<std::string>& row, const std::vector<size_t>& feature_indices) {
    DataPoint numeric_row;
    for (auto idx : feature_indices) {
        if (idx < row.size()) {
            try {
                // Temizlenmiş değeri dönüştürme
                numeric_row.push_back(std::stod(row[idx]));
            } catch (const std::invalid_argument& e) {
                // Dönüşüm başarısız olursa, "M"/"B" için özel işlem yap
                if (row[idx] == "M") {
                    numeric_row.push_back(1.0);
                } else if (row[idx] == "B") {
                    numeric_row.push_back(0.0);
                } else {
                    std::cerr << "Dönüşüm hatası: " << e.what() << " - değer: '" << row[idx] << "'" << std::endl;
                    numeric_row.push_back(0.0);  // Dönüşüm hatalıysa 0 ata
                }
            } catch (const std::out_of_range& e) {
                std::cerr << "Aralık dışı hatası: " << e.what() << " - değer: '" << row[idx] << "'" << std::endl;
                numeric_row.push_back(0.0);  // Dönüşüm hatalıysa 0 ata
            }
        } else {
            std::cerr << "İndeks hata: " << idx << " >= " << row.size() << std::endl;
            numeric_row.push_back(0.0);  // İndeks sınırları dışındaysa 0 ata
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
    
    std::cout << "Verileri özelliklere ve etiketlere ayırıyorum..." << std::endl;
    std::cout << "Sütun sayısı: " << (data.empty() ? 0 : data[0].size()) << std::endl;
    std::cout << "Özellik indeksleri: ";
    for (auto idx : feature_indices) std::cout << idx << " ";
    std::cout << std::endl;
    std::cout << "Hedef indeksi: " << target_index << std::endl;
    
    // İlk satır başlık satırı, bu yüzden 1'den başla
    for (size_t i = 1; i < data.size(); ++i) {
        const auto& row = data[i];
        if (row.size() > target_index) {
            try {
                // Hedef sütun değerini al
                int label = -1;
                std::string target_value = row[target_index];
                
                // Şimdi değeri kontrol et
                if (target_value == "M") {
                    label = 1;  // Malignant (kötü huylu) -> 1
                } else if (target_value == "B") {
                    label = 0;  // Benign (iyi huylu) -> 0
                } else {
                    try {
                        // Eğer sayısal bir değerse doğrudan dönüştür
                        label = std::stoi(target_value);
                    } catch (...) {
                        std::cerr << "Bilinmeyen etiket değeri: " << target_value << " (satır " << i << ")" << std::endl;
                        continue;  // Bu satırı atla
                    }
                }
                
                labels.push_back(label);
                
                // Özellikleri dönüştür
                features.push_back(convert_to_numeric(row, feature_indices));
            } catch (const std::exception& e) {
                // Dönüşüm hatalıysa atla
                std::cerr << "Veri dönüşüm hatası (satır " << i << "): " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Özellikler ve etiketler ayrıldı. Örnek sayısı: " << features.size() << std::endl;
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
    
    std::cout << "Verileri standartlaştırıyorum..." << std::endl;
    std::cout << "Örnek sayısı: " << features.size() << ", Özellik sayısı: " << n_features << std::endl;
    
    // Eğitim verilerinin ortalama ve standart sapmasını hesapla
    std::vector<double> mean(n_features, 0.0);
    std::vector<double> std_dev(n_features, 0.0);
    
    // Ortalamaları hesapla
    for (const auto& sample : train_features) {
        for (size_t j = 0; j < n_features && j < sample.size(); ++j) {
            mean[j] += sample[j];
        }
    }
    
    for (size_t j = 0; j < n_features; ++j) {
        mean[j] /= train_features.size();
    }
    
    // Standart sapmaları hesapla
    for (const auto& sample : train_features) {
        for (size_t j = 0; j < n_features && j < sample.size(); ++j) {
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
    
    std::cout << "Standartlaştırma tamamlandı." << std::endl;
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

// Debugging amaçlı mesafeleri yazdıralım
void KNNClassifier::print_distances(const std::vector<std::pair<double, int>>& distances, int k) {
    std::cout << "K=" << k << " için mesafeler ve sınıflar:" << std::endl;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        std::cout << "Komşu " << i+1 << ": Mesafe = " << distances[i].first 
                  << ", Sınıf = " << (distances[i].second == 0 ? "Benign(0)" : "Malignant(1)") << std::endl;
    }
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
    
    // Debugging: Mesafeleri ve sınıfları yazdır
    print_distances(distances, k);
    
    // Sınıf dağılımını kontrol et
    int class0_count = 0, class1_count = 0;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); i++) {
        if (distances[i].second == 0) class0_count++;
        else if (distances[i].second == 1) class1_count++;
    }
    
    std::cout << "En yakın " << k << " komşuda: Sınıf 0: " << class0_count 
              << ", Sınıf 1: " << class1_count << std::endl;
    
    // Mesafe ağırlıklı oylama
    std::map<int, double> weighted_votes;
    weighted_votes[0] = 0.0;  // Sınıf 0 için başlangıç değeri
    weighted_votes[1] = 0.0;  // Sınıf 1 için başlangıç değeri
    
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); i++) {
        // Mesafe 0'a yakınsa büyük bir değer kullan, 0 olamaz
        double weight = 1.0 / (distances[i].first + 0.0001);
        weighted_votes[distances[i].second] += weight;
    }
    
    // En çok oy alan etiketi bul
    std::cout << "K=" << k << " için ağırlıklı oylar:" << std::endl;
    for (const auto& [label, votes] : weighted_votes) {
        std::cout << "Sınıf " << label << ": " << votes << " ağırlıklı oy" << std::endl;
    }
    
    // En yüksek ağırlıklı oyu alan sınıfı bul
    double max_votes = 0.0;
    int predicted_class = 0; // Eşitlik durumunda sınıf 0'ı tercih et
    
    for (const auto& vote : weighted_votes) {
        if (vote.second > max_votes) {
            max_votes = vote.second;
            predicted_class = vote.first;
        } else if (std::abs(vote.second - max_votes) < 1e-10 && vote.first == 0) {
            // Eşitlik durumunda sınıf 0'ı tercih et
            predicted_class = 0;
        }
    }
    
    std::cout << "Tahmin edilen sınıf: " << predicted_class << "\n\n";
    return predicted_class;
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
    std::cout << "                | Predicted:      | Predicted:      |" << std::endl;
    std::cout << "                | Benign (Class 0)| Malignant (Class 1)" << std::endl;
    std::cout << "----------------+----------------+------------------" << std::endl;
    std::cout << "Actual: Benign  | " << cm[0][0] << " (True Neg)    | " << cm[0][1] << " (False Pos)" << std::endl;
    std::cout << "Actual: Malignant | " << cm[1][0] << " (False Neg)   | " << cm[1][1] << " (True Pos)" << std::endl;
    
    // Metrikler hesaplama
    int tp = cm[1][1]; // True Positive
    int tn = cm[0][0]; // True Negative
    int fp = cm[0][1]; // False Positive
    int fn = cm[1][0]; // False Negative
    
    double accuracy = static_cast<double>(tp + tn) / (tp + tn + fp + fn);
    double precision = tp > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = tp > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double f1_score = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
    
    std::cout << "\nPerformans Metrikleri:" << std::endl;
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
