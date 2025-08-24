#include <iostream>
#include <vector>
#include "data_utils.h"
#include "preprocess_utils.h"
#include "ml_utils.h"
#include "visualization.h"

int main() {
    std::cout << "Breast Cancer Classification - C++ Version" << std::endl;
    // CSV dosyasını oku
    std::string filename = "..\\data\\archive (1)\\data.csv";
    auto data = read_csv(filename);

    // Gereksiz sütunları kaldır
    remove_columns(data, {"Unnamed: 32", "id"});

    // Sütun adını değiştir
    rename_column(data, "diagnosis", "target");

    // Etiketleri dönüştür
    convert_target_column(data, "target");

    std::cout << "Veri ön işleme tamamlandı. Satır sayısı: " << data.size() << std::endl;
    
    // Özellik indekslerini belirleme
    std::vector<size_t> feature_indices;
    // Tüm sütunlar (target hariç) özellik olarak kullanılacak
    for (size_t i = 0; i < data[0].size(); ++i) {
        if (data[0][i] != "target") {
            feature_indices.push_back(i);
        }
    }
    
    // Target indeksini bul
    size_t target_index = 0;
    for (size_t i = 0; i < data[0].size(); ++i) {
        if (data[0][i] == "target") {
            target_index = i;
            break;
        }
    }
    
    // Verileri özellikler ve etiketlere ayır
    DataSet features;
    Labels labels;
    split_data(data, features, labels, feature_indices, target_index);
    
    // Eğitim ve test setlerine ayır
    DataSet X_train, X_test;
    Labels y_train, y_test;
    train_test_split(features, labels, X_train, X_test, y_train, y_test, 0.25);
    
    // Verileri standartlaştır
    standardize_data(X_train, X_train);
    standardize_data(X_test, X_train);
    
    // KNN modeli oluştur ve eğit
    KNNClassifier knn(5);
    knn.fit(X_train, y_train);
    
    // Test setinde doğruluğu hesapla
    double accuracy = knn.accuracy(X_test, y_test);
    std::cout << "KNN Doğruluk: " << accuracy * 100.0 << "%" << std::endl;
    
    // En iyi K değerini bul
    int best_k = find_best_k(X_train, y_train, X_test, y_test, 1, 20);
    
    // En iyi K ile yeniden eğit
    knn.set_k(best_k);
    knn.fit(X_train, y_train);
    
    // Karmaşıklık matrisini hesapla ve yazdır
    Labels y_pred = knn.predict(X_test);
    auto cm = confusion_matrix(y_test, y_pred);
    print_confusion_matrix(cm);
    
    // Görselleştirmeler
    
    // 1. Feature dağılımlarının histogramını çiz (örnek olarak ilk 3 özellik)
    for (size_t i = 0; i < std::min(size_t(3), X_train[0].size()); ++i) {
        std::vector<double> feature_values;
        for (const auto& sample : X_train) {
            feature_values.push_back(sample[i]);
        }
        print_histogram(feature_values, "Özellik " + std::to_string(i+1) + " Dağılımı");
    }
    
    // 2. K değerleri ve doğrulukları için çubuk grafik
    std::vector<std::pair<std::string, double>> k_accuracy;
    for (int k = 1; k <= 5; ++k) {
        KNNClassifier knn_temp(k);
        knn_temp.fit(X_train, y_train);
        double acc = knn_temp.accuracy(X_test, y_test);
        k_accuracy.push_back({"K=" + std::to_string(k), acc});
    }
    print_bar_chart(k_accuracy, "K Değeri ve Doğruluk İlişkisi");
    
    return 0;
}
