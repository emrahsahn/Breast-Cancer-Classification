#include <iostream>
#include <vector>
#include "data_utils.h"
#include "preprocess_utils.h"
#include "ml_utils.h"
#include "visualization.h"
#include "random_forest.h"

int main() {
    try {
        std::cout << "Breast Cancer Classification - C++ Version" << std::endl;
        // CSV dosyasını oku
        std::string filename = "C:\\Users\\sahin\\Documents\\GitHub\\Breast-Cancer-Classification\\data\\archive (1)\\data.csv";
        auto data = read_csv(filename);
        
        if (data.empty()) {
            std::cerr << "Veri okunamadı veya boş. Program sonlandırılıyor." << std::endl;
            return 1;
        }

        // Bazı veri öğelerini yazdıralım
        std::cout << "İlk satır (başlık): ";
        for (size_t i = 0; i < std::min(data[0].size(), size_t(5)); ++i) {
            std::cout << data[0][i] << " | ";
        }
        std::cout << "..." << std::endl;
        
        std::cout << "İkinci satır (ilk veri): ";
        for (size_t i = 0; i < std::min(data[1].size(), size_t(5)); ++i) {
            std::cout << data[1][i] << " | ";
        }
        std::cout << "..." << std::endl;

        // Özellik ve target indekslerini belirleyelim
        size_t id_index = 0;
        size_t diagnosis_index = 0;
        
        // Önce başlık indekslerini bulalım
        for (size_t i = 0; i < data[0].size(); ++i) {
            if (data[0][i] == "id") {
                id_index = i;
            } else if (data[0][i] == "diagnosis") {
                diagnosis_index = i;
            }
        }
        
        // Şimdi manuel olarak özellik indekslerini oluşturalım
        std::vector<size_t> feature_indices;
        for (size_t i = 0; i < data[0].size(); ++i) {
            // id ve diagnosis dışındaki tüm sütunlar özellik
            if (i != id_index && i != diagnosis_index) {
                feature_indices.push_back(i);
            }
        }
        
        // Standart hata akışına veri hakkında bilgi yazdır
        std::cerr << "Hedef sütun: " << data[0][diagnosis_index] << " (indeks: " << diagnosis_index << ")" << std::endl;
        std::cerr << "Özellik sayısı: " << feature_indices.size() << std::endl;
        
        // Verileri özellikler ve etiketlere ayır
        DataSet features;
        Labels labels;
        split_data(data, features, labels, feature_indices, diagnosis_index);
        
        std::cerr << "Örnek sayısı: " << features.size() << std::endl;
        std::cerr << "Etiket dağılımı: ";
        int count_0 = 0, count_1 = 0;
        for (auto label : labels) {
            if (label == 0) count_0++;
            else if (label == 1) count_1++;
        }
        std::cerr << "Class 0 (Benign): " << count_0 << ", Class 1 (Malignant): " << count_1 << std::endl;
        
        // Eğitim ve test setlerine ayır
        DataSet X_train, X_test;
        Labels y_train, y_test;
        train_test_split(features, labels, X_train, X_test, y_train, y_test, 0.2);
        
        std::cerr << "Eğitim seti boyutu: " << X_train.size() << std::endl;
        std::cerr << "Test seti boyutu: " << X_test.size() << std::endl;
        
        // Verileri standartlaştır
        standardize_data(X_train, X_train);
        standardize_data(X_test, X_train);
        
        // ********** KNN MODEL EĞİTİMİ VE DEĞERLENDİRMESİ **********
        std::cout << "\n\n====================================================================" << std::endl;
        std::cout << "                   KNN MODEL SONUÇLARI                              " << std::endl;
        std::cout << "====================================================================" << std::endl;
        
        // Öncelikle veri dağılımını analiz edelim
        std::cout << "\nEğitim seti etiket dağılımı:" << std::endl;
        int train_class0 = 0, train_class1 = 0;
        for (auto label : y_train) {
            if (label == 0) train_class0++;
            else train_class1++;
        }
        std::cout << "Sınıf 0 (Benign): " << train_class0 << " örnek (" 
                 << (static_cast<double>(train_class0) / y_train.size()) * 100.0 << "%)" << std::endl;
        std::cout << "Sınıf 1 (Malignant): " << train_class1 << " örnek (" 
                 << (static_cast<double>(train_class1) / y_train.size()) * 100.0 << "%)" << std::endl;
                 
        std::cout << "\nTest seti etiket dağılımı:" << std::endl;
        int test_class0 = 0, test_class1 = 0;
        for (auto label : y_test) {
            if (label == 0) test_class0++;
            else test_class1++;
        }
        std::cout << "Sınıf 0 (Benign): " << test_class0 << " örnek (" 
                 << (static_cast<double>(test_class0) / y_test.size()) * 100.0 << "%)" << std::endl;
        std::cout << "Sınıf 1 (Malignant): " << test_class1 << " örnek (" 
                 << (static_cast<double>(test_class1) / y_test.size()) * 100.0 << "%)" << std::endl;
                 
        // KNN modeli oluştur ve eğit - farklı k değerleri deneyelim
        std::cout << "\n*** Farklı k değerleri için KNN performansı ***" << std::endl;
        
        std::vector<int> k_values = {3, 5, 7, 9, 11, 13, 15};
        int best_k = 5;
        double best_accuracy = 0.0;
        
        for (int k : k_values) {
            KNNClassifier knn(k);
            knn.fit(X_train, y_train);
            double accuracy = knn.accuracy(X_test, y_test);
            std::cout << "k=" << k << " doğruluk: " << accuracy * 100.0 << "%" << std::endl;
            
            if (accuracy > best_accuracy) {
                best_accuracy = accuracy;
                best_k = k;
            }
        }
        
        std::cout << "\nEn iyi k değeri: " << best_k << " (doğruluk: " << best_accuracy * 100.0 << "%)" << std::endl;
        
        // En iyi KNN modeliyle devam edelim
        KNNClassifier knn(best_k);
        knn.fit(X_train, y_train);
        
        // Test setinde doğruluğu hesapla
        double knn_accuracy = knn.accuracy(X_test, y_test);
        std::cout << "KNN Doğruluk: " << knn_accuracy * 100.0 << "%" << std::endl;
        
        // Karmaşıklık matrisini hesapla ve yazdır
        Labels y_pred_knn = knn.predict(X_test);
        
        // Sınıf tahmin dağılımını göster
        int knn_class0 = 0, knn_class1 = 0;
        for (auto pred : y_pred_knn) {
            if (pred == 0) knn_class0++;
            else knn_class1++;
        }
        std::cout << "KNN Tahmin Dağılımı: Sınıf 0: " << knn_class0 << ", Sınıf 1: " << knn_class1 << std::endl;
        
        // Her bir tahmin sonucunu gerçek değerle karşılaştırarak yazdır
        std::cout << "\nKNN Detaylı Tahmin Sonuçları:" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "Örnek No | Gerçek Sınıf | Tahmin Edilen Sınıf | Doğru Tahmin" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        
        int knn_correctCount = 0;
        for (size_t i = 0; i < std::min(size_t(20), y_test.size()); ++i) {
            bool isCorrect = (y_test[i] == y_pred_knn[i]);
            if (isCorrect) knn_correctCount++;
            
            std::cout << i + 1 << "\t| " 
                      << (y_test[i] == 0 ? "Benign(0)" : "Malignant(1)") << "\t| "
                      << (y_pred_knn[i] == 0 ? "Benign(0)" : "Malignant(1)") << "\t\t| "
                      << (isCorrect ? "Evet ✓" : "Hayır ✗") << std::endl;
        }
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "İlk 20 örnek için doğruluk: " << (static_cast<double>(knn_correctCount) / std::min(size_t(20), y_test.size())) * 100.0 << "%" << std::endl;
        
        auto cm_knn = confusion_matrix(y_test, y_pred_knn);
        std::cout << "\nKNN Modeli için Confusion Matrix:" << std::endl;
        print_confusion_matrix(cm_knn);
        
        
        // ********** RANDOM FOREST MODEL EĞİTİMİ VE DEĞERLENDİRMESİ **********
        std::cout << "\n\n====================================================================" << std::endl;
        std::cout << "                RANDOM FOREST MODEL SONUÇLARI                       " << std::endl;
        std::cout << "====================================================================" << std::endl;
        
        // Random Forest modelini oluştur ve eğit
        std::cout << "\nRandom Forest modeli eğitiliyor..." << std::endl;
        RandomForest rf(100, 10, 2, 0.7, 5); // 100 ağaç, max derinlik 10, min örnek 2, %70 örnekleme, 5 özellik
        
        // Random Forest için veri formatını uyarla (std::vector<std::vector<double>> formatına)
        std::vector<std::vector<double>> X_train_rf, X_test_rf;
        std::vector<int> y_train_rf, y_test_rf;
        
        for (const auto& sample : X_train) {
            X_train_rf.push_back(sample);
        }
        
        for (const auto& sample : X_test) {
            X_test_rf.push_back(sample);
        }
        
        for (int label : y_train) {
            y_train_rf.push_back(label);
        }
        
        for (int label : y_test) {
            y_test_rf.push_back(label);
        }
        
        // Random Forest modelini eğit
        rf.fit(X_train_rf, y_train_rf);
        
        // Random Forest doğruluğunu hesapla
        double rf_accuracy = rf.accuracy(X_test_rf, y_test_rf);
        std::cout << "\n***************************************************" << std::endl;
        std::cout << "RANDOM FOREST MODELİ SONUÇLARI:" << std::endl;
        std::cout << "Random Forest Doğruluk: " << rf_accuracy * 100.0 << "%" << std::endl;
        
        // Random Forest tahminlerini al
        std::vector<int> y_pred_rf = rf.predict(X_test_rf);
        
        // Sınıf tahmin dağılımını göster
        int rf_class0 = 0, rf_class1 = 0;
        for (auto pred : y_pred_rf) {
            if (pred == 0) rf_class0++;
            else rf_class1++;
        }
        std::cout << "Random Forest Tahmin Dağılımı: Sınıf 0: " << rf_class0 << ", Sınıf 1: " << rf_class1 << std::endl;
        
        // Her bir tahmin sonucunu gerçek değerle karşılaştırarak yazdır
        std::cout << "\nRandom Forest Detaylı Tahmin Sonuçları:" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "Örnek No | Gerçek Sınıf | Tahmin Edilen Sınıf | Doğru Tahmin" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        
        int correctCount = 0;
        for (size_t i = 0; i < std::min(size_t(20), y_test_rf.size()); ++i) {
            bool isCorrect = (y_test_rf[i] == y_pred_rf[i]);
            if (isCorrect) correctCount++;
            
            std::cout << i + 1 << "\t| " 
                      << (y_test_rf[i] == 0 ? "Benign(0)" : "Malignant(1)") << "\t| "
                      << (y_pred_rf[i] == 0 ? "Benign(0)" : "Malignant(1)") << "\t\t| "
                      << (isCorrect ? "Evet ✓" : "Hayır ✗") << std::endl;
        }
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "İlk 20 örnek için doğruluk: " << (static_cast<double>(correctCount) / std::min(size_t(20), y_test_rf.size())) * 100.0 << "%" << std::endl;
        
        // Random Forest için confusion matrix
        std::vector<std::vector<int>> cm_rf(2, std::vector<int>(2, 0));
        for (size_t i = 0; i < y_test_rf.size(); ++i) {
            cm_rf[y_test_rf[i]][y_pred_rf[i]]++;
        }
        
        std::cout << "\nRandom Forest Modeli için Confusion Matrix:" << std::endl;
        print_confusion_matrix(cm_rf);
        std::cout << "====================================================================" << std::endl;
        
        // Model karşılaştırması
        std::cout << "\n\n====================================================================" << std::endl;
        std::cout << "                     MODEL KARŞILAŞTIRMASI                          " << std::endl;
        std::cout << "====================================================================" << std::endl;
        std::cout << "KNN Doğruluk: " << knn_accuracy * 100.0 << "%" << std::endl;
        std::cout << "Random Forest Doğruluk: " << rf_accuracy * 100.0 << "%" << std::endl;
        
        // İkisinin tahminlerini karşılaştır
        int both_correct = 0, both_wrong = 0, knn_correct_rf_wrong = 0, knn_wrong_rf_correct = 0;
        for (size_t i = 0; i < y_test.size(); ++i) {
            bool knn_correct = (y_test[i] == y_pred_knn[i]);
            bool rf_correct = (y_test_rf[i] == y_pred_rf[i]);
            
            if (knn_correct && rf_correct) both_correct++;
            else if (!knn_correct && !rf_correct) both_wrong++;
            else if (knn_correct && !rf_correct) knn_correct_rf_wrong++;
            else if (!knn_correct && rf_correct) knn_wrong_rf_correct++;
        }
        
        std::cout << "\nModellerin Karşılaştırmalı Performansı:" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "İki model de doğru tahmin: " << both_correct << " örnek (" 
                 << (static_cast<double>(both_correct) / y_test.size()) * 100.0 << "%)" << std::endl;
        std::cout << "İki model de yanlış tahmin: " << both_wrong << " örnek (" 
                 << (static_cast<double>(both_wrong) / y_test.size()) * 100.0 << "%)" << std::endl;
        std::cout << "KNN doğru, Random Forest yanlış: " << knn_correct_rf_wrong << " örnek (" 
                 << (static_cast<double>(knn_correct_rf_wrong) / y_test.size()) * 100.0 << "%)" << std::endl;
        std::cout << "KNN yanlış, Random Forest doğru: " << knn_wrong_rf_correct << " örnek (" 
                 << (static_cast<double>(knn_wrong_rf_correct) / y_test.size()) * 100.0 << "%)" << std::endl;
        std::cout << "----------------------------------------------------------------------" << std::endl;
        std::cout << "====================================================================" << std::endl;
        
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
    catch (const std::exception& e) {
        std::cerr << "Hata: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Bilinmeyen bir hata oluştu!" << std::endl;
        return 1;
    }
}

