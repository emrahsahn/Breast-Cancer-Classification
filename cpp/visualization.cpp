#include "visualization.h"
#include <algorithm>
#include <iomanip>
#include <cmath>

// Terminalde histogram çizme
void print_histogram(const std::vector<double>& data, const std::string& title, int width) {
    if (data.empty()) return;
    
    // Min ve max değerleri bul
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    
    // Aralıkları hesapla (10 aralık)
    int num_bins = 10;
    double bin_width = (max_val - min_val) / num_bins;
    
    // Histogram için sayma
    std::vector<int> bins(num_bins, 0);
    for (double value : data) {
        int bin = static_cast<int>((value - min_val) / bin_width);
        if (bin == num_bins) bin--; // Son değer için düzeltme
        bins[bin]++;
    }
    
    // Max frekansı bul
    int max_freq = *std::max_element(bins.begin(), bins.end());
    
    // Histogramı yazdır
    std::cout << "\n" << title << std::endl;
    std::cout << std::string(width + 20, '-') << std::endl;
    
    for (int i = 0; i < num_bins; i++) {
        double bin_start = min_val + i * bin_width;
        double bin_end = min_val + (i + 1) * bin_width;
        
        // Bin aralığını yazdır
        std::cout << std::fixed << std::setprecision(2) 
                  << "[" << bin_start << ", " << bin_end << ") | ";
        
        // Çubuğu çiz
        int bar_width = static_cast<int>(static_cast<double>(bins[i]) / max_freq * width);
        std::cout << std::string(bar_width, '#') << " " << bins[i] << std::endl;
    }
    
    std::cout << std::string(width + 20, '-') << std::endl;
}

// Çubuk grafiği çizme (kategorik veriler için)
void print_bar_chart(const std::vector<std::pair<std::string, double>>& data, const std::string& title, int width) {
    if (data.empty()) return;
    
    // Max değeri bul
    double max_val = 0.0;
    for (const auto& item : data) {
        max_val = std::max(max_val, item.second);
    }
    
    // Grafiği yazdır
    std::cout << "\n" << title << std::endl;
    std::cout << std::string(width + 20, '-') << std::endl;
    
    for (const auto& item : data) {
        // Etiketi yazdır
        std::cout << std::setw(15) << std::left << item.first << " | ";
        
        // Çubuğu çiz
        int bar_width = static_cast<int>((item.second / max_val) * width);
        std::cout << std::string(bar_width, '#') << " " << item.second << std::endl;
    }
    
    std::cout << std::string(width + 20, '-') << std::endl;
}

// ROC eğrisi için (Basit ascii grafiği)
void print_roc_curve(const std::vector<std::pair<double, double>>& points) {
    const int size = 25; // ASCII grafiğin boyutu
    
    // Grafiği oluştur (varsayılan olarak boşluk)
    std::vector<std::vector<char>> grid(size, std::vector<char>(size, ' '));
    
    // Köşeleri ve eksenleri işaretle
    grid[0][0] = '+';
    grid[0][size-1] = '+';
    grid[size-1][0] = '+';
    grid[size-1][size-1] = '+';
    
    // X ve Y eksenlerini çiz
    for (int i = 1; i < size-1; i++) {
        grid[i][0] = '|';
        grid[size-1][i] = '-';
    }
    
    // Noktaları işaretle
    for (const auto& point : points) {
        int x = static_cast<int>(point.first * (size-1));
        int y = size-1 - static_cast<int>(point.second * (size-1));
        
        // Sınırlar içinde kontrol
        if (x >= 0 && x < size && y >= 0 && y < size) {
            grid[y][x] = '*';
        }
    }
    
    // Rastgele tahmin çizgisini işaretle
    for (int i = 0; i < size; i++) {
        int j = size-1 - i;
        if (i < size && j < size && grid[j][i] == ' ') {
            grid[j][i] = '.';
        }
    }
    
    // Grafiği yazdır
    std::cout << "\nROC Eğrisi (TPR - FPR)" << std::endl;
    for (const auto& row : grid) {
        std::cout << " ";
        for (char c : row) {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "X: False Positive Rate (FPR)" << std::endl;
    std::cout << "Y: True Positive Rate (TPR)" << std::endl;
}
