#include "data_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<std::vector<std::string>> read_csv(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Hata: '" << filename << "' dosyası açılamadı!" << std::endl;
        return data; // Boş veri döndür
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            // Sadece başlangıç ve sonundaki tırnak işaretlerini kaldır
            if (!cell.empty() && cell.front() == '"' && cell.back() == '"') {
                cell = cell.substr(1, cell.length() - 2);
            }
            row.push_back(cell);
        }
        data.push_back(row);
    }
    
    std::cout << "CSV dosyası okundu. Satır sayısı: " << data.size() << std::endl;
    return data;
}
