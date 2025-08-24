#include "data_utils.h"
#include <iostream>
#include <algorithm>

// Gereksiz sütunları kaldırma
void remove_columns(std::vector<std::vector<std::string>>& data, const std::vector<std::string>& columns_to_remove) {
    if (data.empty()) return;
    std::vector<size_t> indices;
    for (const auto& col : columns_to_remove) {
        auto it = std::find(data[0].begin(), data[0].end(), col);
        if (it != data[0].end()) {
            indices.push_back(std::distance(data[0].begin(), it));
        }
    }
    // Sütunları sondan başa silmek daha güvenli
    std::sort(indices.rbegin(), indices.rend());
    for (auto& row : data) {
        for (auto idx : indices) {
            if (idx < row.size()) row.erase(row.begin() + idx);
        }
    }
}

// Etiketleri dönüştürme ("M" -> 1, "B" -> 0)
void convert_target_column(std::vector<std::vector<std::string>>& data, const std::string& target_col) {
    if (data.empty()) return;
    auto it = std::find(data[0].begin(), data[0].end(), target_col);
    if (it == data[0].end()) return;
    size_t idx = std::distance(data[0].begin(), it);
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i][idx] == "M") data[i][idx] = "1";
        else if (data[i][idx] == "B") data[i][idx] = "0";
    }
}

// Sütun adını değiştirme
void rename_column(std::vector<std::vector<std::string>>& data, const std::string& old_name, const std::string& new_name) {
    if (data.empty()) return;
    auto it = std::find(data[0].begin(), data[0].end(), old_name);
    if (it != data[0].end()) {
        *it = new_name;
    }
}
