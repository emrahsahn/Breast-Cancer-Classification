#pragma once
#include <string>
#include <vector>

// Gereksiz sütunları kaldırma
void remove_columns(std::vector<std::vector<std::string>>& data, const std::vector<std::string>& columns_to_remove);

// Etiketleri dönüştürme ("M" -> 1, "B" -> 0)
void convert_target_column(std::vector<std::vector<std::string>>& data, const std::string& target_col);

// Sütun adını değiştirme
void rename_column(std::vector<std::vector<std::string>>& data, const std::string& old_name, const std::string& new_name);
