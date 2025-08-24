#pragma once
#include <iostream>
#include <string>
#include <vector>

// Terminalde histogram çizme
void print_histogram(const std::vector<double>& data, const std::string& title, int width = 50);

// Çubuk grafiği çizme (kategorik veriler için)
void print_bar_chart(const std::vector<std::pair<std::string, double>>& data, const std::string& title, int width = 50);

// ROC eğrisi için (Basit ascii grafiği)
void print_roc_curve(const std::vector<std::pair<double, double>>& points);
