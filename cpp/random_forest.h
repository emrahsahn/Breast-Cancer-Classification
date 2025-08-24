#pragma once
#include <vector>
#include <random>
#include <set>
#include <functional>
#include <algorithm>
#include <iostream>
#include <numeric>  // std::iota için gerekli

// Decision Tree sınıfı
class DecisionTree {
private:
    struct Node {
        bool isLeaf;
        int featureIndex;
        double threshold;
        int label;
        Node* left;
        Node* right;

        Node() : isLeaf(false), featureIndex(-1), threshold(0.0), label(-1), left(nullptr), right(nullptr) {}
        ~Node() {
            delete left;
            delete right;
        }
    };

    Node* root;
    int maxDepth;
    int minSamplesSplit;
    int maxFeatures;
    std::mt19937 rng;

    Node* buildTree(const std::vector<std::vector<double>>& X, const std::vector<int>& y, 
                   const std::vector<int>& indices, int depth) {
        Node* node = new Node();

        // Çoğunluk sınıfını bul
        int majority = findMajorityClass(y, indices);

        // Tüm örnekler aynı sınıfsa veya max derinliğe ulaşıldıysa veya çok az örnek kaldıysa yaprak düğüm oluştur
        if (depth >= maxDepth || indices.size() <= minSamplesSplit || isHomogeneous(y, indices)) {
            node->isLeaf = true;
            node->label = majority;
            return node;
        }

        // Random feature subset
        std::vector<int> featureIndices;
        for (size_t i = 0; i < X[0].size(); ++i) {
            featureIndices.push_back(i);
        }
        std::shuffle(featureIndices.begin(), featureIndices.end(), rng);
        featureIndices.resize(std::min(maxFeatures, static_cast<int>(featureIndices.size())));

        // En iyi bölünmeyi bul
        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestGini = 1.0;

        for (int feature : featureIndices) {
            // Tüm olası eşikleri topla
            std::vector<double> values;
            for (int idx : indices) {
                values.push_back(X[idx][feature]);
            }
            std::sort(values.begin(), values.end());

            // Eşikleri dene
            for (size_t i = 0; i < values.size() - 1; ++i) {
                double threshold = (values[i] + values[i + 1]) / 2.0;
                
                // Sol ve sağ bölünmeleri oluştur
                std::vector<int> leftIndices, rightIndices;
                for (int idx : indices) {
                    if (X[idx][feature] <= threshold) {
                        leftIndices.push_back(idx);
                    } else {
                        rightIndices.push_back(idx);
                    }
                }

                // Bölünme boş ise atla
                if (leftIndices.empty() || rightIndices.empty()) continue;

                // Gini safsızlığını hesapla
                double leftRatio = static_cast<double>(leftIndices.size()) / indices.size();
                double rightRatio = static_cast<double>(rightIndices.size()) / indices.size();
                double leftGini = calculateGini(y, leftIndices);
                double rightGini = calculateGini(y, rightIndices);
                double gini = leftRatio * leftGini + rightRatio * rightGini;

                if (gini < bestGini) {
                    bestGini = gini;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        // Bölünme bulunamadıysa yaprak düğüm oluştur
        if (bestFeature == -1) {
            node->isLeaf = true;
            node->label = majority;
            return node;
        }

        // Sol ve sağ bölünmeleri oluştur
        std::vector<int> leftIndices, rightIndices;
        for (int idx : indices) {
            if (X[idx][bestFeature] <= bestThreshold) {
                leftIndices.push_back(idx);
            } else {
                rightIndices.push_back(idx);
            }
        }

        // Düğümü doldur ve alt ağaçları oluştur
        node->featureIndex = bestFeature;
        node->threshold = bestThreshold;

        node->left = buildTree(X, y, leftIndices, depth + 1);
        node->right = buildTree(X, y, rightIndices, depth + 1);

        return node;
    }

    bool isHomogeneous(const std::vector<int>& y, const std::vector<int>& indices) {
        if (indices.empty()) return true;
        int first = y[indices[0]];
        for (size_t i = 1; i < indices.size(); ++i) {
            if (y[indices[i]] != first) return false;
        }
        return true;
    }

    int findMajorityClass(const std::vector<int>& y, const std::vector<int>& indices) {
        std::vector<int> counts(2, 0);
        for (int idx : indices) {
            counts[y[idx]]++;
        }
        return (counts[0] > counts[1]) ? 0 : 1;
    }

    double calculateGini(const std::vector<int>& y, const std::vector<int>& indices) {
        if (indices.empty()) return 0.0;
        
        std::vector<int> counts(2, 0);
        for (int idx : indices) {
            counts[y[idx]]++;
        }
        
        double gini = 1.0;
        double size = static_cast<double>(indices.size());
        for (int count : counts) {
            double prob = count / size;
            gini -= prob * prob;
        }
        
        return gini;
    }

    int predict(const Node* node, const std::vector<double>& x) const {
        if (node->isLeaf) {
            return node->label;
        }
        
        if (x[node->featureIndex] <= node->threshold) {
            return predict(node->left, x);
        } else {
            return predict(node->right, x);
        }
    }

public:
    DecisionTree(int maxDepth = 10, int minSamplesSplit = 2, int maxFeatures = -1) 
        : root(nullptr), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit), maxFeatures(maxFeatures), rng(std::random_device{}()) {}
    
    ~DecisionTree() {
        delete root;
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid input data");
        }
        
        if (maxFeatures <= 0) {
            maxFeatures = std::sqrt(X[0].size());
        }
        
        std::vector<int> indices(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            indices[i] = i;
        }
        
        root = buildTree(X, y, indices, 0);
    }

    int predict(const std::vector<double>& x) const {
        if (!root) {
            throw std::runtime_error("Model not trained");
        }
        return predict(root, x);
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<int> predictions;
        for (const auto& x : X) {
            predictions.push_back(predict(x));
        }
        return predictions;
    }
};

// Random Forest sınıfı
class RandomForest {
private:
    std::vector<DecisionTree> trees;
    int numTrees;
    int maxDepth;
    int minSamplesSplit;
    double sampleRatio;
    int maxFeatures;
    std::mt19937 rng;

public:
    RandomForest(int numTrees = 100, int maxDepth = 10, int minSamplesSplit = 2, 
                 double sampleRatio = 0.7, int maxFeatures = -1)
        : numTrees(numTrees), maxDepth(maxDepth), minSamplesSplit(minSamplesSplit),
          sampleRatio(sampleRatio), maxFeatures(maxFeatures), rng(std::random_device{}()) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            throw std::invalid_argument("Invalid input data");
        }
        
        if (maxFeatures <= 0) {
            maxFeatures = std::sqrt(X[0].size());
        }
        
        trees.resize(numTrees);
        int sampleSize = static_cast<int>(X.size() * sampleRatio);
        
        std::cout << "Random Forest eğitimi başlatılıyor (ağaç sayısı: " << numTrees << ")..." << std::endl;
        
        for (int t = 0; t < numTrees; ++t) {
            // Bootstrap örnekleme
            std::vector<std::vector<double>> sampleX;
            std::vector<int> sampleY;
            
            std::vector<int> indices(X.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            
            for (int i = 0; i < sampleSize; ++i) {
                int idx = indices[i % indices.size()];
                sampleX.push_back(X[idx]);
                sampleY.push_back(y[idx]);
            }
            
            // Ağacı eğit
            trees[t] = DecisionTree(maxDepth, minSamplesSplit, maxFeatures);
            trees[t].fit(sampleX, sampleY);
            
            if ((t + 1) % 10 == 0 || t == numTrees - 1) {
                std::cout << "  " << (t + 1) << " ağaç eğitildi." << std::endl;
            }
        }
    }

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        if (trees.empty()) {
            throw std::runtime_error("Model not trained");
        }
        
        std::vector<int> predictions(X.size(), 0);
        
        for (size_t i = 0; i < X.size(); ++i) {
            std::vector<int> votes(2, 0);
            
            for (const auto& tree : trees) {
                int prediction = tree.predict(X[i]);
                votes[prediction]++;
            }
            
            predictions[i] = (votes[0] > votes[1]) ? 0 : 1;
        }
        
        return predictions;
    }

    double accuracy(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
        if (X.size() != y.size() || X.empty()) {
            throw std::invalid_argument("Invalid input data");
        }
        
        std::vector<int> predictions = predict(X);
        
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == y[i]) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / predictions.size();
    }
};
