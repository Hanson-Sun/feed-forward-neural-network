
// #if false
// #pragma once

// #include <iostream>
// #include <vector>
// #include <thread>
// #include <chrono>
// #include "vector.h"

// namespace no
// {
//     class Matrix
//     {
//     public:
//         Matrix(const std::vector<std::vector<double>> &g);
//         std::vector<double> operator[](int i) const;
//         std::vector<double> &operator[](int i);
//         Matrix &operator+=(const Matrix &m);
//         Matrix &operator-=(const Matrix &m);
//         int width() const;
//         int height() const;
//         void push(std::vector<double> v);
//         void print() const;
//         Matrix transpose();
//         static Matrix generateRandom(int width, int height);
//         static Matrix generateEmptyCopy(const Matrix &v, double fill = 0);
//         static Matrix outerProduct(const nVector &v1, const nVector &v2);

//     private:
//         std::vector<std::vector<double>> grid;
//         int w;
//         int h;
//     };

//     Matrix operator+(const Matrix &mat1, const Matrix &mat2);
//     Matrix operator-(const Matrix &mat1, const Matrix &mat2);
//     Matrix operator*(const Matrix &mat1, const Matrix &mat2);
//     Matrix operator*(const Matrix &mat1, const double &n);
//     nVector operator*(const Matrix &mat1, const nVector &vect);
// };

// #endif