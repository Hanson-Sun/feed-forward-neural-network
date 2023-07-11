#pragma once

#include <iostream>
#include <vector>
#include "matrix.h"

// class Matrix
// {
// public:
//     Matrix(const std::vector<std::vector<double>> &g);
//     std::vector<double> operator[](int i) const;
//     std::vector<double> &operator[](int i);
//     int width() const;
//     int height() const;
//     void push(std::vector<double> v);
//     void print() const;
//     void transpose();
//     static Matrix generateRandom(int width, int height);
//     static Matrix generateEmptyCopy(const Matrix &v, double fill = 0);
//     static Matrix outerProduct(const nVector &v1, const nVector &v2);

// private:
//     std::vector<std::vector<double>> grid;
// };

// Matrix operator+(const Matrix &mat1, const Matrix &mat2);
// Matrix operator-(const Matrix &mat1, const Matrix &mat2);
// Matrix operator*(const Matrix &mat1, const Matrix &mat2);
// nVector operator*(const Matrix &mat1, const nVector &vect);

// namespace Math
// {
//     class mVector : public Matrix
//     {
//     public:
//         nVector(const std::vector<double> &v);
//         std::vector<double> getVals() const;
//         int size() const;

//         double operator[](int i) const;
//         double &operator[](int i);
//         Matrix &operator+=(const Matrix &v);
//         Matrix operator*(double x);
//         Matrix &operator*=(double x);
//         Matrix &operator+=(double d);

//         void push(double d);
//         void print() const;
//         static Matrix generateRandom(int size);
//         static Matrix generateEmptyCopy(const nVector &v, double fill = 0);
//         double magSqr();

//     private:
//         std::vector<double> grid;
//     };


//     nVector operator*(const Matrix &vec1, double x);
// };


