// #if false
// #pragma once

// #include <iostream>
// #include <vector>

// namespace Math
// {
//     class nVector
//     {
//     public:
//         nVector(const std::vector<double> &v);
//         std::vector<double> getVals() const;
//         int size() const;
//         double operator[](int i) const;
//         double &operator[](int i);
//         nVector &operator+=(const nVector &v);
//         nVector &operator-=(const nVector &v);

//         nVector operator*(double x);
//         nVector &operator*=(double x);
//         nVector &operator+=(double d);

//         void push(double d);
//         void print() const;
//         static nVector generateRandom(int size);
//         static nVector generateEmptyCopy(const nVector &v, double fill = 0);
//         static nVector pairWiseMult(const nVector v1, const nVector v2);
//         double magSqr();

//     private:
//         std::vector<double> vals;
//     };

//     nVector operator+(const nVector &vec1, const nVector &vec2);
//     nVector operator-(const nVector &vec1, const nVector &vec2);
//     double operator*(const nVector &vec1, const nVector &vec2);
//     nVector operator*(const nVector &vec1, double x);
// };

// #endif