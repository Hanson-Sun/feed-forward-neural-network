
#include <iostream>
#include <vector>
#include "headers/vector.h"

namespace Math
{
    nVector::nVector(const std::vector<double> &v)
    {
        vals = v;
    }

    std::vector<double> nVector::getVals() const
    {
        return vals;
    }

    int nVector::size() const
    {
        return vals.size();
    }

    double nVector::operator[](int i) const { return vals[i]; }
    double &nVector::operator[](int i) { return vals[i]; }

    nVector &nVector::operator+=(const nVector &v)
    {
        if (vals.size() != v.size())
            throw std::length_error("Vectors must have the same size to add them.");

        for (int i = 0; i < vals.size(); i++)
        {
            vals[i] += v[i];
        }
        return *this;
    }

    nVector &nVector::operator-=(const nVector &v)
    {
        if (vals.size() != v.size())
            throw std::length_error("Vectors must have the same size to add them.");

        for (int i = 0; i < vals.size(); i++)
        {
            vals[i] -= v[i];
        }
        return *this;
    }

    nVector &nVector::operator+=(double d)
    {
        for (double &v : vals)
            v += d;
        return *this;
    }

    nVector &nVector::operator*=(double d)
    {
        for (double &v : vals)
            v *= d;
        return *this;
    }

    nVector nVector::operator*(double x)
    {
        std::vector<double> vect;
        for (double &v : vals)
            vect.push_back(v * x);

        return nVector(vect);
    }

    void nVector::push(double d)
    {
        vals.push_back(d);
    }

    void nVector::print() const
    {
        for (double element : vals)
        {
            std::cout << element << " ";
        }
        std::cout << std::endl
                  << std::endl;
    }

    nVector nVector::generateRandom(int size)
    {
        nVector b(std::vector(size, 0.0));
        for (int i = 0; i < size; i++)
            b[i] = static_cast<double>(rand()) / RAND_MAX;
        return b;
    }

    nVector nVector::generateEmptyCopy(const nVector &v, double fill)
    {
        return Math::nVector(std::vector<double>(v.size(), fill));
    }

    double nVector::magSqr()
    {
        double sum = 0;
        for (double v : vals)
        {
            sum += v * v;
        }
        return sum;
    }

    nVector nVector::pairWiseMult(const nVector v1, const nVector v2)
    {
        if (v1.size() != v2.size())
            std::length_error("Vectors must have the same size for pairwise multiplication.");

        std::vector<double> v;
        for (int i = 0; i < v1.size(); i++)
            v.push_back(v1[i] * v2[i]);
        return nVector(v);
    }

    Math::nVector operator+(const Math::nVector &vec1, const Math::nVector &vec2)
    {
        if (vec1.size() != vec2.size())
            throw std::length_error("Vectors must have the same size to add them.");

        std::vector<double> v(vec1.size());

        for (int i = 0; i < vec1.size(); i++)
        {
            v[i] = vec1[i] + vec2[i];
        }

        Math::nVector newVec(v);
        return newVec;
    }

    Math::nVector operator-(const Math::nVector &vec1, const Math::nVector &vec2)
    {
        if (vec1.size() != vec2.size())
            throw std::length_error("Vectors must have the same size to add them.");

        std::vector<double> v(vec1.size());

        for (int i = 0; i < vec1.size(); i++)
        {
            v[i] = vec1[i] - vec2[i];
        }

        Math::nVector newVec(v);
        return newVec;
    }

    double operator*(const Math::nVector &vec1, const Math::nVector &vec2)
    {
        if (vec1.size() != vec2.size())
            throw std::length_error("Vectors must have the same size to perform dot product.");

        double sum{0};

        for (int i = 0; i < vec1.size(); i++)
        {
            sum += (vec1[i] * vec2[i]);
        }

        return sum;
    }

    nVector operator*(const nVector &vec1, double x)
    {
        std::vector<double> vect;
        for (double v : vec1.getVals())
            vect.push_back(v * x);

        return nVector(vect);
    }

};

#if false
int main() {
    std::vector<double> v{1, 2, 3, 4};
    Math::nVector vec1(v);

    std::vector<double> v2{2, 1, 2, 5};
    Math::nVector vec2(v2);

    vec1 + 1.0;

    return 0;
}
#endif