#if false

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include "headers/matrix(old).h"

namespace no
{
    Matrix::Matrix(const std::vector<std::vector<double>> &g)
    {
        if (g.size() == 0)
            throw std::invalid_argument("Matrix must have rows.");

        int size = g[0].size();
        for (std::vector<double> row : g)
        {
            if (size != row.size())
                throw std::invalid_argument("Rows must be same size.");
        }

        grid = g;
        h = g.size();
        w = g[0].size();
    }

    std::vector<double> Matrix::operator[](int i) const { return grid[i]; }
    std::vector<double> &Matrix::operator[](int i) { return grid[i]; }

    int Matrix::width() const
    {
        return grid[0].size();
    }

    int Matrix::height() const
    {
        return grid.size();
    }

    void Matrix::push(std::vector<double> v)
    {
        if (v.size() != width())
        {
            throw std::invalid_argument("Rows must be same size");
        }
        grid.push_back(v);
    }

    void Matrix::print() const
    {
        for (std::vector<double> row : grid)
        {
            for (double v : row)
            {
                std::cout << v << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    Matrix Matrix::transpose()
    {
        std::vector<std::vector<double>> vv;
        for (int i = 0; i < width(); i++)
        {
            std::vector<double> v;
            for (int j = 0; j < height(); j++)
            {
                v.push_back(grid[j][i]);
            }
            vv.push_back(v);
        }
        return Matrix(vv);
    }

    Matrix Matrix::generateEmptyCopy(const Matrix &v, double fill)
    {
        return Matrix(std::vector<std::vector<double>>(v.height(), std::vector<double>(v.width(), fill)));
    }

    Matrix Matrix::generateRandom(int width, int height)
    {
        std::vector<std::vector<double>> vv;
        std::vector<double> row(width, 0.0);
        for (int r = 0; r < height; r++)
        {
            for (int i = 0; i < width; i++)
                row[i] = static_cast<double>(rand()) / RAND_MAX;
            vv.push_back(row);
        }
        return Matrix(vv);
    }

    Matrix Matrix::outerProduct(const nVector &v1, const nVector &v2)
    {
        std::vector<std::vector<double>> vv;
        for (double v : v1.getVals())
        {
            vv.push_back((v2 * v).getVals());
        }
        return Matrix(vv);
    }

    Matrix &Matrix::operator+=(const Matrix &m)
    {
        int h = height();
        int w = width();

        if (w != m.width())
            throw std::invalid_argument("Width mismatch.");
        else if (h != m.height())
            throw std::invalid_argument("Height mismatch.");

        for (int i = 0; i < height(); i++)
        {
            for (int j = 0; j < width(); j++)
            {
                grid[i][j] += m[i][j];
            }
        }
        return *this;
    }

    Matrix &Matrix::operator-=(const Matrix &m)
    {
        int h = height();
        int w = width();

        if (w != m.width())
            throw std::invalid_argument("Width mismatch.");
        else if (h != m.height())
            throw std::invalid_argument("Height mismatch.");

        for (int i = 0; i < height(); i++)
        {
            for (int j = 0; j < width(); j++)
            {
                grid[i][j] -= m[i][j];
            }
        }
        return *this;
    }

    Matrix operator+(const Matrix &mat1, const Matrix &mat2)
    {
        int height = mat1.height();
        int width = mat1.width();

        if (width != mat2.width())
            throw std::invalid_argument("Width mismatch.");
        else if (height != mat2.height())
            throw std::invalid_argument("Height mismatch.");

        std::vector<double> v(width);
        std::vector<std::vector<double>> vv(height, v);

        for (int i = 0; i < mat1.height(); i++)
        {
            for (int j = 0; j < mat1.width(); j++)
            {
                vv[i][j] = mat1[i][j] + mat2[i][j];
            }
        }

        Matrix m(vv);
        return m;
    }

    Matrix operator-(const Matrix &mat1, const Matrix &mat2)
    {
        int height = mat1.height();
        int width = mat1.width();

        if (width != mat2.width())
            throw std::invalid_argument("Width mismatch.");
        else if (height != mat2.height())
            throw std::invalid_argument("Height mismatch.");

        std::vector<double> v(width);
        std::vector<std::vector<double>> vv(height, v);

        for (int i = 0; i < mat1.height(); i++)
        {
            for (int j = 0; j < mat1.width(); j++)
            {
                vv[i][j] = mat1[i][j] - mat2[i][j];
            }
        }

        Matrix m(vv);
        return m;
    }

    Matrix operator*(const Matrix &mat1, const Matrix &mat2)
    {
        // const int MAX_THREADS = std::thread::hardware_concurrency();

        // const int height = mat1.height();
        // const int width = mat2.width();

        // if (mat1.width() != mat2.height())
        //     throw std::invalid_argument("Dimension mismatch.");

        // std::vector<double> v(width, 0.0);
        // std::vector<std::vector<double>> vv(height, v);

        // const int threadNum = std::min({height, MAX_THREADS, 1});
        // std::thread threads[threadNum];
        // int block = std::max(height / threadNum, 1);

        // auto multiply = [width, mat1, mat2, &vv](const int &start, const int &end) -> void
        // {
        //     for (int i = start; i < end; i++)
        //         for (int j = 0; j < width; j++)
        //             for (int k = 0; k < width; k++)
        //                 vv[i][j] += mat1[i][k] * mat2[k][j];
        // };

        // std::reference_wrapper vvRef = std::ref(vv);

        // int start = 0;
        // int end = block;

        // for (int i = 0; i < threadNum - 1; i++)
        // {
        //     threads[i] = std::thread(std::ref(multiply), start, end);
        //     start = end;
        //     end += block;
        // }
        // threads[threadNum - 1] = std::thread(std::ref(multiply), start, height);

        // for (int i = 0; i < threadNum; i++)
        //     threads[i].join();

        // Matrix m(vv);
        // return m;

        const int height = mat1.height();
        const int width = mat2.width();

        if (mat1.width() != mat2.height())
            throw std::invalid_argument("Dimension mismatch.");

        std::vector<std::vector<double>> vv(height, std::vector<double>(width, 0.0));

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                for (int k = 0; k < mat2.height(); k++)
                    vv[i][j] += mat1[i][k] * mat2[k][j];

        Math::Matrix m(vv);
        return m;
    }

    nVector operator*(const Matrix &mat1, const nVector &vect)
    {
        int height = mat1.height();
        int width = mat1.width();
        int len = vect.size();

        if (len != width)
            throw std::invalid_argument("Dimension mismatch.");

        std::vector<double> v(height, 0.0);

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                v[i] += mat1[i][j] * vect[j];

        nVector vv(v);
        return vv;
    }

    Matrix operator*(const Matrix &mat1, const double &n)
    {
        int height = mat1.height();
        int width = mat1.width();

        std::vector<double> v(width);
        std::vector<std::vector<double>> vv(height, v);

        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                vv[i][j] = mat1[i][j] * n;

        Matrix m(vv);
        return m;
    }
}

#if false
Matrix operator*(const Matrix &mat1, const Matrix &mat2)
{
    const int MAX_THREADS = std::thread::hardware_concurrency();
    const int height = mat1.height();
    const int width = mat2.width();

    if (mat1.width() != mat2.height())
        throw std::invalid_argument("Dimension mismatch.");

    std::vector<double> v(width, 0.0);
    std::vector<std::vector<double>> vv(height, v);

    const int threadNum = std::min(height, MAX_THREADS, 1);
    std::thread threads[threadNum];
    int block = std::max(height / threadNum, 1);

    void (*multiply)(const int &start, const int &end, const int &width,
                     std::vector<std::vector<double>> &matC,
                     const Matrix &matA,
                     const Matrix &matB);
    multiply = &Matrix::multiply;

    std::reference_wrapper vvRef = std::ref(vv);

    int start = 0;
    int end = block;

    for (int i = 0; i < threadNum - 1; i++)
    {
        threads[i] = std::thread(std::ref(multiply), start, end, width, vvRef, std::ref(mat1), std::ref(mat2));
        start = end;
        end += block;
    }
    threads[threadNum - 1] = std::thread(std::ref(multiply), start, height, width, vvRef, std::ref(mat1), std::ref(mat2));

    for (int i = 0; i < threadNum; i++)
        threads[i].join();

    Matrix m(vv);
    return m;
}
Math::Matrix mult(const Math::Matrix &mat1, const Math::Matrix &mat2)
{
    const int height = mat1.height();
    const int width = mat2.width();

    if (mat1.width() != mat2.height())
        throw std::invalid_argument("Dimension mismatch.");

    std::vector<double> v(width, 0.0);
    std::vector<std::vector<double>> vv(height, v);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            for (int k = 0; k < mat2.height(); k++)
                vv[i][j] += mat1[i][k] * mat2[k][j];

    Math::Matrix m(vv);
    return m;
}


int main()
{
    int size = 20;
    std::vector<std::vector<double>> vv(size, std::vector<double>(size, 1));
    Math::Matrix m(vv);

    std::vector<std::vector<double>> vv2(size, std::vector<double>(size, 2));
    Math::Matrix m2(vv2);

    std::vector<std::vector<double>> vv3(size, std::vector<double>(4, 30));
    Math::Matrix m3(vv3);

    Math::nVector vect1(std::vector<double>(10, 11));

// m.print();
// m2.print();
// m3.print();
// vect1.print();
#if true
    int n = 5;

    std::cout << "begin naive" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++)
    {
        mult(m, m2);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / n) << std::endl;
    std::cout << "end" << std::endl;

    std::cout << "begin threaded" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++)
    {
        (m * m2);
    }

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / n) << std::endl;
    std::cout << "end " << std::endl;
#endif

    // (m * vect1).print();

    // (m3 * vect1).print();

    return 0;
}
#endif

#endif