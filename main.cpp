#include <iostream>
#include "headers/matrix.h"
#include "headers/vector.h"
#include "headers/nn.h"
#include "headers/read.h"

std::vector<std::tuple<Math::nVector, Math::nVector>> generateData(int len)
{
    std::vector<std::tuple<Math::nVector, Math::nVector>> data;
    for (int i = 0; i < len; i++)
    {
        double x = std::abs(static_cast<double>(rand()) / RAND_MAX);
        double y = std::abs(static_cast<double>(rand()) / RAND_MAX);
        double z = (x) * (x) + (y) * (y)-0.7;
        Math::nVector input = Math::nVector(std::vector<double>{x, y});
        Math::nVector output = (z <= 0) ? Math::nVector(std::vector<double>{1, 0}) : Math::nVector(std::vector<double>{0, 1});
        data.push_back(std::make_tuple(input, output));
    }

    return data;
}

int main()
{
#if false

    std::vector<std::tuple<Math::nVector, Math::nVector>> training = readData("data/mnist_train.csv", 5000);
    std::vector<std::tuple<Math::nVector, Math::nVector>> testing = readData("data/mnist_test.csv", 800);

    FFNN nn(std::vector<int>{784, 30, 10},new Cost::L2Cost(), new Activation::Sigmoid());

    nn.train(training, 30, 10, 3.0, testing);

    std::cout << nn.evaluate(testing) << std::endl;
#endif

#if false
    srand(10);

    FFNN nn(std::vector<int>{2, 4, 2},new Cost::L2Cost(), new Activation::Sigmoid());

    std::vector<std::tuple<Math::nVector, Math::nVector>> training = generateData(10000);
    std::vector<std::tuple<Math::nVector, Math::nVector>> testing = generateData(1000);

    nn.train(training, 30, 1000, 3.0, testing);

    std::cout << nn.evaluate(testing) << std::endl;
#endif

#if false
    Math::Matrix m = Math::Matrix::generateRandom(5, 3);
    Math::Matrix m2 = Math::Matrix::generateRandom(5, 3);
    m.print();
    m2.print();

    m -= m2;
    m.print();
    //m.transpose().print();
    std::cout << std::endl;
    Math::nVector v1 = (Math::nVector({3, 4, 5}));
    Math::nVector v2 = (Math::nVector({1, 2, 3}));

    v1.print();
    v2.print();
    Math::Matrix::outerProduct(v1, v2).print();


    //Math::Matrix::outerProduct(Math::nVector(std::vector<double>{1, 2}), Math::nVector(std::vector<double>{3, 4})).print();
    //Math::nVector::pairWiseMult(Math::nVector(std::vector<double>{1, 2}), Math::nVector(std::vector<double>{3, 4})).print();

#endif

#if false
    auto t = nn.backPropagate(Math::nVector(std::vector<double>{1, 1}), Math::nVector(std::vector<double>{2, 2}));
    auto db = std::get<0>(t);
    auto dw = std::get<1>(t);

    for (auto b : nn.getLayers())
    {
        std::cout << "layer-print ";
        b.print();
    }

    for (auto v : db)
    {
        std::cout << "db-print ";
        v.print();
    }

    for (auto v : dw)
    {
        std::cout << "dw-print ";
        v.print();
    }
    // (m3 * vect1).print();
#endif

#if true
    Math::Matrix M1({{1, 2}, {4, 5}, {7, 8}});
    Math::Matrix M2({{1, 2, 3}, {4, 5, 6}});

    Math::Matrix M3 = Math::Matrix::generateEmptyCopy(Math::Matrix::generateRandom(512, 512), 2);
    Math::Matrix M4 = Math::Matrix::generateEmptyCopy(Math::Matrix::generateRandom(1, 512), 2);

    int n = 10;

    std::cout << "begin multiplication" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++)
    {
        M1 * M2;
        M3 * M3;
        M3 * M4;
        std::cout << i << " finished" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << (duration.count() / n) << std::endl;
    std::cout << "end" << std::endl;
#endif
    return 0;
}
