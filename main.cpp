#include <iostream>
#include "headers/matrix.h"
#include "headers/nn.h"
#include "headers/read.h"

std::vector<std::tuple<Math::Matrix, Math::Matrix>> generateData(int len)
{
    std::vector<std::tuple<Math::Matrix, Math::Matrix>> data;
    for (int i = 0; i < len; i++)
    {
        double x = std::abs(static_cast<double>(rand()) / RAND_MAX);
        double y = std::abs(static_cast<double>(rand()) / RAND_MAX);
        double z = (x) * (x) + (y) * (y) - 0.7;
        Math::Matrix input = Math::Matrix(std::vector<double>{x, y});
        Math::Matrix output = (z <= 0) ? Math::Matrix(std::vector<double>{1, 0}) : Math::Matrix(std::vector<double>{0, 1});
        data.push_back(std::make_tuple(input, output));
    }

    return data;
}

int main()
{

#if false

    Activation::ActivationFn *fn = new Activation::Sigmoid();

    auto bruh = Math::Matrix(std::vector<double>{1, 2, 3, 4});
    bruh.print();

    fn->fn(bruh).print();
    fn->fnDerv(bruh).print();

#endif
#if false

    std::vector<std::tuple<Math::Matrix, Math::Matrix>> training = readData("data/mnist_train.csv", 5000);
    std::vector<std::tuple<Math::Matrix, Math::Matrix>> testing = readData("data/mnist_test.csv", 800);

    FFNN nn(std::vector<int>{784, 100, 30, 10},new Cost::L2Cost(), new Activation::Sigmoid());

    nn.train(training, 30, 5000, 3, testing);

    std::cout << nn.evaluate(testing) << std::endl;
#endif  

#if true
    srand(10);

    FFNN nn(std::vector<int>{2, 4, 2},new Cost::L2Cost(), new Activation::Sigmoid());

    std::vector<std::tuple<Math::Matrix, Math::Matrix>> training = generateData(1000);
    std::vector<std::tuple<Math::Matrix, Math::Matrix>> testing = generateData(100);

    nn.train(training, 300, 1000, 30, testing);

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
    Math::Matrix v1 = (Math::Matrix({3, 4, 5}));
    Math::Matrix v2 = (Math::Matrix({1, 2, 3}));

    v1.print();
    v2.print();
    Math::Matrix::outerProduct(v1, v2).print();


    //Math::Matrix::outerProduct(Math::Matrix(std::vector<double>{1, 2}), Math::Matrix(std::vector<double>{3, 4})).print();
    //Math::Matrix::pairWiseMult(Math::Matrix(std::vector<double>{1, 2}), Math::Matrix(std::vector<double>{3, 4})).print();

#endif

#if false
    FFNN nn(std::vector<int>{2, 4, 2},new Cost::L2Cost(), new Activation::Sigmoid());

    auto output = nn.feedForward(Math::Matrix(std::vector<double>{2.0,3.0}));

    for (auto b : nn.getWeights())
    {
        std::cout << "layer-print " << std::endl;
        b.print();
    }

    output.print();

    // (m3 * vect1).print();
#endif

#if false
    FFNN nn(std::vector<int>{2, 4, 2},new Cost::L2Cost(), new Activation::Sigmoid());
    auto t = nn.backPropagate(Math::Matrix(std::vector<double>{1, 1}), Math::Matrix(std::vector<double>{2, 2}));
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

#if false
    Math::Matrix M1({{1, 2}, {4, 5}, {7, 8}});
    Math::Matrix M2({{1, 2, 3}, {4, 5, 6}});

    Math::Matrix M3 = Math::Matrix(512, 512, 2);
    Math::Matrix M4 = Math::Matrix(512, 1, 2);

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
