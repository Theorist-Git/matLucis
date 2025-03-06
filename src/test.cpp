#include "matLucis.hpp"

int main() {
    Matrix data = readCsv("../diabetes.csv", true);

    std::cout << data << std::endl;

    Matrix X = data.extractMatrix(
        {0, data.shape().first}, {0, data.shape().second - 1}
    );

    Matrix y = data.extractMatrix(
        {0, data.shape().first}, {data.shape().second - 1, data.shape().second}
    );

    std::vector<Matrix> res = train_test_split(
        X,
        y,
        0.8,
        42
    );

    Matrix X_train = res[0];
    Matrix X_test  = res[1];
    Matrix y_train = res[2];
    Matrix y_test  = res[3];

    // Matrix X_train = readCsv("../X_train.csv", true);
    // Matrix X_test = readCsv("../X_test.csv", true);

    // Matrix y_train = readCsv("../y_train.csv", true);
    // Matrix y_test = readCsv("../y_test.csv", true);

    // std::cout << X_train << std::endl;

    linearRegression lr;
    lr.train(X_train, y_train);

    std::cout << lr.coef_ << std::endl;
    std::cout << lr.intercept_ << std::endl;

    std::cout << lr.score(X_test, y_test).first << std::endl;

    linearRegressionGD lrgd(0.1, 10000);
    lrgd.train(X_train, y_train);

    std::cout << lrgd.coef_ << std::endl;
    std::cout << lrgd.intercept_ << std::endl;

    std::cout << lrgd.score(X_test, y_test).first << std::endl;

    return EXIT_SUCCESS;
}