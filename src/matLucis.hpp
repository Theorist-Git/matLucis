#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <tuple>

#include "../core/matOps.hpp"

Matrix readCsv(std::string file_path, bool header=false) {
    std::ifstream file(file_path);

    if ( !file ) {
        throw std::runtime_error("Error opening file");
    }

    if ( header ) {
        std::string temp;
        
        if ( !std::getline(file, temp) ) {
            throw std::runtime_error("Error reading header");
        }
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while ( std::getline(file, line) ) {
        std::stringstream ss(line);
        std::string cell;

        std::vector<double> row;

        while ( std::getline(ss, cell, ',') ) {
            try {
                double cellVal = std::stod(cell);
                row.push_back(cellVal);
            } catch ( const std::invalid_argument& ) {
                throw std::runtime_error("Invalid entries(NaN)");
            }
        }

        data.push_back(row);
    }

    file.close();

    return Matrix(data);
}

std::vector<Matrix> train_test_split(const Matrix& X, const Matrix& y, double train_size, size_t random_state) {
    if ( train_size > 1 || train_size < 0 ) {
        throw std::invalid_argument("Invalid train data size");
    }

    if (train_size == 1) {
        return {};
    }

    Matrix X_copy = X;
    Matrix y_copy = y;

    X_copy.shuffleRows(random_state);
    y_copy.shuffleRows(random_state);

    std::pair<size_t, size_t> shapeX = X.shape();
    std::pair<size_t, size_t> shapeY = y.shape();

    size_t xrows, xcols;
    size_t yrows, ycols;

    std::tie(xrows, xcols) = shapeX;
    std::tie(yrows, ycols) = shapeY;

    if (xrows != yrows) {
        throw std::invalid_argument("X and y must have the same number of rows");
    }

    size_t train_rows = static_cast<size_t>(xrows * train_size);

    Matrix X_train = X_copy.extractMatrix(
        {0, train_rows}, {0, xcols}
    );

    Matrix X_test = X_copy.extractMatrix(
        {train_rows, xrows}, {0, xcols}
    );
    
    Matrix y_train = y_copy.extractMatrix(
        {0, train_rows}, {0, ycols}
    );

    Matrix y_test = y_copy.extractMatrix(
        {train_rows, xrows}, {0, ycols}
    );

    return {X_train, X_test, y_train, y_test};
}

class linearRegression {    
    public:
        double intercept_;
        Matrix coef_;

        linearRegression() 
        : intercept_(std::numeric_limits<double>::quiet_NaN()), coef_(1, 1, 1) {}

        void train(const Matrix& X_train, const Matrix& y_train) {
            Matrix X_train_augmented = X_train.insertCol(1, 0);

            Matrix BETA = ( (X_train_augmented.transpose() * X_train_augmented).inverse() ) * X_train_augmented.transpose() * y_train;

            this->intercept_ = BETA(0, 0);
            this->coef_      = BETA.extractMatrix( {1, BETA.shape().first}, {0, 1} );
        }

        Matrix predict(const Matrix& X_test) {
            if ( std::isnan(this->intercept_) ) {
                throw std::runtime_error("Error: Model is not trained. Call train() before predict().\n");
            }

            return (X_test * this->coef_) + this->intercept_;
        }

        std::pair<double, double> score(const Matrix& X_test, const Matrix& y_test) {
            Matrix y_pred = this->predict(X_test);

            if (y_test.shape() != y_pred.shape()) {
                throw std::runtime_error("y_test not of dim Kx1");
            }

            double ss_total = ( (y_test - y_test.mean()) ^ 2 ).sum();
            double ss_residual = ( (y_test - y_pred) ^ 2 ).sum();

            int n, k;

            std::pair<double, double> dim = X_test.shape();
            n = dim.first;
            k = dim.second;

            double R2    = 1 - (ss_residual / ss_total);
            double adjR2 = 1 - ( (1 - R2) * (n - 1) / (n - 1 - k) );

            return {R2, adjR2};
        }
};

class linearRegressionGD {
    private:
        double learning_rate;
        size_t max_iter;

    public:
        double intercept_;
        Matrix coef_;

        linearRegressionGD(double learning_rate, size_t max_iter)
        : 
        learning_rate(learning_rate),
        max_iter(max_iter),
        intercept_(std::numeric_limits<double>::quiet_NaN()),
        coef_({{-1, -1, -1}}) {
            if (learning_rate <= 0) {
                throw std::invalid_argument("Invalid learning rate");
            }
        }

        void train(const Matrix& X_train, const Matrix& y_train) {
            Matrix X_train_augmented = X_train.insertCol(1, 0);
            Matrix X_train_transpose = X_train_augmented.transpose();

            std::pair<size_t, size_t> dim = X_train_augmented.shape();
            Matrix BETA = Matrix::constValMatrix(dim.second, 1, 1.0);
            
            for (size_t epoch = 0; epoch < this->max_iter; ++epoch) {
                Matrix der_BETA = ( X_train_transpose * ( (X_train_augmented * BETA) - y_train ) ) / dim.first;

                BETA = BETA - this->learning_rate * der_BETA;
            }

            this->intercept_ = BETA(0, 0);
            this->coef_      = BETA.extractMatrix( {1, BETA.shape().first}, {0, 1} );
        }

        Matrix predict(const Matrix& X_test) {
            if ( std::isnan(this->intercept_) ) {
                throw std::runtime_error("Error: Model is not trained. Call train() before predict().\n");
            }

            return (X_test * this->coef_) + this->intercept_;
        }

        std::pair<double, double> score(const Matrix& X_test, const Matrix& y_test) {
            Matrix y_pred = this->predict(X_test);

            if (y_test.shape() != y_pred.shape()) {
                throw std::runtime_error("y_test not of dim Kx1");
            }

            double ss_total = ( (y_test - y_test.mean()) ^ 2 ).sum();
            double ss_residual = ( (y_test - y_pred) ^ 2 ).sum();

            int n, k;

            std::pair<double, double> dim = X_test.shape();
            n = dim.first;
            k = dim.second;

            double R2    = 1 - (ss_residual / ss_total);
            double adjR2 = 1 - ( (1 - R2) * (n - 1) / (n - 1 - k) );

            return {R2, adjR2};
        }
};