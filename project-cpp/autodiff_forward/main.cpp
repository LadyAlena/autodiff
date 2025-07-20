#include <iostream>
#include <cmath>

struct ValueAndPartial { float value, partial; };

struct Variable;

struct Expression {
    virtual ValueAndPartial evaluateAndDerive(Variable *variable) = 0;
    virtual ~Expression() = default;
};

struct Variable final: public Expression {
    float value;
    explicit Variable(const float value): value(value) {}
    ValueAndPartial evaluateAndDerive(Variable *variable) override {
        const float partial = (this == variable) ? 1.0f : 0.0f;
        return {value, partial};
    }
};

struct Plus final: public Expression {
    Expression *a, *b;
    Plus(Expression *a, Expression *b): a(a), b(b) {}
    ValueAndPartial evaluateAndDerive(Variable *variable) override {
        auto [valueA, partialA] = a->evaluateAndDerive(variable);
        auto [valueB, partialB] = b->evaluateAndDerive(variable);
        return {valueA + valueB, partialA + partialB};
    }
};

struct Multiply final: public Expression {
    Expression *a, *b;
    Multiply(Expression *a, Expression *b): a(a), b(b) {}
    ValueAndPartial evaluateAndDerive(Variable *variable) override {
        auto [valueA, partialA] = a->evaluateAndDerive(variable);
        auto [valueB, partialB] = b->evaluateAndDerive(variable);
        return {valueA * valueB, valueB * partialA + valueA * partialB};
    }
};

struct SIN final : public Expression {
    Expression* f;
    explicit SIN(Expression* f) : f(f) {}
    ValueAndPartial evaluateAndDerive(Variable *variable) override {
        auto [value, partial] = f->evaluateAndDerive(variable);
        return {std::sin(value), std::cos(value) * partial};
    }
};

struct COS final : public Expression {
    Expression* f;
    explicit COS(Expression* f) : f(f) {}
    ValueAndPartial evaluateAndDerive(Variable *variable) override {
        auto [value, partial] = f->evaluateAndDerive(variable);
        return {std::cos(value), - std::sin(value) * partial};
    }
};

struct SQR final : public Expression {
    Expression* f;
    explicit SQR(Expression* f) : f(f) {}
    ValueAndPartial evaluateAndDerive(Variable *variable) override {
        auto [value, partial] = f->evaluateAndDerive(variable);
        return {value * value, 2 * value * partial};
    }
};

int main () {
    // Example: Finding the partials of f = sin(x^2 + y^2) at (x, y) = (1, 2)
    Variable x(1), y(2);
    auto sqr_x = SQR(&x);
    auto sqr_y = SQR(&y);
    auto plus = Plus(&sqr_x, &sqr_y);
    auto f = SIN(&plus);

    const auto dfdx = f.evaluateAndDerive(&x).partial;
    const auto dfdy = f.evaluateAndDerive(&y).partial;

    std::cout << "∂f/∂x = " << dfdx << std::endl;
    std::cout << "∂f/∂y = " << dfdy << std::endl;

    // Output: ∂z/∂x = 0.567324, ∂z/∂y = 1.13465
    return 0;
}
