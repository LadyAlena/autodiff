#include <iostream>
#include <cmath>

struct Expression {
    float value{};
    virtual void evaluate() = 0;
    virtual void derive(float seed) = 0;
    virtual ~Expression() = default;
};

struct Variable final: public Expression {
    float partial;
    explicit Variable(const float value) {
        this->value = value;
        partial = 0.0f;
    }
    void evaluate() override {}
    void derive(const float seed) override {
        partial += seed;
    }
};

struct Plus final: public Expression {
    Expression *a, *b;
    Plus(Expression *a, Expression *b): a(a), b(b) {}
    void evaluate() override {
        a->evaluate();
        b->evaluate();
        value = a->value + b->value;
    }
    void derive(const float seed) override {
        a->derive(seed);
        b->derive(seed);
    }
};

struct Multiply final: public Expression {
    Expression *a, *b;

    Multiply(Expression *a, Expression *b): a(a), b(b) {}

    void evaluate() override {
        a->evaluate();
        b->evaluate();
        value = a->value * b->value;
    }

    void derive(const float seed) override {
        a->derive(b->value * seed);
        b->derive(a->value * seed);
    }
};

struct SIN final : public Expression {
    Expression* f;

    explicit SIN(Expression* f) : f(f) {}

    void evaluate() override {
        f->evaluate();
        value = std::sin(f->value);
    }

    void derive(const float seed) override {
        f->derive(std::cos(f->value) * seed);
    }
};

struct SQR final : public Expression {
    Expression* f;

    explicit SQR(Expression* f) : f{f} {}

    void evaluate() override {
        f->evaluate();
        value = f->value * f->value;
    }

    void derive(const float seed) override {
        f->derive(2 * f->value * seed);
    }
};

int main () {
    // Example: Finding the partials of f = sin(x^2 + y^2) at (x, y) = (1, 2)
    Variable x(1), y(2);
    auto sqr_x = SQR(&x);
    auto sqr_y = SQR(&y);
    auto plus = Plus(&sqr_x, &sqr_y);
    auto f = SIN(&plus);
    f.evaluate();
    f.derive(1);
    std::cout << "∂f/∂x = " << x.partial << ", "
              << "∂f/∂y = " << y.partial << std::endl;
    // Output: ∂f/∂x = 0.567324, ∂f/∂y = 1.13465
    return 0;
}