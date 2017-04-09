#!/usr/bin/env python3

# Supervised machine learning
# "A computer program is said to learn from experience E with respect to some
# class of tasks T and performance measure P if its performance at tasks in T,
# as measured by P, improves with experience E."

# Regressive Supervised Machine Learning
# 1: load training set
# 2: find the best function to describe the relation the data (polynomial)
#    - "best" function is the one with small loss function but low on noise errors
# 3: validate the best function on the testing set.
# 4: accept real world input.


# We use this generator to make sure training set and testing set are independent and identically distributed.
# We select a random X, and create a Y where Y = X**3 + a bit of noise.
def gen_xy():
    import random
    x = random.randint(10, 1000)
    y = x ** 3 + (random.randint(-1, 1) * random.random() * 10)
    return x, y

training_set = {}
for _ in range(0,20):
    x, y = gen_xy()
    training_set[x] = y

testing_set = {}
for _ in range(0,20):
    x, y = gen_xy()
    testing_set[x] = y



class regressive_supervised_ML():
    def __init__(self, ts):
        self.data = {}
        self.function = []
        self.function_order = None
        self.loss = 9999999999999999
        self.load_set(ts)
        self.pick_order()

    def load_set(self, ts):
        self.data = ts

    def stats(self):
        print("Function selected          : ", self.function)
        print("Best fitting function order: ", self.function_order)
        print("Loss function              : ", self.loss)

    # polynomial regression
    def find_function(self, order):
        from numpy.polynomial import polynomial as P
        x = list(self.data.keys())
        y = list(self.data.values())
        c, stats = P.polyfit(x, y, order, full=True)
        return c

    def calc_loss(self, c):
        import math
        from numpy.polynomial.polynomial import polyval
        loss = 0
        for k,v in self.data.items():
            loss += math.sqrt(abs((polyval(k, c) - v)))
        return loss

    def pick_order(self):
        import math
        for order in range(0, 40):
            function = self.find_function(order)
            loss = self.calc_loss(function)
            if math.ceil(loss) < (self.loss/100): # make sure that the loss function has significant increase)
                self.function = function
                self.function_order = order
                self.loss = loss

    def predict(self, x):
        from numpy.polynomial.polynomial import polyval
        return polyval(x, self.function)


### Create regressive supervised Machine Learning object
m = regressive_supervised_ML(training_set)
m.stats()

# Validate against testing set
error = 0
for k,v in testing_set.items():
    p = int(m.predict(k))
    e = abs(p - v)
    error += e
#    print("for key %i the value should be %i. Model predicts %i, error: %i" % (k, v, p, e))
print("Average error on test set  : ", error/len(testing_set))
print("===")


print(m.predict(6000))
print(600**3)








