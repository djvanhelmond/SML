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

### Generate training set with 20 elements
training_set = {}
for _ in range(0,20):
    x, y = gen_xy()
    training_set[x] = y

### Generate testing set with 20 elements
testing_set = {}
for _ in range(0,20):
    x, y = gen_xy()
    testing_set[x] = y


### Class that contains all the ML related functions
class regressive_supervised_ML():
    def __init__(self, ts):
        self.data = {}
        self.function = []
        self.function_order = None
        self.loss = 9999999999999999
        self.load_set(ts)
        self.pick_order()

    ### load the training set
    def load_set(self, ts):
        self.data = ts

    ### print all Regression SML information
    def stats(self):
        print("Function selected           : ", list(self.function))
        print("Best fitting function order : ", self.function_order)
        print("Loss function               : ", self.loss)

    # Fit the polynomial on the data (regression)
    def find_function(self, order):
        from numpy.polynomial import polynomial as P
        x = list(self.data.keys())
        y = list(self.data.values())
        c, stats = P.polyfit(x, y, order, full=True)
        return c

    # calculate the loss function of a given polynomial over the training data
    def calc_loss(self, c):
        import math
        from numpy.polynomial.polynomial import polyval
        loss = 0
        for k,v in self.data.items():
            loss += math.sqrt(abs((polyval(k, c) - v)))
        return loss

    # pick the polynomial order with the proper trade-off
    # - fits close to many elements (low loss function)
    # - not to many orders (minimum x100 loss-improvement required for each extra order)
    def pick_order(self):
        import math
        for order in range(0, 40): # go maximum up to 40 orders deep
            function = self.find_function(order)  # map the n-th order polynomial on the data
            loss = self.calc_loss(function)       # calculate the loss function of this particular polynomial
            if math.ceil(loss) < (self.loss/100): # use this polynomial order if the loss function is significant better
                self.function = function
                self.function_order = order
                self.loss = loss

    # use the Regression SML model to make predictions
    def predict(self, x):
        from numpy.polynomial.polynomial import polyval
        return polyval(x, self.function)


### Create regressive supervised Machine Learning object
m = regressive_supervised_ML(training_set)
print("Example of Regression based Supervised Machine Learning")
print("-------------------------------------------------------")
m.stats()

# Validate against testing set and measure the error
error = 0
for k,v in testing_set.items():
    p = int(m.predict(k))
    e = abs(p - v)
    error += e
#    print("for key %i the value should be %i. Model predicts %i, error: %i" % (k, v, p, e))
print("Average error on test set   : ", error/len(testing_set))
print("===")

# Input a real world value
val = 6000
print("Realworld input value       : ", val)
print("Supervised ML prediction    : ", m.predict(val))
print("Actual value                : ", val**3)








