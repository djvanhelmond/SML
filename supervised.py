#!/usr/bin/env python3

# Written by: Dirk-Jan van Helmond (djvanhelmond@gmail.com)

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
def gen_xy(x=0, Noise=False):
    # Define a polynomial for generating the sets
    # f(x) = c0 + C1 * x + c1 * x^2 + c2 * X^3 ...etc.
    c = [400, 300, 200, 100, 0, 0]
    import random
    import math
    # If no x is passed into the function given then x will be random.
    if x == 0:
        x = random.randint(1, 1000)
    # Y is f(x)
    y = (c[0]) + (c[1] * x ** 2) + (c[2] * x ** 3) + (c[3] * x ** 4) + (c[4] * x ** 6) + (c[5] * x ** 6)
    # We might want to add some noise to y so that we have some deviation in the training and test sets
    if Noise:
        y += (random.randint(-1, 1) * random.random() * math.sqrt(x) * 5)
    return x, y


### Class that contains all the ML related functions
class regressive_supervised_ML():
    def __init__(self, ts):
        self.data = ts
        self.function = []
        self.function_order = None
        self.loss = 9999999999999999
        self.pick_order()

    ### load the training set
    def add_pair_to_set(self, pair):
        self.data[pair[0]] = pair[1]

    ### print all Regression SML information
    def stats(self):
        import math
        print("Function selected           : ", list(self.function))
        print("Best fitting function order : ", self.function_order)
        print("Training loss function      : ", self.loss)
        test_loss = 0
        for k, v in testing_set.items():
            test_loss += math.sqrt(abs(int(self.predict(k)) - v))
        print("Average error on test set   : ", test_loss / len(testing_set))

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

    def regenerate_polymonial(self):
        self.function = self.find_function(self.function_order)  # map the n-th order polynomial on the data

    # use the Regression SML model to make predictions
    def predict(self, x):
        from numpy.polynomial.polynomial import polyval
        return polyval(x, self.function)


### Generate a training set with 20 elements
training_set = {}
for _ in range(0,20):
    x, y = gen_xy(Noise=True)
    training_set[x] = y

### Generate a testing set with 20 elements
testing_set = {}
for _ in range(0,20):
    x, y = gen_xy(Noise=True)
    testing_set[x] = y


### Create regressive supervised Machine Learning object
sml = regressive_supervised_ML(training_set)
print("Example of Regression based Supervised Machine Learning")

input_key=0
while input_key != 2:
    print("-------------------------------------------------------")
    sml.stats()
    print("-------------------------------------------------------")
    print("0: Give a prediction")
    print("1: Add a value")
    print("2: Exit")
    input_key = int(input("Press a number and then Enter to continue: "))
    if input_key == 0:
        val = int(input("Give a value to predict outcome: "))
        print("Realworld input value       : ", val)
        print("Supervised ML prediction    : ", sml.predict(val))
        print("Actual calculated value     : ", gen_xy(val)[1])
    if input_key == 1:
        x_val = int(input("Give x value: "))
        y_val = int(input("Give y value: "))
        sml.add_pair_to_set((x_val, y_val))
        sml.regenerate_polymonial()










