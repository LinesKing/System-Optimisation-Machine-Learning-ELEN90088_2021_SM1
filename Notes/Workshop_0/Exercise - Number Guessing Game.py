import random

target = random.random()
print("target= {}".format(target))
max = 1 # initial maximum
min = 0 # initial minimum
guess = (max + min)/2 # First guess
count = 0 # guessing times
bias = abs(target - guess) # guessing bias
tolerance = 1.0e-06 # The number depending on a solver's stopping criteria.

while bias > tolerance:
    if guess < target:
        print("guess {} = {} is smaller than target".format(count,guess))
        min = guess # If the guess number is small, assign the result of this guess to min as the next minimum
        guess = (guess + max)/2
        count += 1
        bias = abs(target - guess) # guessing bias
    else:
        print("guess {} = {} is bigger than target".format(count,guess))
        max = guess# If the guess number is large, assign the result of this guess to max as the next maximum
        guess = (min + guess)/2
        count += 1
        bias = abs(target - guess) # guessing bias

print("Total guessing times are {}".format(count))