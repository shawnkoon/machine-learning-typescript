# Linear Regression car estimator

App to create Linear Regression algorithm and apply to car data set.

## Goal

- Given some data about cars, can we predict relationship between Miles Per Gallon (MPG) and car's Horsepower?

## Lesson

1. Identify Feature && Labels.

   - **Features** _(Independent Variables)_
     - Horsepower
     - etc...
   - **Labels** _(Dependent Variables)_
     - Gas Millage (MPG).

2. **Regression** technique.

   - The **Label** is finite value _MPG_. Which means we will have to use Regression technique by creating linear regression prediction our self.

3. **Gradient Descent** Algorithm _(Linear Regression)_

   - `v1`, We are trying to estimate good M(Slope) and B(Starting) value for our linear equation. One algorithm we can apply here is **Gradient Descent** which essentially is reducing M & B value with learning rate then recursively execute same deduction until we some convergence.

   - `v2` using vectorized solution. We take derivative form Mean squared error slope equations, turn operations into Matrix multiplications (utilizing it's sum) to find final M & B after a **Descent**.

4. Use **Test Set** and **Training Set**.

   - Test set is used to test against LinearRegression class to how accurate the trained weight is by checking the **Coefficient of Determination**. Which the more closer we get to value `1` the better estimate we can generate.

5. Data **Normalization** vs **Standardization**.

   - To improve accruacy, we decided to **Standardize** data set. Just basic execution using mean and variance.

6. Perform **Performance Enhancement**.

   - After implmeneting the linear regression algorithm, applied bath/stochastic gradient descent method to enhance performance of the algorithm. By reducing possible MSE value per execution.

## Output

```bash

$ npm run start

...

Tensor // predict
    [[...]]

r2 is ... // Coefficient of Determination value indicating accuracy of current linear regression function.
...
```

## Misc

- The basic JavaScript version of application setup is provided by [StephenGrider](https://github.com/StephenGrider/MLKits/tree/master/regression) from his Udemy course.
