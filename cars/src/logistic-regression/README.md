# Logistic Regression car estimator

App to create Logistic Regression algorithm and apply to car data set.

## Goal

- Given some data about cars, can we predict relationship with Emission pass or fail?

## Lesson

1. Identify Feature && Labels.

   - **Features** _(Independent Variables)_
     - Horsepower
     - Engine Displacement
     - Weight
     - etc...
   - **Labels** _(Dependent Variables)_
     - Emission status

2. **Classification** technique.

   - The **Label** is finite value either pass/fail, which means we are estimating by performing a binary classification.

3. **Gradient Descent** Algorithm _(Logistic Regression)_

   - `v1`, We are trying to estimate good M(Slope) and B(Starting) value for our logistic equation. One algorithm we can apply here is **Gradient Descent** which essentially is reducing M & B value with learning rate then recursively execute same deduction until we some convergence.

   - `v2` using vectorized solution. We take derivative form Mean squared error slope equations, turn operations into Matrix multiplications (utilizing it's sum) to find final M & B after a **Descent**.

   - `extra` Very similar to linear-regression algorithm, but we are using sigmoid instead of MSE equations.

4. Use **Test Set** and **Training Set**.

   - We are performing test by checking to see how accurate our prediction is by counting number of unmatched results and comparing to what was expected.

5. Data **Normalization** vs **Standardization**.

   - To improve accruacy, we decided to **Standardize** data set. Just basic execution using mean and variance.

## Output

```bash

$ npm run start

...

Executing Logistic regression analysis 📈

Accuracy : x %

Finished Logistic regression analysis 📉
...
```

## Misc

- The basic JavaScript version of application setup is provided by [StephenGrider](https://github.com/StephenGrider/MLKits/tree/master/regression) from his Udemy course.
