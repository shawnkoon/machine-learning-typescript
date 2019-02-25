# House Price estimator Application

Estimate House price based on near location house prices using KNN algorithm with Tensorflow & TypeScript.

## Goal

- Given a House with locations, can we figure out an estimated value of the house based on houses near that point?

## Lesson

1. Identify Feature && Labels.

   - **Features** _(Independent Variables)_
     - Longitude
     - Latitude
   - **Labels** _(Dependent Variables)_
     - House price (_Thousands \$_)

2. **Regression** technique.

   - The **Label** is finite value _price_. Which means we will have to use Regression technique by taking average of near house prices.

3. **K-Nearest Neighbor** Algorithm _(knn)_

4. Use **Test Set** and **Training Set**.

5. Data **Normalization**.

6. Perform **Feature Selection**.

## Misc

- The basic JavaScript version of application setup is provided by [StephenGrider](https://github.com/StephenGrider/MLKits/tree/master/knn-tf) from his Udemy course.
