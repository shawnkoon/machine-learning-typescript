# House Price estimator Application

Estimate House price based on near location house prices using KNN algorithm with Tensorflow & TypeScript.

## Goal

- Given a House with locations, can we figure out an estimated value of the house based on houses near that point?

## Lesson

1. Identify Feature && Labels.

   - **Features** _(Independent Variables)_
     - Longitude
     - Latitude
     - etc...
   - **Labels** _(Dependent Variables)_
     - House price (_Thousands \$_)

2. **Regression** technique.

   - The **Label** is finite value _price_. Which means we will have to use Regression technique by taking an average of near house prices.

3. **K-Nearest Neighbor** Algorithm _(knn)_

   - Basic algorithm starts off by taking in [Lat, Lon] feature sets, compare the error value in test result. Start adding in more features to reduce the error boundary and find perfect set.

4. Use **Test Set** and **Training Set**.

   - Will be taking in certain amount of test sets to test against our algorithms and feature set.

5. Data **Normalization** vs **Standardization**.

   - As I found out Longitude and Latitudes are not enough feature set to correctly estimate house price, decided to take in `sqft_lot` value. However, because of possible outliers in the new feature, I had to go with **Standardized** method to more evenly distribute our data.

6. Perform **Feature Selection**.

   - After trials and errors, came to a conclusion that `'lat', 'long', 'sqft_lot', 'sqft_living'` are good set of features to be used to estimate house price using KNN algorithm.

## Output

```bash

$ npm run start
...
Prediction: 1251260 Real: 1085000 Error: -15.323502304147466%
Prediction: 519756.5 Real: 466800 Error: -11.344580119965723%
Prediction: 433700 Real: 425000 Error: -2.047058823529412%
Prediction: 455800 Real: 565000 Error: 19.327433628318584%
Prediction: 699750 Real: 759000 Error: 7.806324110671936%
Prediction: 584260 Real: 512031 Error: -14.106372465729613%
Prediction: 835450 Real: 768000 Error: -8.782552083333334%
Prediction: 1329790 Real: 1532500 Error: 13.227406199021207%
Prediction: 279422.5 Real: 204950 Error: -36.336911441815076%
Prediction: 228767.5 Real: 247000 Error: 7.381578947368421%
```

Most of the Errors are below 20% range.

## Misc

- The basic JavaScript version of application setup is provided by [StephenGrider](https://github.com/StephenGrider/MLKits/tree/master/knn-tf) from his Udemy course.
