# Plinko Application

Plinko application.

## Goal

- Given some data about where a ball is dropped from, can we predict what bucket it will end up in?

## Lesson

1. Identify Feature && Labels.

   - **Features** _(Independent Variables)_
     - Drop Position
     - Ball Bounciness
     - Ball Size
   - **Labels** _(Dependent Variables)_
     - Bucket a ball lands in.

2. **Classification** technique.

   - Value of label belongs to a **discrete** set.

3. **K-Nearest Neighbor** Algorithm _(knn)_

   1. List = |dropPosition - targetPosition|
   2. Asending(List).
   3. Take first **K** amount of Labels.
   4. Guess based on result.

4. Use **Test Set** and **Training Set**.

   - To determine accuracy on various different feature sets.

5. Data **Normalization**.

   - By normalizing we gain more accruacy on calculations by converting data into specific range of between 0 - 1. Instead of say, 0 to 750 in our plinko application.

6. Perform **Feature Selection**.
   - It is nice to consider multiple different features into our calculation, however, some just does not make sense to include in our feature set.
   - _Excluding some feature_ from calculation feature set is called feature selection.
   - In our case, we ended up excluding **Bounciness** feature due to in accuracy in our result. Check `normalizeData(...)`.

## Misc

- The basic application setup is provided by [StephenGrider](https://github.com/StephenGrider/MLKits/tree/master/plinko) from his Udemy course.
