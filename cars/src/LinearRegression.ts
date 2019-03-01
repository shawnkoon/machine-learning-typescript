// import
import * as tf from '@tensorflow/tfjs';
import _ from 'lodash';

type Tensor = tf.Tensor;
type Options = {
  learningRate: number;
  iterations: number;
};

export interface LRProps {
  train(): void;
  gradientDescent(): void;
  test(testFeatures: Tensor, testLabels: Tensor): Promise<number>;
}

class LinearRegression implements LRProps {
  public weights: Tensor;
  public mean: Tensor | undefined;
  public variance: Tensor | undefined;

  constructor(public features: Tensor, public labels: Tensor, public options: Options) {
    this.features = this.processFeatures(features);
    this.weights = tf.zeros([2, 1]);
  }

  /**
   * Version #1 of this function can be found git hash f0e85eb.
   */
  gradientDescent() {
    // 2D array of mx + b
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  /**
   * Test existing linear regression algorithm against test data set.
   * returns back with **Coefficient of Determination** of test set.
   */
  async test(testFeatures: Tensor, testLabels: Tensor): Promise<number> {
    testFeatures = this.processFeatures(testFeatures);

    // 2D array of mx + b.
    const predictions = testFeatures.matMul(this.weights);

    const ssResiduals = <number>await testLabels
      .sub(predictions)
      .pow(2)
      .sum()
      .array();

    const ssTotal = <number>await testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .array();

    return 1 - ssResiduals / ssTotal;
  }

  /**
   * Helper method to update incoming tensor by inserting
   * column of 1s then, standardize the new feature set
   * to perform matrix multiplication with
   * the `this.weight` and simulate `mx + b`.
   */
  private processFeatures(features: Tensor): Tensor {
    let newFeatures: Tensor;

    if (this.mean && this.variance) {
      newFeatures = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      newFeatures = this.standardize(features);
    }

    newFeatures = tf.ones([newFeatures.shape[0], 1]).concat(newFeatures, 1);

    return newFeatures;
  }

  private standardize(features: Tensor): Tensor {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }
}

export default LinearRegression;
