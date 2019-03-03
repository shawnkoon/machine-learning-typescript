// import
import * as tf from '@tensorflow/tfjs';
import _ from 'lodash';

type Tensor = tf.Tensor;
type Options = {
  batchSize: number;
  iterations: number;
  learningRate: number;
};

export interface LRProps {
  train(): void;
  gradientDescent(features: Tensor, labels: Tensor): void;
  test(testFeatures: Tensor, testLabels: Tensor): Promise<number>;
  predict(observations: Tensor): Tensor;
}

class LinearRegression implements LRProps {
  public weights: Tensor;
  public mean: Tensor | undefined;
  public variance: Tensor | undefined;
  public mseHistory: number[];

  constructor(public features: Tensor, public labels: Tensor, public options: Options) {
    this.features = this.processFeatures(features);
    this.weights = tf.zeros([<number>this.features.shape[1], 1]);
    this.mseHistory = [];
  }

  /**
   * Version #1 of this function can be found git hash f0e85eb.
   */
  gradientDescent(features: Tensor, labels: Tensor) {
    // 2D array of mx + b
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    const slopesOfMSE = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopesOfMSE.mul(this.options.learningRate));
  }

  train() {
    const { batchSize, iterations } = this.options;
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);

    for (let i = 0; i < iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;

        const featureSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations: Tensor): Tensor {
    return this.processFeatures(observations).matMul(this.weights);
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

  private recordMSE() {
    const mse = <number>this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .arraySync();

    this.mseHistory.unshift(mse);
  }

  private updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

export default LinearRegression;
