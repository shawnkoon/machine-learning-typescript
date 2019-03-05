// import
import * as tf from '@tensorflow/tfjs';
import _ from 'lodash';

type Tensor = tf.Tensor;
type Options = {
  batchSize: number;
  iterations: number;
  learningRate: number;
  decisionBoundary: number;
};

export interface LRProps {
  train(): void;
  gradientDescent(features: Tensor, labels: Tensor): void;
  test(testFeatures: Tensor, testLabels: Tensor): number;
  predict(observations: Tensor): Tensor;
}

class LogisticRegression implements LRProps {
  public weights: Tensor;
  public mean: Tensor | undefined;
  public variance: Tensor | undefined;
  public costHistory: number[];

  constructor(public features: Tensor, public labels: Tensor, public options: Options) {
    this.features = this.processFeatures(features);
    this.weights = tf.zeros([<number>this.features.shape[1], 1]);
    this.costHistory = [];
  }

  gradientDescent(features: Tensor, labels: Tensor) {
    // 2D array of mx + b
    const currentGuesses = features.matMul(this.weights).sigmoid();
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
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations: Tensor): Tensor {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary);
  }

  /**
   * Test existing logistic regression by counting unmatched results
   * and returning % of accuracy
   */
  test(testFeatures: Tensor, testLabels: Tensor): number {
    const predictions = this.predict(testFeatures);
    const incorrect = <number>predictions
      .sub(testLabels)
      .abs()
      .sum()
      .arraySync();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
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

  private recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();
    const termOne = this.labels.transpose().matMul(guesses.log());
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );
    const cost = <number>termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .arraySync();

    this.costHistory.unshift(cost);
  }

  private updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

export default LogisticRegression;
