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
}

class LinearRegression implements LRProps {
  public weights: Tensor;

  constructor(
    private features: Tensor,
    private labels: Tensor,
    private options: Options
  ) {
    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1);
    this.weights = tf.zeros([2, 1]);
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  /**
   * Version #1 of this function can be found git hash f0e85eb.
   */
  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }
}

export default LinearRegression;
