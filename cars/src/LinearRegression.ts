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
  public m: number;
  public b: number;

  constructor(
    private features: Tensor,
    private labels: Tensor,
    private options: Options
  ) {
    this.m = 0;
    this.b = 0;
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }

  gradientDescent() {
    const features = <number[][]>this.features.arraySync();
    const labels = <number[][]>this.labels.arraySync();

    const bSlope =
      (_.sum(
        features.map(
          (feature, i) => this.m * feature[0] + this.b - labels[i][0]
        )
      ) *
        2) /
      features.length;

    const mSlope =
      (_.sum(
        features.map(
          (feature, i) =>
            -1 * feature[0] * (labels[i][0] - this.m * feature[0] + this.b)
        )
      ) *
        2) /
      features.length;

    this.m = this.m - this.options.learningRate * mSlope;
    this.b = this.b - this.options.learningRate * bSlope;
  }
}

export default LinearRegression;
