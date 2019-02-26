// Lib
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// App
import loadCSV, { LoadOptionProps } from './load-csv';

type Tensor = tf.Tensor;

interface IHousePriceKNN {
  print(): void;
  test(): void;
  knn(prediction: Tensor): number;
  standardize(target: Tensor, mean: Tensor, variance: Tensor): Tensor;
}

interface HousePriceKNNProps {
  k?: number;
  fileName: string;
  loadOptions: LoadOptionProps;
}

class HousePriceKNN implements IHousePriceKNN {
  private k: number;
  private features: Tensor;
  private labels: Tensor;
  private testFeatures: Tensor;
  private testLabels: Tensor;

  public constructor({ k = 10, fileName, loadOptions }: HousePriceKNNProps) {
    this.k = k;

    const data = loadCSV(fileName, loadOptions);

    this.features = tf.tensor(data.features);
    this.labels = tf.tensor(data.labels);
    this.testFeatures = tf.tensor(data.testFeatures);
    this.testLabels = tf.tensor(data.testLabels);
  }

  public knn(prediction: Tensor): number {
    const { mean, variance } = tf.moments(this.features, 0);

    const scaledPrediction: Tensor = this.standardize(
      prediction,
      mean,
      variance
    );

    return (
      this.standardize(this.features, mean, variance)
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .expandDims(1)
        .concat(this.labels, 1)
        .unstack()
        .sort((a: Tensor, b: Tensor) =>
          a.arraySync()[0] > b.arraySync()[0] ? 1 : -1
        )
        .slice(0, this.k)
        .reduce((acc: number, obj: Tensor) => acc + obj.arraySync()[1], 0) /
      this.k
    );
  }

  public standardize(target: Tensor, mean: Tensor, variance: Tensor): Tensor {
    return target.sub(mean).div(variance.pow(0.5));
  }

  public print(): void {
    console.log(this.testFeatures);
    console.log(this.testLabels);
  }

  public test(): void {
    const x = <number[][]>this.testFeatures.arraySync();
    const testTarget = <number[][]>this.testLabels.arraySync();

    x.forEach((testFeatures, i) => {
      const predictionPrice = this.knn(tf.tensor(testFeatures));
      const realPrice = testTarget[i][0];
      const err = ((realPrice - predictionPrice) / realPrice) * 100;
      console.log(
        'Prediction:',
        predictionPrice,
        'Real:',
        realPrice,
        `Error: ${err}%`
      );
    });
  }
}

const analyzer = new HousePriceKNN({
  k: 10,
  fileName: 'kc_house_data.csv',
  loadOptions: {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
  }
});

// analyzer.print();
analyzer.test();
