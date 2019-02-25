// Lib
import * as tf from '@tensorflow/tfjs-core';

// App
import loadCSV, { LoadOptionProps } from './load-csv';

type Tensor = tf.Tensor;

interface IHousePriceKNN {
  print(): void;
  test(): void;
  knn(predictionTensor: Tensor): number;
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

  public knn(predictionTensor: Tensor): number {
    return (
      this.features
        .sub(predictionTensor)
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

  public print(): void {
    console.log(this.testFeatures);
    console.log(this.testLabels);
  }

  public async test(): Promise<void> {
    const x = await this.testFeatures.array();
    const y = await this.testLabels.array();
    console.log('Guess', this.knn(tf.tensor(x[0])), y[0]);
  }
}

const analyzer = new HousePriceKNN({
  k: 10,
  fileName: 'kc_house_data.csv',
  loadOptions: {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
  }
});

analyzer.print();
analyzer.test();
