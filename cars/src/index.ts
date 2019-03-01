// Lib
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// App
import loadCSV from './load-csv';
import LinearRegression from './LinearRegression';

const { features, labels, testFeatures, testLabels } = loadCSV('cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(
  tf.tensor(features),
  tf.tensor(labels),
  { learningRate: 0.0001, iterations: 100 }
);

regression.train();

const weights = <number[][]>regression.weights.arraySync();

console.log('\nUpdated M is:', weights[1][0], 'Updated B is:', weights[0][0]);
