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
  { learningRate: 0.000001, iterations: 1000 }
);

regression.train();

console.log('\nUpdated M is:', regression.m, 'Updated B is:', regression.b);
