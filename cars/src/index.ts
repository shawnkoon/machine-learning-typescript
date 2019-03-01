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

const featuresTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);
const testFeaturesTensor = tf.tensor(<Array<Array<number>>>testFeatures);
const testLabelsTensor = tf.tensor(<Array<Array<number>>>testLabels);

const regression = new LinearRegression(featuresTensor, labelsTensor, {
  learningRate: 0.0001,
  iterations: 100
});

regression.train();

regression
  .test(testFeaturesTensor, testLabelsTensor)
  .then(r2 => console.log('r2 is', r2));
