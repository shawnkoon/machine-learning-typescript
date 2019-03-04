// Lib
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';
import plot from 'node-remote-plot';

// App
import loadCSV from '../data/load-csv';
import LinearRegression from './LinearRegression';

console.log('\nExecuting Linear regression analysis 📈\n');

const { features, labels, testFeatures, testLabels } = loadCSV('cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg']
});

const featuresTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);
const testFeaturesTensor = tf.tensor(<Array<Array<number>>>testFeatures);
const testLabelsTensor = tf.tensor(<Array<Array<number>>>testLabels);

const regression = new LinearRegression(featuresTensor, labelsTensor, {
  batchSize: 10,
  iterations: 3,
  learningRate: 0.1
});

/**
 * Trains the LinearRegression class.
 */
regression.train();

/**
 * Test to see if trained regression matches up with test set.
 */
const r2 = regression.test(testFeaturesTensor, testLabelsTensor);
console.log('> r2 is', r2);

/**
 * Draw graph of MSE / iteration relationship to ./plot.png
 */
plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
});
console.log('> Finished creating MSE/iteration graph to plot.png');

/**
 * Predict MPG value of observation Tensor set you pass in.
 */
console.log(
  `> Predict result ${regression.predict(tf.tensor([[120, 2, 380]])).toString()}`
);

console.log('\nFinished Linear regression analysis 📉');
