// Lib
import '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs';

// App
import loadCSV from '../data/load-csv';
import LogisticRegression from './LogisticRegression';

console.log('\nExecuting Logistic regression analysis 📈\n');

const { features, labels, testFeatures, testLabels } = loadCSV('cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],
  shuffle: true,
  splitTest: 50,
  converters: {
    passedemissions: value => {
      return value === 'TRUE' ? 1 : 0;
    }
  }
});

const featuresTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);
const testFeaturesTensor = tf.tensor(<Array<Array<number>>>testFeatures);
const testLabelsTensor = tf.tensor(<Array<Array<number>>>testLabels);

const regression = new LogisticRegression(featuresTensor, labelsTensor, {
  learningRate: 0.5,
  iterations: 100,
  decisionBoundary: 0.6,
  batchSize: 10
});

regression.train();

console.log(`Accuracy : ${regression.test(testFeaturesTensor, testLabelsTensor) * 100}%`);

console.log('\nFinished Logistic regression analysis 📉');
