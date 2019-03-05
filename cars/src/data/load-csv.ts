/**
 * Original js file was provided by StephenGrider.
 * https://github.com/StephenGrider/MLKits/blob/master/regression/load-csv.js
 */
import fs from 'fs';
import _ from 'lodash';
import shuffleSeed from 'shuffle-seed';
import path from 'path';

function extractColumns(data, columnNames) {
  const headers = _.first(data);

  const indexes = _.map(columnNames, column => headers.indexOf(column));
  const extracted = _.map(data, row => _.pullAt(row, indexes));

  return extracted;
}

export interface LoadOptionProps {
  dataColumns?: string[];
  labelColumns?: string[];
  converters?: { [column: string]: (value: any) => any };
  shuffle?: boolean;
  splitTest?: number;
}

export interface LoadResult {
  features: number[][];
  labels: number[][];
  testFeatures?: number[][];
  testLabels?: number[][];
}

export default function loadCSV(
  filename: string,
  {
    dataColumns = [],
    labelColumns = [],
    converters = {},
    shuffle = false,
    splitTest = 0
  }: LoadOptionProps
): LoadResult {
  let data: any = fs.readFileSync(path.resolve(__dirname, filename), {
    encoding: 'utf-8'
  });
  data = _.map(data.split('\n'), d => d.split(','));
  data = _.dropRightWhile(data, val => _.isEqual(val, ['']));
  const headers = _.first(data);

  data = _.map(data, (row, index) => {
    if (index === 0) {
      return row;
    }
    return _.map(row, (element, index) => {
      if (converters[headers[index]]) {
        const converted = converters[headers[index]](element);
        return _.isNaN(converted) ? element : converted;
      }

      const result = parseFloat(element.replace('"', ''));
      return _.isNaN(result) ? element : result;
    });
  });

  let labels = extractColumns(data, labelColumns);
  data = extractColumns(data, dataColumns);

  data.shift();
  labels.shift();

  if (shuffle) {
    data = shuffleSeed.shuffle(data, 'phrase');
    labels = shuffleSeed.shuffle(labels, 'phrase');
  }

  if (splitTest) {
    const trainSize = _.isNumber(splitTest) ? splitTest : Math.floor(data.length / 2);

    return {
      features: data.slice(trainSize),
      labels: labels.slice(trainSize),
      testFeatures: data.slice(0, trainSize),
      testLabels: labels.slice(0, trainSize)
    };
  } else {
    return { features: data, labels };
  }
}
