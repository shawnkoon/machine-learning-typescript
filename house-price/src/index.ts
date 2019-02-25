interface IStudent {
  id?: number;
  name: string;
  age: number;
}

const x: IStudent = { name: 'hi', age: 1 };

console.log('NUmber x is ', x);
