outlets = 2;

function sigmoid(x) {
  return x.map(function (row) { return row.map(function (num) { return sigmoidFunction(num) }) });
}

function absMatrixMean(x) {
  var total = 0;
  var values = 0;
  for (var i = 0; i < x.length; i++) {
    for (var j = 0; j < x[i].length; j++) {
      values++;
      if (x[i][j] > 0) {
        total += x[i][j];
      } else {
        total -= x[i][j];
      }
    }
  }
  return total / values;
}

function sigmoidFunction(x) {
  return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
  var ePos = Math.exp(x);
  var eNeg = Math.exp(-x);
  return (ePos - eNeg) / (ePos + eNeg);
}

function tanhNormalized(x) {
  return x.map(function (row) { return row.map(function (num) { return (tanh(num) + 1) / 2 }) });
}

function tanhNormalizedDerivative(x) {
  return x.map(function (row) {
    return row.map(function (num) {
      return (1 - Math.pow(tanh(num), 2)) / 2;
    });
  });
}

function silu(x) {
  return x.map(function (row) { return row.map(function (num) { return num / (1 + Math.exp(-num)) }) });
}

function siluDerivative(x) {
  return x.map(function (row) {
    return row.map(function (num) {
      var sigmoidValue = 1 / (1 + Math.exp(-num));
      return sigmoidValue + num * sigmoidValue * (1 - sigmoidValue);
    })
  });
}

function softplus(x) {
  return x.map(function (row) { return row.map(function (num) { return Math.log(1 + Math.exp(num)) }) });
}

function sigmoidDerivative(x) {
  return x.map(function (row) {
    return row.map(function (num) {
      var sigmoidValue = sigmoidFunction(num);
      return sigmoidValue * (1 - sigmoidValue);
    });
  });
}

function buildArray(length, createFunction) {
  var arr = [];
  for (var i = 0; i < length; i++) {
    arr.push(createFunction());
  }
  return arr;
}

function transposeMatrix(matrix) {
  var height = matrix[0].length;
  if (!height > 0) {
    throw new Error("Invalid Matrix transposition");
  }
  var result = buildArray(height, function () { return [] });
  for (var i = 0; i < matrix.length; i++) {
    for (var j = 0; j < matrix[0].length; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

function multiplyMatrices(X, Y) {
  if (X[0].length !== Y.length) {
    throw new Error("Invalid Matrix multiplication");
  }
  var result = buildArray(X.length, function () { return [] });
  var Y_T = transposeMatrix(Y);
  for (var i = 0; i < X.length; i++) {
    for (var j = 0; j < Y_T.length; j++) {
      var sum = 0;
      for (var k = 0; k < X[i].length; k++) {
        sum += X[i][k] * Y_T[j][k];
      }
      result[i].push(sum);
    }
  }
  return result;
}

function subtractMatricies(X, Y) {
  if (X.length !== Y.length || X[0].length !== Y[0].length) {
    throw new Error("Invalid Matrix subtraction");
    return;
  }
  var result = buildArray(X.length, function () { return [] });
  for (var i = 0; i < X.length; i++) {
    for (var j = 0; j < X[0].length; j++) {
      result[i][j] = X[i][j] - Y[i][j];
    }
  }
  return result;
}

function multiplyMatricies(X, Y) {
  if (X.length !== Y.length || X[0].length !== Y[0].length) {
    throw new Error("Invalid Matrix multiplication");
  }
  var result = buildArray(X.length, function () { return [] });
  for (var i = 0; i < X.length; i++) {
    for (var j = 0; j < X[0].length; j++) {
      result[i][j] = X[i][j] * Y[i][j];
    }
  }
  return result;
}

function addMatricies(X, Y) {
  if (X.length !== Y.length || X[0].length !== Y[0].length) {
    throw new Error("Invalid Matrix addition");
  }
  var result = buildArray(X.length, function () { return [] });
  for (var i = 0; i < X.length; i++) {
    for (var j = 0; j < X[0].length; j++) {
      result[i][j] = X[i][j] + Y[i][j];
    }
  }
  return result;
}

function multiplyMatrixByFloat(matrix, num) {
  var result = buildArray(matrix.length, function () { return [] });
  for (var i = 0; i < matrix.length; i++) {
    for (var j = 0; j < matrix[0].length; j++) {
      result[i][j] = matrix[i][j] * num;
    }
  }
  return result;
}

function testNetwork(input, weights, inputs, outputs, activationFunction, outputFunction) {
  for (var i = 0; i < weights.length; i++) {
    inputs[i] = multiplyMatrices(i < 1 ? input : outputs[i - 1], weights[i]);
    if (i === weights.length - 1) {
      outputs[i] = outputFunction(inputs[i]);
    } else {
      outputs[i] = activationFunction(inputs[i]);
    }

  }
  return outputs[outputs.length - 1];
}

function backpropagation(error, weights, gradients, inputs, outputs, derivativeFunction, outputDerivative) {
  for (var i = weights.length - 1; i >= 0; i--) {
    gradients[i] = multiplyMatricies(i === weights.length - 1 ? error :
      multiplyMatrices(gradients[i + 1], transposeMatrix(weights[i + 1])),
      i === weights.length - 1 ? outputDerivative(inputs[i]) : derivativeFunction(inputs[i]));
  }
}

function updateWeights(input, outputs, weights, gradients, learningRate) {
  for (var i = weights.length - 1; i >= 0; i--) {
    weights[i] = addMatricies(weights[i], multiplyMatrixByFloat(multiplyMatrices(transposeMatrix(i < 1 ? input : outputs[i - 1]), gradients[i]), learningRate));
  }
}

function train(inputMatrix, outputMatrix, config) {
  var X = inputMatrix;
  var y = outputMatrix;
  var inputSize = X[0].length;
  var outputSize = y[0].length;
  var hiddenSizes = config.hiddenSizes;
  const numLayers = hiddenSizes.length + 1;
  hiddenSizes.unshift(inputSize);
  hiddenSizes.push(outputSize);
  var sizes = config.hiddenSizes;
  var gradients = Array(numLayers);
  var outputs = Array(numLayers);
  var inputs = Array(numLayers);
  var epochs = config.epochs;
  var learningRate = config.learningRate;
  var activationFunction = config.activationFunction;
  var derivativeFunction = config.derivativeFunction;
  var outputFunction = config.outputFunction;
  var outputDerivative = config.outputDerivative;
  var weights = config.weights || [];


  if (weights.length === 0) {
    for (var i = 0; i < sizes.length - 1; i++) {
      weights[i] = buildArray(sizes[i], function () { return buildArray(sizes[i + 1], function () { return Math.random() * 2 - 1 }) });
    }
  }


  // Training loop
  for (var i = 0; i < epochs; i++) {


    testNetwork(X, weights, inputs, outputs, activationFunction, outputFunction);

    // Calculate error
    var error = subtractMatricies(y, outputs[numLayers - 1]);

    backpropagation(error, weights, gradients, inputs, outputs, derivativeFunction, outputDerivative);

    updateWeights(X, outputs, weights, gradients, learningRate);
  }

  outlet(1, absMatrixMean(error));

  return weights;
}




function colorBackground(input, activationFunction, outputFunction, weights, colorInterval, lcd) {
  var currentOutput = [[input[0] / colorInterval, input[1] / colorInterval]];
  for (var i = 0; i < weights.length; i++) {
    var currentInput = multiplyMatrices(currentOutput, weights[i]);
    if (i === weights.length - 1) {
      currentOutput = outputFunction(currentInput);
    } else {
      currentOutput = activationFunction(currentInput);
    }

  }
  lcd.message("frgb", currentOutput[0][0] * 255, currentOutput[0][1] * 255, currentOutput[0][2] * 255);
  lcd.message("pensize", 1, 1);
  lcd.message("paintrect", (input[0] / colorInterval) * 400,
    (input[1] / colorInterval) * 400,
    (input[0] / colorInterval) * 400 + 400 / colorInterval,
    (input[1] / colorInterval) * 400 + 400 / colorInterval);
}


function trainNetwork(input, output, layers, epochs, learningRate, activation, weights, outputActivation) {

  var activationFunction = null;
  var derivativeFunction = null;



  if (weights === "empty") {
    weights = [];
  } else {
    weights = JSON.parse(weights);
  }



  if (activation === "sigmoid") {
    activationFunction = sigmoid;
    derivativeFunction = sigmoidDerivative;
  }

  if (activation === "softplus") {
    activationFunction = softplus;
    derivativeFunction = sigmoid;
  }

  if (activation === "tanhNormalized") {
    activationFunction = tanhNormalized;
    derivativeFunction = tanhNormalizedDerivative;
  }

  if (activation === "silu") {
    activationFunction = silu;
    derivativeFunction = siluDerivative;
  }

  if (outputActivation === "sigmoid") {
    outputFunction = sigmoid;
    outputDerivative = sigmoidDerivative;
  }

  if (outputActivation === "softplus") {
    outputFunction = softplus;
    outputDerivative = sigmoid;
  }

  if (outputActivation === "tanhNormalized") {
    outputFunction = tanhNormalized;
    outputDerivative = tanhNormalizedDerivative;
  }

  if (outputActivation === "silu") {
    outputFunction = silu;
    outputDerivative = siluDerivative;
  }

  var X = JSON.parse(input);
  var y = JSON.parse(output);
  layers = JSON.parse(layers);

  var config = {
    hiddenSizes: layers,
    epochs: epochs,
    learningRate: learningRate,
    activationFunction: activationFunction,
    derivativeFunction: derivativeFunction,
    outputFunction: outputFunction,
    outputDerivative: outputDerivative,
    weights: weights
  }


  var weights = train(X, y, config);

  var lcd = this.patcher.getnamed("lcd_panel2");

  var colorInterval = 100;

  for (var i = 0; i < colorInterval; i++) {
    for (var j = 0; j < colorInterval; j++) {
      colorBackground([i, j], activationFunction, outputFunction, weights, colorInterval, lcd);
    }
  }
  outlet(0, JSON.stringify(weights));
}

