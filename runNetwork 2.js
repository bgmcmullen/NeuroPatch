function sigmoid(x) {
	return x.map(function(row) {return row.map(function(num) { return 1 / (1 + Math.exp(-num))})});
}

function softplus(x) {
    return x.map(function(row) {return row.map(function(num) {return Math.log(1 + Math.exp(num))})});
}

function buildArray(length, createFunction) {
	var arr = [];
	for (var i = 0; i < length; i++) {
		arr.push(createFunction());
	}
	return arr;
}

function silu(x) {
    return x.map(function(row) {return row.map(function(num) {return num / (1 + Math.exp(-num))})});
}

function siluDerivative(x) {
    return x.map(function(row) {return row.map(function(num) {
        var sigmoidValue = 1 / (1 + Math.exp(-num));
        return sigmoidValue + num * sigmoidValue * (1 - sigmoidValue);
    })});
}

function tanh(x) {
    var ePos = Math.exp(x);
    var eNeg = Math.exp(-x);
    return (ePos - eNeg) / (ePos + eNeg);
}

function tanhNormalized(x) {
    return x.map(function(row) {return row.map(function(num) {return (tanh(num) + 1) / 2})});
}

function tanhNormalizedDerivative(x) {
    return x.map(function(row) {
        return row.map(function(num) { 
            return (1 - Math.pow(tanh(num), 2)) / 2;
        });
    });
}

function transposeMatrix(matrix) {
  var height = matrix[0].length;
  if (!height > 0) {
    throw new Error("Invalid Matrix transposition");
  }
  var result = buildArray(height, function() {return []});
  for (var i = 0; i < matrix.length; i++) {
    for (var j = 0; j < matrix[0].length; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

function dotProductMatricies(X, Y) {
  if (X[0].length !== Y.length) {
    throw new Error("Invalid Matrix dot multiplication");
  }
  var result = buildArray(X.length, function() {return []});
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


function runNetwork(input, weights, activation, outputActivation) {
	try {
		input = JSON.parse(input);
		weights = JSON.parse(weights);
	} catch(error) {
		return;
	}
	
	if(activation === "sigmoid") {
		activationFunction = sigmoid;
	}
	
	if(activation === "softplus") {
		activationFunction = softplus;
	}
	
	if(activation === "tanhNormalized") {
		activationFunction = tanhNormalized;
	}
	
	if(activation === "silu") {
		activationFunction = silu;
	}
	
	if(outputActivation === "sigmoid") {
		outputFunction = sigmoid;
	}
	
	if(outputActivation === "softplus") {
		outputFunction = softplus;
	}
	
	if(outputActivation === "tanhNormalized") {
		outputFunction = tanhNormalized;
	}
	
	if(outputActivation === "silu") {
		outputFunction = silu;
	}
		
  	var currentOutput = input;
  	for (var i = 0; i < weights.length; i++) {
    	var currentInput = dotProductMatricies(currentOutput, weights[i]);
		if(i === weights.length - 1) {
			currentOutput = outputFunction(currentInput);
		} else {
			currentOutput = activationFunction(currentInput);
		}
    	
  	}
		outlet(0, currentOutput[0]);
}