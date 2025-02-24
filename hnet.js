////////////////////////////////////////////////
// Hnet - A simple neural network package.    //
// Version: 0.1                               //
// Author:  Rodrigo Ourique                   //
////////////////////////////////////////////////

//Application-specific functions.
const activations = {
  relu: {
    function: x => x > 0 ? x : 0,
    derivative: x => x > 0 ? 1 : 0
  },
  sigmoid: {
    function: x => 1 / (1 + Math.exp(-x)),
    derivative: x => {
      const s = 1 / (1 + Math.exp(-x))
      return s * (1 - s)
    }
  },
  tanh: {
    function: x => Math.tanh(x),
    derivative: x => 1 - Math.pow(Math.tanh(x), 2)
  },
  linear: {
    function: x => x,
    derivative: x => 1
  }
}

const costs = {
  mse: {
    function: (y_true, y_pred) => {
      let sum = 0
      for (let i = 0; i < y_true.length; i++) {
        sum += Math.pow(y_true[i] - y_pred[i], 2)
      }
      return sum / y_true.length
    },
    derivative: (y_true, y_pred) => {
      const result = []
      for (let i = 0; i < y_true.length; i++) {
        result.push(2 * (y_pred[i] - y_true[i]) / y_true.length)
      }
      return result
    }
  },
  cross_entropy: {
    function: (y_true, y_pred) => {
      let loss = 0
      for (let i = 0; i < y_true.length; i++) {
        // Add small epsilon to avoid log(0)
        const epsilon = 1e-15
        y_pred[i] = Math.max(Math.min(y_pred[i], 1 - epsilon), epsilon)
        loss += y_true[i] * Math.log(y_pred[i]) + (1 - y_true[i]) * Math.log(1 - y_pred[i])
      }
      return -loss
    },
    // For sigmoid activation, this simplifies to y_pred - y_true
    derivative: (y_true, y_pred) => {
      const result = []
      for (let i = 0; i < y_true.length; i++) {
        result.push(y_pred[i] - y_true[i])
      }
      return result
    }
  },
  mae: {
    function: (y_true, y_pred) => {
      let sum = 0
      for (let i = 0; i < y_true.length; i++) {
        sum += Math.abs(y_true[i] - y_pred[i])
      }
      return sum / y_true.length
    },
    derivative: (y_true, y_pred) => {
      const result = []
      for (let i = 0; i < y_true.length; i++) {
        result.push(y_pred[i] > y_true[i] ? 1 : -1)
      }
      return result
    }
  }
}

//Globals - now fully customizable
const defaults = {
  activation: activations.sigmoid,
  cost: costs.cross_entropy,
  learning_rate: 0.1,
  momentum: 0.9,
  use_momentum: false
}

class Neuron {
  constructor() {
    this.bias = 1
    this.bias_change = 0 // For momentum
    this.weights = []
    this.weight_changes = [] // For momentum
    this.value = 0
    this.net_input = 0 // Store the weighted sum before activation
    this.delta = 0 // For backpropagation
  }
}

//Simple example with two hidden layers.
exports.example_topology = [
  [new Neuron(), new Neuron(), new Neuron()], //Input.
  [new Neuron(), new Neuron(), new Neuron()],
  [new Neuron(), new Neuron(), new Neuron()],
  [new Neuron()], //Output.
]

// Allow changing default settings
exports.configure = (config) => {
  Object.assign(defaults, config)
}

// Function to get available activation functions and cost functions
exports.get_available_functions = () => {
  return {
    activations: activations,
    costs: costs
  }
}

//We need to initialize input values on each neuron and randomize the weights. "Wire" the neurons together.
exports.init = (topology) => {
  for (let layer = 0; layer < topology.length; layer++) {
    //Won't map the weights of inputs as they don't have predecessors.
    if (layer == 0) continue

    //Assigning random weights.
    for (let neuron of topology[layer]) {
      neuron.bias = Math.random() * 2 - 1  // Initialize between -1 and 1
      neuron.bias_change = 0
      neuron.weights = []
      neuron.weight_changes = []
      for (let i = 0; i < topology[layer - 1].length; i++) {
        neuron.weights.push(Math.random() * 2 - 1)  // Initialize between -1 and 1
        neuron.weight_changes.push(0)
      }
    }
  }
  return topology
}

exports.feed_forward = (topology, inputs) => {
  //Mapping input values to input neurons.
  for (let e in inputs) {
    topology[0][e].value = inputs[e]
  }

  for (let layer = 1; layer < topology.length; layer++) {
    //Add behind neuron values multiplied by their weight, then apply activation function. 
    for (let n = 0; n < topology[layer].length; n++) {
      let neuron = topology[layer][n]
      let net_input = 0

      // Sum weighted inputs
      for (let i = 0; i < topology[layer - 1].length; i++) {
        let behind_neuron_value = topology[layer - 1][i].value
        net_input += neuron.weights[i] * behind_neuron_value
      }

      // Add bias
      net_input += neuron.bias

      // Store net input for backpropagation
      neuron.net_input = net_input

      // Apply activation function
      neuron.value = defaults.activation.function(net_input)
    }
  }

  //Getting output neuron values.
  return topology[topology.length - 1].map(n => n.value)
}

exports.train = (topology, inputs, outputs, epochs, callback) => {
  const errors = []

  for (let i = 0; i < epochs; i++) {
    const error = exports.backpropagate(topology, inputs, outputs)
    errors.push(error)

    // Call callback if provided (for monitoring training)
    if (callback && i % callback.frequency === 0) {
      callback.function(i, error, topology)
    }
  }

  return {
    topology: topology,
    errors: errors
  }
}

exports.backpropagate = (topology, inputs, outputs) => {
  let total_error = 0

  for (let i = 0; i < inputs.length; i++) {
    // 1. Feed forward
    exports.feed_forward(topology, inputs[i])

    // 2. Calculate output layer error
    let output_layer = topology[topology.length - 1]
    let output_values = output_layer.map(n => n.value)
    let target_values = outputs[i]

    // Calculate cost
    total_error += defaults.cost.function(target_values, output_values)

    // Calculate delta for output layer
    let cost_derivatives = defaults.cost.derivative(target_values, output_values)

    for (let n = 0; n < output_layer.length; n++) {
      let neuron = output_layer[n]
      // Delta = cost_derivative * activation_derivative
      neuron.delta = cost_derivatives[n] * defaults.activation.derivative(neuron.net_input)
    }

    // 3. Calculate hidden layer errors (backpropagate)
    for (let layer = topology.length - 2; layer > 0; layer--) {
      let current_layer = topology[layer]
      let next_layer = topology[layer + 1]

      for (let n = 0; n < current_layer.length; n++) {
        let neuron = current_layer[n]
        let error = 0

        // Sum error contributions from next layer
        for (let next_n = 0; next_n < next_layer.length; next_n++) {
          error += next_layer[next_n].delta * next_layer[next_n].weights[n]
        }

        // Calculate delta
        neuron.delta = error * defaults.activation.derivative(neuron.net_input)
      }
    }

    // 4. Update weights and biases
    update_weights(topology)
  }

  return total_error / inputs.length
}

// Function to update weights based on calculated deltas
function update_weights(topology) {
  const learning_rate = defaults.learning_rate
  const momentum = defaults.momentum
  const use_momentum = defaults.use_momentum

  for (let layer = 1; layer < topology.length; layer++) {
    let current_layer = topology[layer]
    let prev_layer = topology[layer - 1]

    for (let n = 0; n < current_layer.length; n++) {
      let neuron = current_layer[n]

      // Update weights
      for (let prev_n = 0; prev_n < prev_layer.length; prev_n++) {
        let delta_weight = learning_rate * neuron.delta * prev_layer[prev_n].value

        if (use_momentum) {
          // Apply momentum (add a fraction of the previous weight change)
          delta_weight += momentum * neuron.weight_changes[prev_n]
          neuron.weight_changes[prev_n] = delta_weight
        }

        neuron.weights[prev_n] += delta_weight
      }

      // Update bias
      let delta_bias = learning_rate * neuron.delta

      if (use_momentum) {
        delta_bias += momentum * neuron.bias_change
        neuron.bias_change = delta_bias
      }

      neuron.bias += delta_bias
    }
  }
}

// For prediction
exports.predict = (topology, input) => {
  return exports.feed_forward(topology, input)
}

// Calculate error on a dataset
exports.calculate_error = (topology, inputs, expected_outputs) => {
  let total_error = 0

  for (let i = 0; i < inputs.length; i++) {
    let outputs = exports.feed_forward(topology, inputs[i])
    total_error += defaults.cost.function(expected_outputs[i], outputs)
  }

  return total_error / inputs.length
}

// Export the Neuron constructor for creating custom topologies
exports.Neuron = Neuron
