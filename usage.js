//Run on the repo root
const hnet = require('./hnet.js')

// Example: XOR problem
const training_inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

const training_outputs = [
  [0],
  [1],
  [1],
  [0]
]

// Show available activation and cost functions
const available_functions = hnet.get_available_functions()
console.log("Available activation functions:", Object.keys(available_functions.activations))
console.log("Available cost functions:", Object.keys(available_functions.costs))

// Configure the network to use specific activation and cost functions
hnet.configure({
  activation: available_functions.activations.tanh,
  cost: available_functions.costs.mae,
  learning_rate: 0.05,
  use_momentum: true,
  momentum: 0.9
})

// Create a topology for XOR (2 inputs, 4 hidden, 1 output)
const xor_topology = [
  [new hnet.Neuron(), new hnet.Neuron()],           // Input layer
  [new hnet.Neuron(), new hnet.Neuron(),
  new hnet.Neuron(), new hnet.Neuron()], // Hidden layer
  [new hnet.Neuron()]                                 // Output layer
]

// Initialize the network
const my_topo = hnet.init(xor_topology)

// Train the network with a monitoring callback
console.log("Training the network...")
const epochs = 10000
const training_result = hnet.train(my_topo, training_inputs, training_outputs, epochs, {
  frequency: 1000,
  function: (epoch, error, topology) => {
    console.log(`Epoch ${epoch}, Error: ${error.toFixed(6)}`)
  }
})

// Test the network
console.log("\nTesting the network:")
for (let input of training_inputs) {
  const output = hnet.feed_forward(my_topo, input)
  console.log(`Input: [${input}], Output: ${output[0].toFixed(4)}, Expected: ${training_outputs[training_inputs.indexOf(input)]}`)
}

// Try the example from the original code
console.log("\nTesting with [20, 20]:")
console.log(hnet.feed_forward(my_topo, [20, 20]))

// Switch to different activation and cost functions
console.log("\nSwitching to ReLU activation and MSE cost...")
hnet.configure({
  activation: available_functions.activations.relu,
  cost: available_functions.costs.mse
})

// Create and train a new network with the new settings
const relu_topo = hnet.init(xor_topology)
hnet.train(relu_topo, training_inputs, training_outputs, 5000, {
  frequency: 1000,
  function: (epoch, error, topology) => {
    console.log(`Epoch ${epoch}, Error: ${error.toFixed(6)}`)
  }
})

// Show the results with the new activation/cost combo
console.log("\nTesting the ReLU network:")
for (let input of training_inputs) {
  const output = hnet.feed_forward(relu_topo, input)
  console.log(`Input: [${input}], Output: ${output[0].toFixed(4)}, Expected: ${training_outputs[training_inputs.indexOf(input)]}`)
}

// Create a custom activation and cost function
console.log("\nCreating custom activation and cost functions...")
hnet.configure({
  activation: {
    function: x => x / (1 + Math.abs(x)),  // Custom SoftSign activation
    derivative: x => 1 / Math.pow(1 + Math.abs(x), 2)
  },
  cost: {
    function: (y_true, y_pred) => {  // Huber loss
      const delta = 1.0
      let loss = 0
      for (let i = 0; i < y_true.length; i++) {
        const error = Math.abs(y_true[i] - y_pred[i])
        loss += error < delta ?
          0.5 * Math.pow(error, 2) :
          delta * (error - 0.5 * delta)
      }
      return loss / y_true.length
    },
    derivative: (y_true, y_pred) => {
      const delta = 1.0
      const result = []
      for (let i = 0; i < y_true.length; i++) {
        const error = y_true[i] - y_pred[i]
        result.push(Math.abs(error) < delta ?
          -error :
          error > 0 ? -delta : delta)
      }
      return result
    }
  },
  learning_rate: 0.02
})

// Create and train a network with custom functions
const custom_topo = hnet.init(xor_topology)
hnet.train(custom_topo, training_inputs, training_outputs, 5000, {
  frequency: 1000,
  function: (epoch, error, topology) => {
    console.log(`Epoch ${epoch}, Error: ${error.toFixed(6)}`)
  }
})

// Show the results with the custom functions
console.log("\nTesting the custom network:")
for (let input of training_inputs) {
  const output = hnet.feed_forward(custom_topo, input)
  console.log(`Input: [${input}], Output: ${output[0].toFixed(4)}, Expected: ${training_outputs[training_inputs.indexOf(input)]}`)
}
