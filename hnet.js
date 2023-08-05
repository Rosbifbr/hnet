//Application-specific functions. I'll leave some defaults here for the problems that I want to solve.
const relu = x => x > 0 ? x : 0
const sigmoid = x => 1 / (1 + Math.exp(-x))
const sigmoid_derivative = x => sigmoid(x) * (1 - sigmoid(x))
const cross_entropy_loss = (y_true, y_pred) => {
        let loss = 0
        for (let i in y_true) {
                loss += y_true[i] * Math.log(y_pred[i]) + (1 - y_true[i]) * Math.log(1 - y_pred[i])
        }
        return -loss
}

//Globals
const defaults = {
        activation_function: sigmoid,
        activation_derivative: sigmoid_derivative,
        cost_function: cross_entropy_loss,
}

//TODO: Implement biases for the neurons. I was too lazy to have done it until now 
class Neuron {
        constructor() {
                this.bias = 1; //0 to 1
                this.weights = []
                this.value = 0
        }
}

//Simple example with two hidden layers.
exports.example_topology = [
        [new Neuron(), new Neuron(), new Neuron()],
        [new Neuron(), new Neuron(), new Neuron()],
        [new Neuron()], //Output.
]

//We need to initialize input values on each neuron and randomize the weights. "Wire" the neurons together.
exports.init = (topology) => {
        for (layer in topology) {
                //Won't map the weights of inputs As they dont have predecessors.
                if (layer == 0) continue

                //Assigining random weights.
                for (let neuron of topology[layer]) {
                        for (let neuron_ant of topology[layer - 1]) {
                                neuron.weights.push(Math.random())
                        }
                }
        }
        return topology
}

exports.feed_forward = (topology, inputs) => {
        //Mapping inputs as neurons for simplicity of the other. This is lazy.
        topology.unshift(inputs.map(e => {
                let neuron = new Neuron()
                neuron.value = e
                return neuron
        }))

        for (layer in topology) {
                //Inputs have no weigths
                if (layer == 0) continue
                
                //Add behind neuron values multiplied by their weight, then apply activation function. 
                for (let neuron of topology[layer]) {
                        neuron.value = defaults.activation_function(neuron.weights.reduce((total, weight, i) => {
                                let behind_neuron_value = topology[layer - 1][i].value
                                return total + (weight * behind_neuron_value)
                        }))
                }
        }
        return topology.slice(-1).map(n => n.value) //Getting output neuron values.
}

//TODO:
exports.backpropagate = (topology) => {

}
