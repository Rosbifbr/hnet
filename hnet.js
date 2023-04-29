//Application-specific functions. I'll leave some defaults here for the problems that I want to solve.
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
//Neuron class. Maybe will completely ditch classical OOP later. 
class Neuron {
        constructor() {
                this.weights = []
                this.value = 0
        }
}

//FIXME: This has no reason to be initialized before the init function. Rewrite the function to accept topology "maps" and return the actual network.
exports.example_topology = [
        [new Neuron()], //Input.
        [new Neuron(), new Neuron(), new Neuron()],
        [new Neuron(), new Neuron(), new Neuron()],
        [new Neuron()], //Output.
]

//We need to initialize input values on each neuron and randomize the weights. "Wire" the neurons together.
exports.init = (topology) => {
        for (layer in topology){
                if (layer == 0) continue //Won't map the weights of inputs As they dont have predecessors.

                //Assigining random weights.
                for (let neuron in topology[layer]){
                        let neuron_obj = topology[layer][neuron]
                        for (index in topology[layer-1]) {
                                neuron_obj.weights.push(Math.random())
                        }
                }
        }
        return topology
}

exports.feed_forward = (topology, inputs) => {
        for (i in topology[0]) {
                topology[0][i].value = inputs[i]
        }

        for (layer in topology) {
                if (layer == 0) continue //Inputs have no weigths
                for (let neuron of topology[layer]){
                        neuron.value = defaults.activation_function(neuron.weights.reduce((total, value, i) => {
                                let behind_neuron_value = topology[layer-1][i].value
                                return total + (value * behind_neuron_value)
                        }))
                }
        }
        return topology.slice(-1)[0].map(n => n.value) //Getting output neuron values.
}

//TODO:
exports.backpropagate = (topology) => {

}
