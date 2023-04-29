//Application-specific functions
const sigmoid = x => 1 / (1 + Math.exp(-x))
const sigmoid_derivative = x => sigmoid(x) * (1 - sigmoid(x))
const cross_entropy_loss_function = (y_true, y_pred) => {
        let loss = 0
        for (let i in y_true) {
                loss += y_true[i] * Math.log(y_pred[i]) + (1 - y_true[i]) * Math.log(1 - y_pred[i])
        }
        return -loss
}

//Data-Structs
const neuron = {
        weights: [],
        activation_function: sigmoid,
	activation_derivative: sigmoid_derivative,
        value:0,
}

exports.base_topology = [
        [neuron.structuredClone(), neuron.structuredClone(),], //This is an input layer, not a hidden one.
        [neuron.structuredClone(), neuron.structuredClone(), neuron.structuredClone()],
        [neuron.structuredClone(), neuron.structuredClone(), neuron.structuredClone()],
        [neuron.structuredClone()], //Output 
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
                               // neuron_obj.weights.push(1) //TODO: REMOVE
                        }
                }
        }
        return topology
}

exports.feed_forward = (topology, inputs) => {
        let tmp_topology = topology.structuredClone() //FIXME: This copy is ugly and won't scale.
        for (i in tmp_topology[0]) tmp_topology[0][i].value = inputs[i]
        for (layer in tmp_topology) {
                if (layer == 0) continue //We won't calculate the weight of the inputs.
                for (let neuron of tmp_topology[layer]){
                        neuron.value = neuron.activation_function(neuron.weights.reduce((total, weight_value, index) => {
                                let behind_neuron_value = tmp_topology[layer-1][index].value
                                total += weight_value * behind_neuron_value
                        }))
                }
        }
        return tmp_topology.slice(-1).map(n => n.value) //Returns the value of the output neurons.
}

//TODO:
exports.backpropagate = (topology) => {

}
