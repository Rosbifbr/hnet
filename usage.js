//Run on the repo root
const hnet = require('./hnet.js')

var my_topo = hnet.init(hnet.example_topology)
console.log(
    hnet.feed_forward(my_topo,[20,20])
)