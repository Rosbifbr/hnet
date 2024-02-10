//Generic data normalizer for classifier neural networks
const fs = require('fs')

let inp = fs.readFileSync(process.argv[2], 'utf8')
let out = []

//Generate or import JSON
if (process.argv[3]) conversion_map = JSON.parse(fs.readFileSync(process.argv[3]))
else conversion_map = {}

for (line of inp.split('\n')){
	let newLine = []
	for (field of line.split(',')){
		if (!isNaN(parseInt(field))) newLine.push(parseInt(field))
		else {
			if(conversion_map[field.trim()] !== undefined) newLine.push(conversion_map[field.trim()])
			else conversion_map[field.trim()] = Math.random()
		}
	}
	out.push(newLine.join(','))
}

//Logging tranlsations
fs.writeFileSync('map.json', JSON.stringify(conversion_map))
fs.writeFileSync(process.argv[2] + '.out', out.join('\n'))
