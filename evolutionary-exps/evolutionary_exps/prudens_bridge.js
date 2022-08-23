/////////// Bridge code for Python-Prudens communication via NodeJS ///////////
var fs = require('fs');
var data = fs.readFileSync(0, 'utf-8');
const inputs = JSON.parse(data);

for (var kb of inputs["kbs"]) {
    kb["constraints"] = new Map();
    var resultsForKB = [];
    for (const context of inputs["data"]) {
        const output = forwardChaining(kb, context);
        resultsForKB.push(Object.keys(output["graph"]));
    }
    console.log(JSON.stringify(resultsForKB));
}
