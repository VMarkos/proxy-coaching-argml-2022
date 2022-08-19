const fs = require("fs");

function getFullPathRule(path, targetPrediction) {
    let ruleString = "A :: ";
    const body = [];
    for (let i=0; i<path.length; i++) {
        body.push(path[i]["name"]);
    }
    ruleString += body.join(", ");
    let sign = "-";
    if (targetPrediction) {
        sign = "";
    }
    return ruleString + " implies " + sign + "h;";
}

function getMinimalPrefix(path, targetPrediction, agentExplanationBody) {
    let ruleString = "A :: ";
    let i = 0, explCounter = 0;
    let currentNode = path[0];
    const body = [];
    // console.log("agentExplanation:", agentExplanationBody);
    while (i < path.length && (body.length === 0 || explCounter < agentExplanationBody.length + 1 || Math.abs(targetPrediction - currentNode["prob"]) > 0.5)) {
        // console.log("path:", path);
        body.push(currentNode["name"]);
        if (agentExplanationBody.includes(currentNode["name"]) || agentExplanationBody.length === 0) {
            explCounter++;
        }
        i++;
        currentNode = path[i];
    }
    ruleString += body.join(", "); // This is a really bad practice, you know...
    let sign = "-";
    if (targetPrediction) {
        sign = "";
    }
    return ruleString + " implies " + sign + "h;";
}

function coachExperiments(advicePolicy, name, kbIdsPath, treesPath, policiesPath, coachingPath, trainTestPath) {
    const kbIds = JSON.parse(fs.readFileSync(kbIdsPath, "utf8"));
    const trees = JSON.parse(fs.readFileSync(treesPath, "utf8"));
    const kbs = JSON.parse(fs.readFileSync(policiesPath, "utf8"));
    const coachingSets = JSON.parse(fs.readFileSync(coachingPath, "utf8"));
    const trainTest = JSON.parse(fs.readFileSync(trainTestPath, "utf8"));
    let tree, coaching, kbId;
    const results = {}
    for (let i = 0; i < kbIds.length; i++) {
        process.stdout.write("kbId: " + kbId + " | " + (i + 1) + " / " + kbIds.length + "\r");
        tree = trees[kbId]["tree"];
        coaching = coachingSets[kbId]; // Full as well as partial.
        // console.log(coaching["full"]);
        results[kbId] = {
            full: coachingCycle(tree, "", coaching["full"], advicePolicy, trainTest[kbId]["train"], trainTest[kbId]["test"], parsers.parseKnowledgeBase("@KnowledgeBase\n" + kbs[kbId]["kb"])),
            // partial: coachingCycle(tree, "", coaching["partial"], advicePolicy),
        };
    }
    if (!fs.existsSync("results")) {
        fs.mkdirSync("results", {recursive: true});
    }
    fs.writeFileSync("results/" + name + ".json", JSON.stringify(results, null, 2));
    process.stdout.write("Complete! | Resullts @ results/" + name + ".json\n");
}

function testOnTrainTest(kb, set) {
    let output, context, label, correct = 0, wrong = 0, abstain = 0, dilemmasCount = 0;
    for (let i=0; i<set.length; i++) {
        label = set[i]["label"];
        if (label === null) {
            continue;
        }
        // console.log(i, set[i]["context"]);
        context = parsers.parseContext(set[i]["context"].join(";") + ";")["context"];
        output = prudens.deduce(kb, context);
        inferences = Object.keys(output["graph"]);
        dilemmas = output["dilemmas"];
        // console.log(inferences, label);
        if (inferences.length === 0) {
            if (dilemmas !== undefined && dilemmas.length > 0) {
                dilemmasCount++;
            } else {
                abstain++;
            }
        } else if (inferences.includes(label)) {
            correct++;
        } else {
            wrong++;
        }
    }
    return {
        correct: correct,
        wrong: wrong,
        abstain: abstain,
        dilemmas: dilemmasCount,
    };
}

function testOnCoaching(kb, coaching, initKb) {
    const labelledSet = [];
    let inferences;
    for (const context of coaching) {
        inferences = Object.keys(prudens.deduce(initKb, parsers.parseContext(context.join(";") + ";")["context"])["graph"]);
        if (inferences.includes("h")) {
            label = "h";
        } else if (inferences.includes("-h")) {
            label = "-h";
        } else {
            label = null;
        }
        labelledSet.push({
            context: context,
            label: label,
        });
    }
    return testOnTrainTest(kb, labelledSet);
}

function fullPathExps(kbIdsPath = "../data-gen/used-data/kbs.json", treesPath = "../data-gen/used-data/trainedTrees.json", policiesPath = "../data-gen/used-data/policies.json", coachingPath = "../data-gen/used-data/coaching.json", trainTestPath = "../data-gen/used-data/trainTest.json") {
    coachExperiments(getFullPathRule, "fullPathExps", kbIdsPath, treesPath, policiesPath, coachingPath, trainTestPath);
}

function minPrefixExps(kbIdsPath = "../data-gen/used-data/kbs.json", treesPath = "../data-gen/used-data/trainedTrees.json", policiesPath = "../data-gen/used-data/policies.json", coachingPath = "../data-gen/used-data/coaching.json", trainTestPath = "../data-gen/used-data/trainTest.json") {
    coachExperiments(getMinimalPrefix, "minPrefixExps", kbIdsPath, treesPath, policiesPath, coachingPath, trainTestPath);
}

module.exports = {
    fullPathExps,
    minPrefixExps,
};