const parsers = require("../prudens-local/parsers.js");
const prudens = require("../prudens-local/prudens.js");
const { performance } = require("perf_hooks");
const fs = require("fs");
const dTree = require("decision-tree");

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

function getActivePath(model, context) {
    let sign;
    let val = context.includes(model["name"]);
    let currentNode = model;
    sign = "-";
    if (val) {
        sign = "";
    }
    let currentPathNode = {
        name: sign + model["name"],
        coverage: 1.0,
        prob: 1.0,
    };
    const sampleSize = model["sampleSize"];
    const fullPath = [];
    while (currentNode["type"] !== "result") {
        // console.log("context:", context, "\nval:", val);
        // console.log("currentNode['vals']:", currentNode["vals"]);
        fullPath.push(currentPathNode);
        for (const nodeVal of currentNode["vals"]) {
            // console.log("nodeVal:", nodeVal);
            if (val == nodeVal["name"]) {
                currentNode = nodeVal["child"];
                val = context.includes(currentNode["name"]);
                sign = "-";
                if (val) {
                    sign = "";
                }
                currentPathNode = {
                    name: sign + currentNode["name"],
                    coverage: nodeVal["sampleSize"] / sampleSize,
                    prob: nodeVal["prob"],
                };
                break;
            }
        }
    }
    // console.log("activePath:", fullPath, "context:", context);
    return {
        activePath: fullPath,
        prediction: currentNode["val"],
    };
}

function getRuleFromTree(model, context, agentPrediction, agentExplanation, advicePolicy) { // Context is a list of strings.
    const activePathObject = getActivePath(model, context);
    // console.log("getRuleFromTree");
    if (agentPrediction === activePathObject["prediction"]) {
        return undefined;
    }
    // console.log("here");
    return advicePolicy(activePathObject["activePath"], activePathObject["prediction"], agentExplanation["body"]); // FIXME Ensure that agentExplanation is indeed a JSON Object with a "body" field.
}

function translateContext(context) {
    const labelledContext = {};
    for (const item of context) {
        if (item.charAt(0) === "-") {
            labelledContext[item.substring(1)] = 0;
        } else {
            labelledContext[item] = 1;
        }
    }
    return labelledContext;
}

function labelWithTree(treeJson, data, isCoaching) {
    const tree = new dTree(treeJson);
    let prediction, context;
    const lData = [];
    for (const item of data) {
        if (isCoaching) {
            context = item;
        } else {
            context = item["context"];
        }
        prediction = tree.predict(translateContext(context));
        if (prediction === 1) {
            lData.push({
                context: context,
                label: "h",
            });
        } else if (prediction === 0) {
            lData.push({
                context: context,
                label: "-h",
            });
        } else {
            lData.push({
                context: context,
                label: null,
            });
        }
    }
    return lData;
}

function parseExplanation(explanation) {
    // console.log(explanation);
    const nameSplit = explanation.split("::");
    const bodySplit = nameSplit[1].split(" implies ");
    const body = [];
    for (const literal of bodySplit[0].split(",").filter(Boolean)) {
        body.push(literal.trim());
    }
    const head = bodySplit[1].trim().substring(0, bodySplit[1].trim().length - 1);
    return {
        name: nameSplit[0].trim(),
        body: body,
        head: head,
    };
}

function coachingCycle(treeJson, initialTheory, coachingSets, advicePolicy, train, test, initKb) {
    // let consecutiveApprovals = 0;
    let totalTries = 0;
    let advice, currentContext, inferences, startTime, endTime, contextObject, kbObject, rules = 0;
    // const advicePolicy = getMinimalPrefix;
    const maxIter = coachingSets.length;
    const predictionAccuracy = [], trainAcc = [], testAcc = [], coachAcc = [], trainCon = [], testCon = [], coachCon = [];
    const deductionTimes = [];
    const trainLab = labelWithTree(treeJson, train, false);
    const testLab = labelWithTree(treeJson, test, false);
    const coachLab = labelWithTree(treeJson, coachingSets, true);
    // const testAccuracy = [];
    while (totalTries < maxIter) {
        // console.log(totalTries, consecutiveApprovals);
        currentContext = coachingSets[totalTries];
        // console.log(currentContext);
        if (initialTheory === "") {
            // console.log("Empty Theory!");
            advice = getRuleFromTree(treeJson["model"], currentContext, undefined, {body: []}, advicePolicy);
            initialTheory += advice;
            rules++;
            predictionAccuracy.push(0);
            trainAcc.push(null);
            testAcc.push(null);
            coachAcc.push(null);
            trainCon.push(null);
            testCon.push(null);
            coachCon.push(null);
            deductionTimes.push(null);
            kbObject = parsers.parseKnowledgeBase("@KnowledgeBase\n" + initialTheory);
            // testAccuracy.push(0);
            // console.log(advice);
        } else {
            // console.log("context:", currentContext.join("; "), "\nKB:", initialTheory);
            contextObject = parsers.parseContext(currentContext.join(";") + ";")["context"];
            // kbObject = parsers.parseKnowledgeBase("@KnowledgeBase\n" + initialTheory);
            startTime = performance.now();
            // if (parsers.parseKnowledgeBase("@KnowledgeBase" + initialTheory)["type"] === "error") {
            //     console.log("error");
            // }
            // console.log(parsers.parseKnowledgeBase("@KnowledgeBase\n" + initialTheory));
            inferences = prudens.deduce(kbObject, contextObject, prudens.specificityPriorities); // TODO You may need this line!
            endTime = performance.now();
            deductionTimes.push(endTime - startTime);
            // console.log("inferences:", inferences["graph"]);
            // testAccuracy.push(predict(initialTheory, testSet));
            let agentExplanation, agentPrediction;
            if (Object.keys(inferences["graph"]).length === 0) {
                // console.log("no inference");
                agentExplanation = {body: []};
                // agentPrediction = undefined;
                // predictionAccuracy.push(0);
            } else if (Object.keys(inferences["graph"]).includes("-h")) {
                agentPrediction = 0;
                agentExplanation = parseExplanation(parsers.ruleToString(inferences["graph"]["-h"][inferences["graph"]["-h"].length - 1]));
                // predictionAccuracy.push(1);
            } else {
                agentPrediction = 1;
                agentExplanation = parseExplanation(parsers.ruleToString(inferences["graph"]["h"][inferences["graph"]["h"].length - 1]));
                // predictionAccuracy.push(1);
            }
            // console.log("agentPrediction:", agentPrediction);
            advice = getRuleFromTree(treeJson["model"], currentContext, agentPrediction, agentExplanation, advicePolicy);
            if (advice !== undefined) {
            //     consecutiveApprovals++;
            // } else {
                advice = "A" + totalTries + " " + advice.substring(advice.indexOf("::"));
                initialTheory += "\n" + advice;
                rules++;
                predictionAccuracy.push(0);
                trainAcc.push(testOnTrainTest(kbObject, train));
                testAcc.push(testOnTrainTest(kbObject, test));
                coachAcc.push(testOnCoaching(kbObject, coachingSets, initKb));
                trainCon.push(testOnTrainTest(kbObject, trainLab));
                testCon.push(testOnTrainTest(kbObject, testLab));
                coachCon.push(testOnTrainTest(kbObject, coachLab));
                // consecutiveApprovals = 0;
            } else {
                predictionAccuracy.push(1);
                trainAcc.push(trainAcc[trainAcc.length - 1]);
                testAcc.push(testAcc[testAcc.length - 1]);
                coachAcc.push(coachAcc[coachAcc.length - 1]);
                trainCon.push(trainCon[trainCon.length - 1]);
                testCon.push(testCon[testCon.length - 1]);
                coachCon.push(coachCon[coachCon.length - 1]);
            }
            kbObject = parsers.parseKnowledgeBase("@KnowledgeBase\n" + initialTheory);
            // console.log(initialTheory, kbObject);
            // console.log(coachCon);
        }
        totalTries++;
    }
    return {
        theory: initialTheory,
        rules: rules,
        literals: (initialTheory.match(/,/g) || []).length + rules,
        totalTries: totalTries,
        deductionTimes: deductionTimes,
        predictionConformance: predictionAccuracy,
        trainAccuracy: trainAcc,
        testAccuracy: testAcc,
        coachingAccuracy: coachAcc,
        trainConformance: trainCon,
        testConformance: testCon,
        coachingConformance: coachCon,
    };
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
        kbId = kbIds[i];
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

function forestExperiments(forestsFile, kbIdsPath, policiesPath, coachingPath, trainTestPath) {
    let kbIds = JSON.parse(fs.readFileSync(kbIdsPath, "utf8"));
    const forests = JSON.parse(fs.readFileSync(forestsFile, "utf8"));
    const kbs = JSON.parse(fs.readFileSync(policiesPath, "utf8"));
    const coachingSets = JSON.parse(fs.readFileSync(coachingPath, "utf8"));
    const trainTest = JSON.parse(fs.readFileSync(trainTestPath, "utf8"));
    let forest, coaching, startTime, endTime, kbId;
    const results = {};
    for (let i = 0; i < kbIds.length; i++) {
        kbId = kbIds[i]
        process.stdout.write("kbId: " + kbId + " | " + (i + 1) + " / " + kbIds.length + "\r");
        if (kbId === "kb_20_2_6_1") {
            continue;
        }
        forest = forests[kbId];
        coaching = coachingSets[kbId];
        startTime = performance.now();
        results[kbId] = {
            full: forestCoachingCycle(forest, "", coaching["full"], trainTest[kbId]["train"], trainTest[kbId]["test"], parsers.parseKnowledgeBase("@KnowledgeBase\n" + kbs[kbId]["kb"])),
        };
        endTime = performance.now();
    }
    if (!fs.existsSync("results")) {
        fs.mkdirSync("results", {recursive: true});
    }
    fs.writeFileSync("results/forestExps.json", JSON.stringify(results, null, 2));
    process.stdout.write("Complete! | Resullts @ results/forestExps.json\n");
}

function labelWithForest(forest, data, isCoaching) {
    let prediction, context, predObject;
    const lData = [];
    for (const item of data) {
        if (isCoaching) {
            context = item;
        } else {
            context = item["context"];
        }
        predObject = getWidestRule(forest, context);
        prediction = predObject["prediction"];
        if (prediction === 1) {
            lData.push({
                context: context,
                label: "h",
            });
        } else if (prediction === 0) {
            lData.push({
                context: context,
                label: "-h",
            });
        } else {
            lData.push({
                context: context,
                label: null,
            });
        }
    }
    return lData;
}

function forestCoachingCycle(forest, initialTheory, coachingSets, train, test, initKb) {
    let totalTries = 0;
    let advice, currentContext, inferences, startTime, endTime, contextObject, kbObject, forestResponse, rules = 0;
    const maxIter = coachingSets.length;
    const predictionAccuracy = [], trainAcc = [], testAcc = [], coachAcc = [], trainCon = [], testCon = [], coachCon = [];
    const deductionTimes = [];
    const trainLab = labelWithForest(forest, train, false);
    const testLab = labelWithForest(forest, test, false);
    const coachLab = labelWithForest(forest, coachingSets, true);
    while (totalTries < maxIter) {
        currentContext = coachingSets[totalTries];
        if (initialTheory === "") {
            advice = getWidestRule(forest, currentContext)["advice"];
            initialTheory += advice;
            rules++;
            predictionAccuracy.push(0);
            trainAcc.push(null);
            testAcc.push(null);
            coachAcc.push(null);
            trainCon.push(null);
            testCon.push(null);
            coachCon.push(null);
            deductionTimes.push(null);
            kbObject = parsers.parseKnowledgeBase("@KnowledgeBase\n" + initialTheory);
        } else {
            contextObject = parsers.parseContext(currentContext.join(";") + ";")["context"];
            startTime = performance.now();
            inferences = prudens.deduce(kbObject, contextObject, prudens.specificityPriorities);
            endTime = performance.now();
            deductionTimes.push(endTime - startTime);
            let agentExplanation, agentPrediction;
            if (Object.keys(inferences["graph"]).length === 0) {
                // console.log("no inference");
                // agentExplanation = {body: []};
                // agentPrediction = undefined;
                // predictionAccuracy.push(0);
            } else if (Object.keys(inferences["graph"]).includes("-h")) {
                agentPrediction = 0;
            } else {
                agentPrediction = 1;
            }
            forestResponse = getWidestRule(forest, currentContext);
            if (forestResponse["prediction"] === agentPrediction) {
                advice = undefined;
            } else {
                advice = forestResponse["advice"];
            }
            if (advice !== undefined) {
                advice = "A" + totalTries + " " + advice.substring(advice.indexOf("::"));
                initialTheory += "\n" + advice;
                rules++;
                predictionAccuracy.push(0);
                trainAcc.push(testOnTrainTest(kbObject, train));
                testAcc.push(testOnTrainTest(kbObject, test));
                coachAcc.push(testOnCoaching(kbObject, coachingSets, initKb));
                trainCon.push(testOnTrainTest(kbObject, trainLab));
                testCon.push(testOnTrainTest(kbObject, testLab));
                coachCon.push(testOnTrainTest(kbObject, coachLab));
            } else {
                predictionAccuracy.push(1);
                trainAcc.push(trainAcc[trainAcc.length - 1]);
                testAcc.push(testAcc[testAcc.length - 1]);
                coachAcc.push(coachAcc[coachAcc.length - 1]);
                trainCon.push(trainCon[trainCon.length - 1]);
                testCon.push(testCon[testCon.length - 1]);
                coachCon.push(coachCon[coachCon.length - 1]);
            }
            kbObject = parsers.parseKnowledgeBase("@KnowledgeBase\n" + initialTheory);
        }
        totalTries++;
    }
    return {
        theory: initialTheory,
        rules: rules,
        literals: (initialTheory.match(/,/g) || []).length + rules,
        totalTries: totalTries,
        deductionTimes: deductionTimes,
        predictionConformance: predictionAccuracy,
        trainAccuracy: trainAcc,
        testAccuracy: testAcc,
        coachingAccuracy: coachAcc,
        trainConformance: trainCon,
        testConformance: testCon,
        coachingConformance: coachCon,
    };
}

function getWidestRule(forest, context) { // Returns a rule with only the common premises in each tree of the forest.
    let activePathObject, advice, maxCovIndex, maxCov = -1,  prediction = 0, activePaths = [], coverages = [], predictions = [];
    for (const treeJson of forest["trees"]) {
        activePathObject = getActivePath(treeJson["model"], context);
        activePaths.push(activePathObject["activePath"]);
        predictions.push(activePathObject["prediction"])
        if (activePathObject["activePath"].length === 0) {
            return {
                advice: "",
                prediction: undefined,
            };
        }
        coverages.push(activePathObject["activePath"][activePathObject["activePath"].length - 1]["coverage"]);
        if (activePathObject["prediction"] === 1) {
            prediction++;
        } else {
            prediction--;
        }
    }
    if (prediction > 0) {
        prediction = 1;
    } else if (prediction < 0) {
        prediction = 0;
    } else {
        prediction = Math.floor(2 * Math.random());
    }
    for (let i=0; i<coverages.length; i++) {
        if (predictions[i] === prediction && coverages[i] > maxCov) {
            maxCov = coverages[i];
            maxCovIndex = i;
        }
    }
    advice = getFullPathRule(activePaths[maxCovIndex], prediction);
    return {
        advice: advice,
        prediction: prediction,
    }
}

function fullPathExps(kbIdsPath = "../data-gen/used-data/kbs.json", treesPath = "../data-gen/used-data/trainedTrees.json", policiesPath = "../data-gen/used-data/policies.json", coachingPath = "../data-gen/used-data/coaching.json", trainTestPath = "../data-gen/used-data/trainTest.json") {
    coachExperiments(getFullPathRule, "fullPathExps", kbIdsPath, treesPath, policiesPath, coachingPath, trainTestPath);
}

function minPrefixExps(kbIdsPath = "../data-gen/used-data/kbs.json", treesPath = "../data-gen/used-data/trainedTrees.json", policiesPath = "../data-gen/used-data/policies.json", coachingPath = "../data-gen/used-data/coaching.json", trainTestPath = "../data-gen/used-data/trainTest.json") {
    coachExperiments(getMinimalPrefix, "minPrefixExps", kbIdsPath, treesPath, policiesPath, coachingPath, trainTestPath);
}

function forestExps(forestPath = "../data-gen/used-data/trainedForests.json", kbIdsPath = "../data-gen/used-data/kbs.json", policiesPath = "../data-gen/used-data/policies.json", coachingPath = "../data-gen/used-data/coaching.json", trainTestPath = "../data-gen/used-data/trainTest.json") {
    forestExperiments(forestPath, kbIdsPath, policiesPath, coachingPath, trainTestPath);
}

module.exports = {
    fullPathExps,
    minPrefixExps,
    forestExps,
};