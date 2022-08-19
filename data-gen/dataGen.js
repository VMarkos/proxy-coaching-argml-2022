const fs = require("fs");
const parsers = require("../prudens-local/parsers.js");
const prudens = require("../prudens-local/prudens.js");
const dTree = require("decision-tree");
const { performance } = require("perf_hooks");

function randomSubset(set, k) {
    if (k > set.length) {
        return undefined;
    }
    let shuffled = set.slice(0);
    let i = set.length;
    let j;
    while (i--) {
        j = Math.floor((i + 1) * Math.random());
        temp = shuffled[i];
        shuffled[i] = shuffled[j];
        shuffled[j] = temp;
    }
    return shuffled.slice(0, k);
}

function randomSplit(set, p) { // Implements a random split of set to two parts with rates p and q := 1-p.
    let i, temp, n = set.length;
    const shuffled = set.slice(0);
    while (n--) {
        i = Math.floor((n + 1) * Math.random());
        temp = shuffled[i];
        shuffled[i] = shuffled[n];
        shuffled[n] = temp;
    }
    const split = Math.ceil(p * set.length);
    return {
        p: shuffled.slice(0, split),
        q: shuffled.slice(split, set.length),
    };
}

function randomStringReassignment(fullContext) { // Context as array of strings.
    const newContext = [];
    for (const literal of fullContext) {
        if (Math.random() < 0.5) {
            newContext.push("-" + literal);
        } else {
            newContext.push(literal);
        }
    }
    return newContext;
}


function randomSubcontext(context, k) {
    const mixedContext = [];
    for (const literal of context) {
        if (Math.random() < 0.5) {
            mixedContext.push(literal);
        } else {
            mixedContext.push(negateString(literal));
        }
    }
    return randomSubset(mixedContext, k);
}

function negateString(literal) {
    if (literal.charAt(0) === "-") {
        return literal.substring(1);
    }
    return "-" + literal;
}

function getContext(n) {
    const context = [];
    for (let i=0; i<n; i++) {
        context.push("p" + i);
    }
    return context;
}

function stackKbGenerator(fullContext, avgBatchSize, targetFlips, nStacks) {
    let newStack, stacks = 0, size = 0, flips = 0, head = "h", kb = "";
    const stackCounts = partition(targetFlips, nStacks), bodies = [], bodySizes = [];
    // console.log(stackCounts);
    while (flips < targetFlips && stacks < nStacks) {
        newStack = generateStack(fullContext, stackCounts[stacks], bodySizes, head, bodies, size, avgBatchSize);
        kb += newStack["stack"];
        size += newStack["size"];
        flips += stackCounts[stacks];
        stacks++;
    }
    return {
        kb: kb,
        fullContext: fullContext,
        size: size,
        flips: flips,
        stackCount: stacks,
        bodySizes: bodySizes,
        avgBatchSize: avgBatchSize,
    }
}

function partition(n, k) {
    if (k > n) {
        return undefined;
    }
    const splits = [0];
    let split, i = 0;;
    while (i < k - 1) {
        split = Math.ceil((n - 1) * Math.random());
        if (!splits.includes(split)) {
            splits.push(split);
            i++;
        }
    }
    splits.sort((a, b) => {return a - b;});
    splits.push(n);
    const partition = [];
    for (let i=0; i<splits.length-1; i++) {
        partition.push(splits[i + 1] - splits[i]);
    }
    return partition;
}

function generateStack(fullContext, depth, bodySizes, head, bodies, currentSize, avgBatchSize) {
    let rootBody, newStack, batchSize, stack = "", size = 0;
    rootBody = fullContext[Math.floor(fullContext.length * Math.random())];
    while (bodies.includes(rootBody)) {
        rootBody = fullContext[Math.floor(fullContext.length * Math.random())];
    }
    bodySizes.push(1);
    stack = "R" + (currentSize + size) + " :: " + rootBody + " implies " + head + ";\n";
    size++;
    let currentBody = [rootBody];
    for (let i=0; i<depth; i++) {
        head = negateString(head);
        batchSize = Math.ceil(3 * Math.random() - 2 + avgBatchSize);
        newStack = getExceptionBatch(fullContext, currentBody, head, size + currentSize, batchSize, bodies, bodySizes);
        stack += newStack["batch"];
        currentBody = newStack["tempBody"];
        size += batchSize;
    }
    return {
        stack: stack,
        size: size,
    };
}

function getExceptionBatch(fullContext, body, head, currentSize, batchSize, bodies, bodySizes) {
    let tempBody, joinedTempBody, testBody, newLiteral, sign, batch = "";
    for (let i=0; i<batchSize; i++) {
        tempBody = [];
        tempBody.push(...body);
        sign = "";
        if (Math.random() < 0.5) {
            sign = "-";
        }
        newLiteral = sign + fullContext[Math.floor(fullContext.length * Math.random())];
        testBody = [...tempBody];
        testBody.push(newLiteral);
        joinedTempBody = testBody.join(", ");
        while (tempBody.includes(newLiteral) || tempBody.includes(negateString(newLiteral)) || bodies.includes(joinedTempBody)) {
            sign = "";
            if (Math.random() < 0.5) {
                sign = "-";
            }
            newLiteral = sign + fullContext[Math.floor(fullContext.length * Math.random())];
            testBody = [...tempBody];
            testBody.push(newLiteral);
            joinedTempBody = testBody.join(", ");
        }
        tempBody.push(newLiteral);
        joinedTempBody = tempBody.join(", ");
        bodies.push(joinedTempBody);
        bodySizes.push(tempBody.length);
        batch += "R" + (currentSize + i) + " :: " + joinedTempBody + " implies " + head + ";\n";
    }
    return {
        batch: batch,
        tempBody: tempBody,
    };
}

function stackGen() {
    let fullContext, output, kbs;
    if (!fs.existsSync("data/policies")) {
        fs.mkdirSync("data/policies", {recursive: true});
    }
    const contextSize = 20;
    fullContext = getContext(contextSize);
    kbs = {};
    for (let avgBatchSize=2; avgBatchSize<Math.min(contextSize, 15); avgBatchSize++) {
        for (let targetFlips=1; targetFlips<14; targetFlips++) {
            for (let nStacks=1; nStacks<targetFlips / 2; nStacks++) {
                output = stackKbGenerator(fullContext, avgBatchSize, targetFlips, nStacks);
                kbs["kb_" + [contextSize, avgBatchSize, targetFlips, nStacks].join("_")] = output;
            }
        }
    }
    process.stdout.write("Complete! | Results @ data/policies/genKbs.json\n");
    fs.writeFileSync("data/policies/genKbs.json", JSON.stringify(kbs, null, 2));
}

function extractKbs() {
    const index = JSON.parse(fs.readFileSync("kbs/final/index.json", "utf8"));
    const kbs = [];
    for (const value of Object.values(index)) {
        for (const kb of value["highs"]) {
            if (!kbs.includes(kb)) {
                kbs.push(kb);
            }
        }
        for (const kb of value["lows"]) {
            if (!kbs.includes(kb)) {
                kbs.push(kb);
            }
        }
    }
    fs.writeFileSync("kbs/final/kbs.json", JSON.stringify(kbs, null, 2));
    // console.log(kbs.length);
}

function generateTrainTest(kbIdsPath = "used-data/kbs.json", kbsPath = "used-data/policies.json") {
    const kbIds = JSON.parse(fs.readFileSync(kbIdsPath, "utf8"));
    const kbs = JSON.parse(fs.readFileSync(kbsPath, "utf8"));
    const trainTest = {}, size = 1000;
    let kbObject, kb, fullContext, tempContext, label, fullSet, split, labels, kbId;
    for (let i = 0; i < kbIds.length; i++) {
        kbId = kbIds[i];
        process.stdout.write("kbId: " + kbId + " | " + (i + 1) + " / " + kbIds.length + "\r");
        // console.log("kbId:", kbId);
        kbObject = kbs[kbId];
        // console.log("kbObject:", kbObject);
        kb = parsers.parseKnowledgeBase("@KnowledgeBase\n" + kbObject["kb"]);
        fullContext = kbObject["fullContext"];
        fullSet = [];
        labels = 0;
        for (let i=0; i<size; i++) {
            tempContext = randomStringReassignment(fullContext);
            outputObject = prudens.deduce(kb, parsers.parseContext(tempContext.join(";") + ";")["context"]);
            label = null;
            if (Object.keys(outputObject["graph"]).includes("h")) {
                label = "h";
                labels++;
            } else if (Object.keys(outputObject["graph"]).includes("-h")) {
                label = "-h";
                labels++;
            }
            fullSet.push({
                context: tempContext,
                label: label,
            });
        }
        split = randomSplit(fullSet, 0.7);
        trainTest[kbId] = {
            train: split["p"],
            test: split["q"],
            labels: labels,
        };
    }
    if (!fs.existsSync("data/datasets")) {
        fs.mkdirSync("data/datasets", {recursive: true});
    }
    fs.writeFileSync("data/datasets/trainTest.json", JSON.stringify(trainTest, null, 2));
    process.stdout.write("Completed! | Results @ data/datasets/trainTest.json\n");
}

function fullCoachingSets(fullContext, n) {
    const coachingSets = [];
    for (let i=0; i<n; i++) {
        coachingSets.push(randomStringReassignment(fullContext));
    }
    return coachingSets;
}

function partialCoachingSets(fullContext, n) {
    const coachingSets = [];
    const l = fullContext.length;
    for (let i=0; i<n; i++) {
        coachingSets.push(randomSubcontext(fullContext, Math.ceil(l * Math.random())));
    }
    return coachingSets;
}

function generateCoaching(n = 500, kbIdsPath = "used-data/kbs.json", kbsPath = "used-data/policies.json") {
    const kbIds = JSON.parse(fs.readFileSync(kbIdsPath, "utf8"));
    const kbs = JSON.parse(fs.readFileSync(kbsPath, "utf8"));
    const coaching = {};
    let fullContext, kbId;
    for (let i = 0; i < kbIds.length; i++) {
        kbId = kbIds[i];
        process.stdout.write("kbId: " + kbId + " | " + (i + 1) + " / " + kbIds.length + "\r");
        fullContext = kbs[kbId]["fullContext"];
        coaching[kbId] = {
            full: fullCoachingSets(fullContext, n),
            partial: partialCoachingSets(fullContext, n),
        };
    }
    if (!fs.existsSync("data/datasets")) {
        fs.mkdirSync("data/datasets", {recursive: true});
    }
    fs.writeFileSync("data/datasets/coaching.json", JSON.stringify(coaching, null, 2));
    process.stdout.write("Completed! | Results @ data/datasets/coaching.json\n");
}

function trainTrees(kbIdsPath = "used-data/kbs.json", kbsPath = "used-data/policies.json", trainTestPath = "used-data/trainTest.json") {
    const kbIds = JSON.parse(fs.readFileSync(kbIdsPath, "utf8"));
    const kbs = JSON.parse(fs.readFileSync(kbsPath, "utf8"));
    const trainTest = JSON.parse(fs.readFileSync(trainTestPath, "utf8"));
    const trainedTrees = {};
    let data, train, test, dt, fullContext, accuracy, jsonTree, metrics, start, end, kbId;
    for (let i = 0; i < kbIds.length; i++) {
        kbId = kbIds[i];
        process.stdout.write("kbId: " + kbId + " | " + (i + 1) + " / " + kbIds.length + "\r");
        fullContext = kbs[kbId]["fullContext"];
        data = trainTest[kbId];
        train = parseData(data["train"]);
        test = parseData(data["test"]);
        start = performance.now();
        dt = new dTree(train, "h", fullContext);
        end = performance.now();
        accuracy = dt.evaluate(test);
        jsonTree = dt.toJSON();
        metrics = treeMetrics(jsonTree["model"]);
        trainedTrees[kbId] = {
            tree: jsonTree,
            depth: metrics["depth"],
            size: metrics["size"],
            paths: metrics["paths"],
            trainTime: end - start,
        };
    }
    if (~fs.existsSync("data/coaches")) {
        fs.mkdirSync("data/coaches", {recursive: true});
    }
    fs.writeFileSync("data/coaches/trainedTrees.json", JSON.stringify(trainedTrees, null, 2));
    process.stdout.write("Completed! | Results @ data/coaches/trainedTrees.json\n");
}

function treeMetrics(model) {
    let maxDepth = 0, size = 0, paths = 0;
    let runningDepth = [0];
    let currentNode, currentDepth;
    const front = [model];
    while (front.length) {
        currentNode = front.pop();
        size++;
        currentDepth = runningDepth.pop();
        if (currentNode["type"] === "result") {
            paths++;
            if (currentDepth > maxDepth) {
                maxDepth = currentDepth;
            }
            continue;
        }
        for (const val of currentNode["vals"]) {
            front.push(val["child"]);
            runningDepth.push(currentDepth + 1);
        }
    }
    return {
        depth: maxDepth,
        size: size,
        paths: paths,
    };
}

function parseData(data) {
    const parsedData = [];
    let newItem;
    for (const item of data) {
        newItem = {};
        if (item["label"] === null) {
            continue;
        }
        for (const literal of item["context"]) {
            if (literal.charAt(0) === "-") {
                newItem[literal.substring(1)] = 0;
            } else {
                newItem[literal] = 1;
            }
        }
        if (item["label"] === "h") {
            newItem["h"] = 1;
        } else {
            newItem["h"] = 0;
        }
        parsedData.push(newItem);
    }
    return parsedData;
}

function trainForests(kbIdsPath = "used-data/kbs.json", kbsPath = "used-data/policies.json", trainTestPath = "used-data/trainTest.json") {
    const kbIds = JSON.parse(fs.readFileSync(kbIdsPath, "utf8"));
    const kbs = JSON.parse(fs.readFileSync(kbsPath, "utf8"));
    const trainTest = JSON.parse(fs.readFileSync(trainTestPath, "utf8"));
    const trainedForests = {};
    const nTrees = 20; lambda = 0.20;
    let data, train, test, forest, fullContext, start, end, kbId;
    for (let i = 0; i < kbIds.length; i++) {
        kbId = kbIds[i];
        process.stdout.write("kbId: " + kbId + " | " + (i + 1) + " / " + kbIds.length + "\r");
        // console.log("kbId:", kbId);
        fullContext = kbs[kbId]["fullContext"];
        data = trainTest[kbId];
        // console.log(data);
        train = parseData(data["train"]);
        test = parseData(data["test"]);
        start = performance.now();
        forest = trainForest(train, test, fullContext, nTrees, lambda);
        end = performance.now();
        trainedForests[kbId] = forest;
        // console.log("Training Time:", end - start);
    }
    if (~fs.existsSync("data/coaches")) {
        fs.mkdirSync("data/coaches", {recursive: true});
    }
    fs.writeFileSync("data/coaches/trainedForests.json", JSON.stringify(trainedForests, null, 2));
    process.stdout.write("Completed! | Results @ data/coaches/trainedForests.json\n");
}

function trainForest(train, test, fullContext, nTrees, lambda) {
    const depths = [], sizes = [], paths = [], trainTimes = [], trees = [], accuracies = [];
    let trainSubset, end, start, accuracy, jsonTree, metrics, dt;
    for (let i=0; i<nTrees; i++) {
        trainSubset = randomSubset(train, lambda * train.length);
        start = performance.now();
        dt = new dTree(trainSubset, "h", fullContext);
        end = performance.now();
        accuracy = dt.evaluate(test);
        jsonTree = dt.toJSON();
        metrics = treeMetrics(jsonTree["model"]);
        trees.push(jsonTree);
        accuracies.push(accuracy);
        depths.push(metrics["depth"]);
        sizes.push(metrics["size"]);
        paths.push(metrics["paths"]);
        trainTimes.push(end - start);
    }
    return {
        accuracies: accuracies,
        depths: depths,
        sizes: sizes,
        paths: paths,
        trainTimes: trainTimes,
        trees: trees,
    };
}

module.exports = {
    stackGen,
    generateTrainTest,
    trainTrees,
    trainForests,
    generateCoaching,
};