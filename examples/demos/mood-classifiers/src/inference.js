importScripts('./lib/tf.min.3.5.0.js');
importScripts('./lib/essentia.js-model.umd.js');

let model;
let modelName = "";
let modelLoaded = false;
let modelReady = false;

const modelTagOrder = {
    'mood_happy': [true, false],
    'mood_sad': [false, true],
    'mood_relaxed': [false, true],
    'mood_aggressive': [true, false],
    'danceability': [true, false],
    'gender': ["female", "male"],
    'genre_dortmund': ["alternative", "blues", "electronic", "folkcountr", "funksoulrnb", "jazz", "pop", "raphiphop", "rock"],
    'genre_rosamerica': ["cla", "dan", "hip", "jaz", "pop", "rhy", "roc", "spe"],
    'genre_tzanetakis': ["blu", "cla", "cou", "dis", "hip", "jaz", "met", "pop", "reg", "roc"],
    'mood_acoustic': ["acoustic", "non_acoustic"],
    'mood_party': ["non_party", "party"],
    'tonal_atonal': ["tonal", "atonal"],
    'voice_instrumental': ["instrumental", "voice"]
};

function initModel() {
    model = new EssentiaModel.TensorflowMusiCNN(tf, getModelURL(modelName));
    
    loadModel(modelName).then((isLoaded) => {
        if (isLoaded) {
            modelLoaded = true;
            // perform dry run to warm them up
            warmUp();
        } 
    });
}

function getModelURL() {
    return `../models/${modelName}-musicnn-msd-2/model.json`;
}

async function loadModel() {
    await model.initialize();
    // warm-up: perform dry run to prepare WebGL shader operations
    console.info(`Model ${modelName} has been loaded!`);
    return true;
}

function warmUp() {
    const fakeFeatures = {
        melSpectrum: getZeroMatrix(187, 96),
        frameSize: 187,
        melBandsSize: 96,
        patchSize: 187
    };

    const fakeStart = Date.now();

    model.predict(fakeFeatures, false).then(() => {
        console.info(`${modelName}: Warm up inference took: ${Date.now() - fakeStart}`);
        modelReady = true;
        if (modelLoaded && modelReady) console.log(`${modelName} loaded and ready.`);
    });
}

async function initTensorflowWASM() {
    if (tf.getBackend() != 'wasm') {
        importScripts('./lib/tf-backend-wasm-3.5.0.js');
        // importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js');
        tf.setBackend('wasm');
        tf.ready().then(() => {
            console.info('tfjs WASM backend successfully initialized!');
            initModel();
        }).catch(() => {
            console.error(`tfjs WASM could NOT be initialized, defaulting to ${tf.getBackend()}`);
            return false;
        });
    }
}


function outputPredictions(p) {
    postMessage({
        predictions: p
    });
}

function twoValuesAverage (arrayOfArrays) {
    let firstValues = [];
    let secondValues = [];

    arrayOfArrays.forEach((v) => {
        firstValues.push(v[0]);
        secondValues.push(v[1]);
    });

    const firstValuesAvg = firstValues.reduce((acc, val) => acc + val) / firstValues.length;
    const secondValuesAvg = secondValues.reduce((acc, val) => acc + val) / secondValues.length;

    return [firstValuesAvg, secondValuesAvg];
}

function genreDortmundValuesAverage (arrayOfArrays) {
    let values0 = [];
    let values1 = [];
    let values2 = [];
    let values3 = [];
    let values4 = [];
    let values5 = [];
    let values6 = [];
    let values7 = [];
    let values8 = [];

    arrayOfArrays.forEach((v) => {
        values0.push(v[0]);
        values1.push(v[1]);
        values2.push(v[2]);
        values3.push(v[3]);
        values4.push(v[4]);
        values5.push(v[5]);
        values6.push(v[6]);
        values7.push(v[7]);
        values8.push(v[8]);
    });

    const values0Avg = values0.reduce((acc, val) => acc + val) / values0.length;
    const values1Avg = values1.reduce((acc, val) => acc + val) / values1.length;
    const values2Avg = values2.reduce((acc, val) => acc + val) / values2.length;
    const values3Avg = values3.reduce((acc, val) => acc + val) / values3.length;
    const values4Avg = values4.reduce((acc, val) => acc + val) / values4.length;
    const values5Avg = values5.reduce((acc, val) => acc + val) / values5.length;
    const values6Avg = values6.reduce((acc, val) => acc + val) / values6.length;
    const values7Avg = values7.reduce((acc, val) => acc + val) / values7.length;
    const values8Avg = values8.reduce((acc, val) => acc + val) / values8.length;

    console.info(`* Genre Dortmund predictions:
alternative: ${values0Avg}
blues: ${values1Avg}
electronic: ${values2Avg}
folkcountr: ${values3Avg}
funksoulrnb: ${values4Avg}
jazz: ${values5Avg}
pop: ${values6Avg}
raphiphop: ${values7Avg}
rock: ${values8Avg}`);

}

function genreRosamericaValuesAverage (arrayOfArrays) {
    let values0 = [];
    let values1 = [];
    let values2 = [];
    let values3 = [];
    let values4 = [];
    let values5 = [];
    let values6 = [];
    let values7 = [];

    arrayOfArrays.forEach((v) => {
        values0.push(v[0]);
        values1.push(v[1]);
        values2.push(v[2]);
        values3.push(v[3]);
        values4.push(v[4]);
        values5.push(v[5]);
        values6.push(v[6]);
        values7.push(v[7]);
    });

    const values0Avg = values0.reduce((acc, val) => acc + val) / values0.length;
    const values1Avg = values1.reduce((acc, val) => acc + val) / values1.length;
    const values2Avg = values2.reduce((acc, val) => acc + val) / values2.length;
    const values3Avg = values3.reduce((acc, val) => acc + val) / values3.length;
    const values4Avg = values4.reduce((acc, val) => acc + val) / values4.length;
    const values5Avg = values5.reduce((acc, val) => acc + val) / values5.length;
    const values6Avg = values6.reduce((acc, val) => acc + val) / values6.length;
    const values7Avg = values7.reduce((acc, val) => acc + val) / values7.length;

    console.info(`* Genre Rosamerica predictions:
classic: ${values0Avg}
dance: ${values1Avg}
hip hop: ${values2Avg}
jazz: ${values3Avg}
pop: ${values4Avg}
rhythm and blues: ${values5Avg}
rock: ${values6Avg}
speech: ${values7Avg}`);

}

function genreTzanetakisValuesAverage (arrayOfArrays) {
    let values0 = [];
    let values1 = [];
    let values2 = [];
    let values3 = [];
    let values4 = [];
    let values5 = [];
    let values6 = [];
    let values7 = [];
    let values8 = [];
    let values9 = [];

    arrayOfArrays.forEach((v) => {
        values0.push(v[0]);
        values1.push(v[1]);
        values2.push(v[2]);
        values3.push(v[3]);
        values4.push(v[4]);
        values5.push(v[5]);
        values6.push(v[6]);
        values7.push(v[7]);
        values8.push(v[8]);
        values9.push(v[9]);
    });

    const values0Avg = values0.reduce((acc, val) => acc + val) / values0.length;
    const values1Avg = values1.reduce((acc, val) => acc + val) / values1.length;
    const values2Avg = values2.reduce((acc, val) => acc + val) / values2.length;
    const values3Avg = values3.reduce((acc, val) => acc + val) / values3.length;
    const values4Avg = values4.reduce((acc, val) => acc + val) / values4.length;
    const values5Avg = values5.reduce((acc, val) => acc + val) / values5.length;
    const values6Avg = values6.reduce((acc, val) => acc + val) / values6.length;
    const values7Avg = values7.reduce((acc, val) => acc + val) / values7.length;
    const values8Avg = values8.reduce((acc, val) => acc + val) / values8.length;
    const values9Avg = values9.reduce((acc, val) => acc + val) / values9.length;

    console.info(`* Genre Tzanetakis predictions:
blues: ${values0Avg}
classic: ${values1Avg}
country: ${values2Avg}
disco: ${values3Avg}
hip hop: ${values4Avg}
jazz: ${values5Avg}
metal: ${values6Avg}
pop: ${values7Avg}
reggae: ${values8Avg}
rock: ${values9Avg}`);

}

function modelPredict(features) {
    if (modelReady) {
        const inferenceStart = Date.now();

        if ( modelName === "mood_happy" || modelName === "mood_sad" || modelName === "mood_relaxed" || modelName === "mood_aggressive" || modelName === "danceability" ) {

            model.predict(features, true).then((predictions) => {
                const summarizedPredictions = twoValuesAverage(predictions);
                // format predictions, grab only positive one
                const results = summarizedPredictions.filter((_, i) => modelTagOrder[modelName][i])[0];

                console.info(`${modelName}: Inference took: ${Date.now() - inferenceStart}`);
                // output to main thread
                outputPredictions(results);
                model.dispose();
            });
        }
        
        if (modelName === "gender") {
            
            model.predict(features, true).then((predictions) => {
                const summarizedPredictions = twoValuesAverage(predictions);
                
                const results = `* ${modelName} predictions: female:${summarizedPredictions[0]}, male:${summarizedPredictions[1]}`;
                console.info(results);

                model.dispose();
            });
        }
        
        if (modelName === "genre_dortmund") {
            
            model.predict(features, true).then((predictions) => {
                genreDortmundValuesAverage(predictions);

                model.dispose();
            });
        }
        
        if (modelName === "genre_rosamerica") {
            
            model.predict(features, true).then((predictions) => {
                genreRosamericaValuesAverage(predictions);

                model.dispose();
            });
        }
        
        if (modelName === "genre_tzanetakis") {
            
            model.predict(features, true).then((predictions) => {
                genreTzanetakisValuesAverage(predictions);

                model.dispose();
            });
        }
        
        if (modelName === "mood_acoustic") {
            
            model.predict(features, true).then((predictions) => {
                const summarizedPredictions = twoValuesAverage(predictions);
                
                const results = `* ${modelName} predictions: acoustic:${summarizedPredictions[0]}, non_acoustic:${summarizedPredictions[1]}`;
                console.info(results);

                model.dispose();
            });
        }
        
        if (modelName === "mood_party") {
            
            model.predict(features, true).then((predictions) => {
                const summarizedPredictions = twoValuesAverage(predictions);
                
                const results = `* ${modelName} predictions: non party:${summarizedPredictions[0]}, party:${summarizedPredictions[1]}`;
                console.info(results);

                model.dispose();
            });
        }        
        
        if (modelName === "tonal_atonal") {
            
            model.predict(features, true).then((predictions) => {
                const summarizedPredictions = twoValuesAverage(predictions);
                
                const results = `* ${modelName} predictions: tonal:${summarizedPredictions[0]}, atonal:${summarizedPredictions[1]}`;
                console.info(results);

                model.dispose();
            });
        }
        
        if (modelName === "voice_instrumental") {
            
            model.predict(features, true).then((predictions) => {
                const summarizedPredictions = twoValuesAverage(predictions);
                
                const results = `* ${modelName} predictions: instrumental:${summarizedPredictions[0]}, voice:${summarizedPredictions[1]}`;
                console.info(results);

                model.dispose();
            });
        }
    }
}

function getZeroMatrix(x, y) {
    let matrix = new Array(x);
    for (let f = 0; f < x; f++) {
        matrix[f] = new Array(y).fill(0);
    }
    return matrix;
}


onmessage = function listenToMainThread(msg) {
    // listen for audio features
    if (msg.data.name) {
        modelName = msg.data.name;
        initTensorflowWASM();
    } else if (msg.data.features) {
        console.log("From inference worker: I've got features!");
        // should/can this eventhandler run async functions
        modelPredict(msg.data.features);
    }
};
