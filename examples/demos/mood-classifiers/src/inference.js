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
    'voice_instrumental': ["instrumental", "voice"],
    // 'msd-musicnn-1' : ["rock", "pop", "alternative", "indie", "electronic", "female vocalists", "dance", "00s", "alternative rock", "jazz", "beautiful", "metal", "chillout", "male vocalists", "classic rock", "soul", "indie rock", "Mellow", "electronica", "80s", "folk", "90s", "chill", "instrumental", "punk", "oldies", "blues", "hard rock", "ambient", "acoustic", "experimental", "female vocalist", "guitar", "Hip-Hop", "70s", "party", "country", "easy listening", "sexy", "catchy", "funk", "electro", "heavy metal", "Progressive rock", "60s", "rnb", "indie pop", "sad", "House", "happy"]
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

// function msdMusicnnValuesAverage (arrayOfArrays) {
//     let values0 = [];
//     let values1 = [];
//     let values2 = [];
//     let values3 = [];
//     let values4 = [];
//     let values5 = [];
//     let values6 = [];
//     let values7 = [];
//     let values8 = [];
//     let values9 = [];
//     let values10 = [];
//     let values11 = [];
//     let values12 = [];
//     let values13 = [];
//     let values14 = [];
//     let values15 = [];
//     let values16 = [];
//     let values17 = [];
//     let values18 = [];
//     let values19 = [];
//     let values20 = [];
//     let values21 = [];
//     let values22 = [];
//     let values23 = [];
//     let values24 = [];
//     let values25 = [];
//     let values26 = [];
//     let values27 = [];
//     let values28 = [];
//     let values29 = [];
//     let values30 = [];
//     let values31 = [];
//     let values32 = [];
//     let values33 = [];
//     let values34 = [];
//     let values35 = [];
//     let values36 = [];
//     let values37 = [];
//     let values38 = [];
//     let values39 = [];
//     let values40 = [];
//     let values41 = [];
//     let values42 = [];
//     let values43 = [];
//     let values44 = [];
//     let values45 = [];
//     let values46 = [];
//     let values47 = [];
//     let values48 = [];
//     let values49 = [];

//     arrayOfArrays.forEach((v) => {
//         values0.push(v[0]);
//         values1.push(v[1]);
//         values2.push(v[2]);
//         values3.push(v[3]);
//         values4.push(v[4]);
//         values5.push(v[5]);
//         values6.push(v[6]);
//         values7.push(v[7]);
//         values8.push(v[8]);
//         values9.push(v[9]);
//         values10.push(v[10]);
//         values11.push(v[11]);
//         values12.push(v[12]);
//         values13.push(v[13]);
//         values14.push(v[14]);
//         values15.push(v[15]);
//         values16.push(v[16]);
//         values17.push(v[17]);
//         values18.push(v[18]);
//         values19.push(v[19]);
//         values20.push(v[20]);
//         values21.push(v[21]);
//         values22.push(v[22]);
//         values23.push(v[23]);
//         values24.push(v[24]);
//         values25.push(v[25]);
//         values26.push(v[26]);
//         values27.push(v[27]);
//         values28.push(v[28]);
//         values29.push(v[29]);
//         values30.push(v[30]);
//         values31.push(v[31]);
//         values32.push(v[32]);
//         values33.push(v[33]);
//         values34.push(v[34]);
//         values35.push(v[35]);
//         values36.push(v[36]);
//         values37.push(v[37]);
//         values38.push(v[38]);
//         values39.push(v[39]);
//         values40.push(v[40]);
//         values41.push(v[41]);
//         values42.push(v[42]);
//         values43.push(v[43]);
//         values44.push(v[44]);
//         values45.push(v[45]);
//         values46.push(v[46]);
//         values47.push(v[47]);
//         values48.push(v[48]);
//         values49.push(v[49]);
//     });

//     const values0Avg = values0.reduce((acc, val) => acc + val) / values0.length;
//     const values1Avg = values1.reduce((acc, val) => acc + val) / values1.length;
//     const values2Avg = values2.reduce((acc, val) => acc + val) / values2.length;
//     const values3Avg = values3.reduce((acc, val) => acc + val) / values3.length;
//     const values4Avg = values4.reduce((acc, val) => acc + val) / values4.length;
//     const values5Avg = values5.reduce((acc, val) => acc + val) / values5.length;
//     const values6Avg = values6.reduce((acc, val) => acc + val) / values6.length;
//     const values7Avg = values7.reduce((acc, val) => acc + val) / values7.length;
//     const values8Avg = values8.reduce((acc, val) => acc + val) / values8.length;
//     const values9Avg = values9.reduce((acc, val) => acc + val) / values9.length;
//     const values10Avg = values10.reduce((acc, val) => acc + val) / values10.length;
//     const values11Avg = values11.reduce((acc, val) => acc + val) / values11.length;
//     const values12Avg = values12.reduce((acc, val) => acc + val) / values12.length;
//     const values13Avg = values13.reduce((acc, val) => acc + val) / values13.length;
//     const values14Avg = values14.reduce((acc, val) => acc + val) / values14.length;
//     const values15Avg = values15.reduce((acc, val) => acc + val) / values15.length;
//     const values16Avg = values16.reduce((acc, val) => acc + val) / values16.length;
//     const values17Avg = values17.reduce((acc, val) => acc + val) / values17.length;
//     const values18Avg = values18.reduce((acc, val) => acc + val) / values18.length;
//     const values19Avg = values19.reduce((acc, val) => acc + val) / values19.length;
//     const values20Avg = values20.reduce((acc, val) => acc + val) / values20.length;
//     const values21Avg = values21.reduce((acc, val) => acc + val) / values21.length;
//     const values22Avg = values22.reduce((acc, val) => acc + val) / values22.length;
//     const values23Avg = values23.reduce((acc, val) => acc + val) / values23.length;
//     const values24Avg = values24.reduce((acc, val) => acc + val) / values24.length;
//     const values25Avg = values25.reduce((acc, val) => acc + val) / values25.length;
//     const values26Avg = values26.reduce((acc, val) => acc + val) / values26.length;
//     const values27Avg = values27.reduce((acc, val) => acc + val) / values27.length;
//     const values28Avg = values28.reduce((acc, val) => acc + val) / values28.length;
//     const values29Avg = values29.reduce((acc, val) => acc + val) / values29.length;
//     const values30Avg = values30.reduce((acc, val) => acc + val) / values30.length;
//     const values31Avg = values31.reduce((acc, val) => acc + val) / values31.length;
//     const values32Avg = values32.reduce((acc, val) => acc + val) / values32.length;
//     const values33Avg = values33.reduce((acc, val) => acc + val) / values33.length;
//     const values34Avg = values34.reduce((acc, val) => acc + val) / values34.length;
//     const values35Avg = values35.reduce((acc, val) => acc + val) / values35.length;
//     const values36Avg = values36.reduce((acc, val) => acc + val) / values36.length;
//     const values37Avg = values37.reduce((acc, val) => acc + val) / values37.length;
//     const values38Avg = values38.reduce((acc, val) => acc + val) / values38.length;
//     const values39Avg = values39.reduce((acc, val) => acc + val) / values39.length;
//     const values40Avg = values40.reduce((acc, val) => acc + val) / values40.length;
//     const values41Avg = values41.reduce((acc, val) => acc + val) / values41.length;
//     const values42Avg = values42.reduce((acc, val) => acc + val) / values42.length;
//     const values43Avg = values43.reduce((acc, val) => acc + val) / values43.length;
//     const values44Avg = values44.reduce((acc, val) => acc + val) / values44.length;
//     const values45Avg = values45.reduce((acc, val) => acc + val) / values45.length;
//     const values46Avg = values46.reduce((acc, val) => acc + val) / values46.length;
//     const values47Avg = values47.reduce((acc, val) => acc + val) / values47.length;
//     const values48Avg = values48.reduce((acc, val) => acc + val) / values48.length;
//     const values49Avg = values49.reduce((acc, val) => acc + val) / values49.length;

//     console.info(`* Msd musicnn 1 predictions:
// rock: ${values0Avg}
// pop: ${values1Avg}
// alternative: ${values2Avg}
// indie: ${values3Avg}
// electronic: ${values4Avg}
// female vocalists: ${values5Avg}
// dance: ${values6Avg}
// 00s: ${values7Avg}
// alternative rock: ${values8Avg}
// jazz: ${values9Avg}
// beautiful: ${values10Avg}
// metal: ${values11Avg}
// chillout: ${values12Avg}
// male vocalists: ${values13Avg}
// classic rock: ${values14Avg}
// soul: ${values15Avg}
// indie rock: ${values16Avg}
// Mellow: ${values17Avg}
// electronica: ${values18Avg}
// 80s: ${values19Avg}
// folk: ${values20Avg}
// 90s: ${values21Avg}
// chill: ${values22Avg}
// instrumental: ${values23Avg}
// punk: ${values24Avg}
// oldies: ${values25Avg}
// blues: ${values26Avg}
// hard rock: ${values27Avg}
// ambient: ${values28Avg}
// acoustic: ${values29Avg}
// experimental: ${values30Avg}
// female vocalist: ${values31Avg}
// guitar: ${values32Avg}
// Hip-Hop: ${values33Avg}
// 70s: ${values34Avg}
// party: ${values35Avg}
// country: ${values36Avg}
// easy listening: ${values37Avg}
// sexy: ${values38Avg}
// catchy: ${values39Avg}
// funk: ${values40Avg}
// electro: ${values41Avg}
// heavy metal: ${values42Avg}
// Progressive rock: ${values43Avg}
// 60s: ${values44Avg}
// rnb: ${values45Avg}
// indie pop: ${values46Avg}
// sad: ${values47Avg}
// House: ${values48Avg}
// happy: ${values49Avg}`);

// }


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
        
        // if (modelName === "msd-musicnn-1") {
            
        //     model.predict(features, true).then((predictions) => {
        //         msdMusicnnValuesAverage(predictions);

        //         model.dispose();
        //     });
        // }
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
