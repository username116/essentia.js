import EventBus from "./event-bus";
import JSZip from "jszip";
import audioEncoder from "audio-encoder";
import freesound from 'freesound';
import apiKey from '../.env/key';

/* 
    "Signal Processing Unit" for audio-only purposes (no views)
    Connects to EventBus for responding to global events
*/

export default class DSP {
    constructor () {
        this.audioCtx = new (AudioContext || webkitAudioContext)();
        // create audio worker, set up comms
        this.audioWorker = new Worker("./audio-worker.js", {type: "module"});
        this.audioWorker.postMessage({
            request: 'updateParams',
            params: {sampleRate: this.audioCtx.sampleRate}
        });
        this.audioWorker.onmessage = (msg) => {
            if (msg.data instanceof Float32Array) {
                if (msg.data.length == 0) {
                    EventBus.$emit("analysis-finished-empty");
                } else {
                    EventBus.$emit("analysis-finished-onsets", Array.from(msg.data));
                }
            } else if (msg.data instanceof Array && msg.data[0] instanceof Float32Array) {
                console.info('worker returned sliced audio', msg.data);
                this.downloadSlicesAsZip(msg.data);
            } else {
                throw TypeError("Worker failed. Analysis results should be of type Float32Array");
            }
        };

        this.soundData = {};
        this.fileSampleRate = this.audioCtx.sampleRate;
        
        // set up global event handlers
        EventBus.$on("sound-selected", (url) => this.handleSoundSelect(url) );
        EventBus.$on("sound-read", (sound) => this.decodeSoundBlob(sound) );
        EventBus.$on("algo-params-init", params => {
            this.audioWorker.postMessage({
                request: 'initParams',
                params: params
            })
        });
        EventBus.$on("algo-params-updated", params => {
            this.audioWorker.postMessage({
                request: 'updateParams',
                params: params
            })
        });
        EventBus.$on("download-slices", () => {
            this.audioWorker.postMessage({
                request: 'slice'
            })
        });
    }

    async getAudioFile (url) {
        let resp = await fetch(url);
        return await resp.blob();
    }
    
    async decodeBuffer (arrayBuffer) {
        let audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
        return audioBuffer;
    }
    
    async handleSoundSelect (data) {
        if (data.url === '' && data.id) {
            try {
                let sound = await freesoundGetById(data.id);
                data.name = sound.name;
                data.url = sound.previews["preview-hq-mp3"];
                data.user = sound.username;
                data.fsLink = sound.url;
                data.license = sound.license;
            } catch (err) {
                EventBus.$emit('fs-search-failed', err);
                return;
            }
        }
        let blob = await this.getAudioFile(data.url);
        EventBus.$emit("sound-read", {blob: blob, ...data});
    }
    
    async decodeSoundBlob (sound) {
        this.soundData = sound;
        let buffer = await sound.blob.arrayBuffer();
        let audioBuffer = await this.decodeBuffer(buffer);
        this.fileSampleRate = audioBuffer.sampleRate;
        let audioArray = audioBuffer.getChannelData(0);
        // send audioArray to Worker
        this.audioWorker.postMessage({
            request: 'analyse',
            audio: audioArray
        });
    }

    async encodeSlices (slicesArray) {
        let resampledSlices = [];
    
        for (let i = 0; i < slicesArray.length; i++) {
            let buffer = this.audioCtx.createBuffer(1, slicesArray[i].length, this.fileSampleRate);
            let data = buffer.getChannelData(0);
            for (let j = 0; j < slicesArray[i].length; j++) data[j] = slicesArray[i][j];
            let resampledBuffer = await this.resample(buffer, 44100); // audioEncoder only supports 44100hz sampling rate
            resampledSlices.push(resampledBuffer);
        }

        
        let blobs = resampledSlices.map( (sliceBuffer) => {
            return audioEncoder(sliceBuffer, 'WAV', null, null);
        });

        let encodedSlices = blobs.map( (b, idx) => {
            return {
                name: `${this.soundData.id}${this.soundData.id ? '_' : ''}${this.soundData.name}_${idx}.wav`,
                blob: b
            }
        });

        return encodedSlices;
    }

    async downloadSlicesAsZip (slicesArray) {
        const downloadName = `onset-slices_${this.soundData.id}${this.soundData.id ? '_' : ''}${this.soundData.name}`;
        const slices = await this.encodeSlices(slicesArray);
        let zip = new JSZip();
        if (this.soundData.license) {
            zip.folder(downloadName).file('LICENSE.md', `File name: ${this.soundData.name}\nby Freesound user: ${this.soundData.user}\nLicensed with ${this.soundData.license}`);
        }
        slices.forEach( (s) => {
            zip.folder(downloadName).file(s.name, s.blob, {
                compression: 'STORE' // no compression
            })
        });

        zip.generateAsync({type: 'blob', compression: 'STORE'})
        .then( zipBlob => {
            const zipURL = URL.createObjectURL(zipBlob);
            const link = document.createElement('a');
            link.setAttribute('href', zipURL );
            link.setAttribute('download', downloadName+'.zip');
            console.info('download link', link);
            link.click();
            URL.revokeObjectURL(zipURL);
            EventBus.$emit('slices-downloaded');
        })
    }

    async resample (sourceAudioBuffer, targetSampleRate) {
        let offlineCtx = new OfflineAudioContext(sourceAudioBuffer.numberOfChannels,
            sourceAudioBuffer.duration * targetSampleRate,
            targetSampleRate);

        // Play it from the beginning.
        let offlineSource = offlineCtx.createBufferSource();
        offlineSource.buffer = sourceAudioBuffer;
        offlineSource.connect(offlineCtx.destination);
        offlineSource.start();
        let resampled = await offlineCtx.startRendering();
        return resampled;
    } 
}

async function freesoundGetById (id) {
    return new Promise((res, rej) => {
        freesound.getSound(id, sound => {
            res(sound)
        }, rej)
    })
}
