import './App.css';
import React from 'react';
import Webcam from 'react-webcam';

// const ort = require('onnxruntime-web');
/*global ort */

const vocab = ["bay-bridge", "castro", "coit-tower", "golden-gate-bridge", "port-of-oakland", "scribd-logo", "sutro-tower", "transamerica-pyramid"];

const SQSZ = 128 // this square size is everywhere

const unclampAndTranspose201 = (bhwc, bchw) => {
  // 8-bit bhwc canvas ImageData -> float32 bchw
  for(let c=0;c<=2;c++) {
    for(let h=0;h<SQSZ;h++) {
      for(let w=0;w<SQSZ;w++) {
        bchw[c*SQSZ*SQSZ + h*SQSZ + w] = bhwc[4 * (h * SQSZ + w) + c] / 255.0
      }
    }
  }
}

const videoConstraints = {
  width: SQSZ,
  height: SQSZ,
}

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inference_result: vocab.map(() => 0),
      timing: "",
      bchw: Float32Array.from({length: SQSZ * SQSZ * 3}),
      facingMode: "environment",
      inferenceToggle: true,
    };
    this.webcamref = React.createRef();
  }
  componentDidMount() {
    this.handleWebcam();
    this.handleOnnx();
  }
  async handleOnnx() {
    const ortsession = await ort.InferenceSession.create('./landmarks-eight-categories.onnx', {executionProviders: ['wasm']});
    this.setState({ ortsession })
  }
  handleWebcam() {
    if(this.webcamref && this.webcamref.current && this.webcamref.current.getCanvas) {
      const canvas = this.webcamref.current.getCanvas();
      if (canvas !== null && this.state.ortsession !== undefined) {
        const ctx = canvas.getContext("2d");
        if (ctx !== null) {
          const imagedata = ctx.getImageData(0, 0, SQSZ, SQSZ).data;
          this.ingestPicture(imagedata);
        } else {
          console.log("no 2d context lol");
          requestAnimationFrame(() => this.handleWebcam());
        }
      } else {
        this.setState({timing: `canvas ${canvas !== null} session ${this.state.ortsession !== undefined}`});
        requestAnimationFrame(() => this.handleWebcam());
      }
    }
  }
  async ingestPicture(imagedata) {
    const {inferenceToggle} = this.state;
    if(inferenceToggle) {
      const startDate = Date.now();
      const {bchw, ortsession} = this.state;
      unclampAndTranspose201(imagedata, bchw);
      const input = new ort.Tensor("float32", bchw, [1, 3, SQSZ, SQSZ]);
      const {output} = await ortsession.run({input});
      const inference_result = output.data;
      const inferDate = Date.now();
      const timing = `${inferDate - startDate}`;
      this.setState({inference_result, timing});
    }
    requestAnimationFrame(() => this.handleWebcam());
  }
  render() {
    const top_k = 5;
    const {inference_result, timing} = this.state;
    const top_k_probs = new Float32Array([...inference_result]).sort().reverse().slice(0, top_k);
    const top_k_ids = top_k_probs.map(prob => inference_result.indexOf(prob));
    return <div className="App">
      <div className="top">
        BRYANT STREET IMAGING
      </div>
      <div className="camerapicker">
        <select id="facingMode" onChange={e => this.setState({facingMode: e.target.value})} value={this.state.facingMode}>
          <option value="user">selfie</option>
          <option value="environment">photo</option>
          <option value=""></option>
        </select>
      </div>
      <Webcam
        audio={false}
        height={videoConstraints.height}
        width={videoConstraints.width}
        ref={this.webcamref}
        videoConstraints={{...videoConstraints, facingMode: this.state.facingMode}}
      />
      <center>
        <div className="inferenceToggle-container" onClick={() => this.setState({inferenceToggle: !this.state.inferenceToggle})}>
          <span className="inferenceToggle sans">
            {(this.state && this.state.inferenceToggle) ? "Browse" : "Keep Looking"}
          </span> 
        </div>
      </center>
      <div className="millis">
        {timing} ms: {this.state.ortsession === undefined ? " --- no session :(" : ""}
        {JSON.stringify(top_k_probs.map(p => Math.floor(p * 10000)))}
        {JSON.stringify(top_k_ids)}
      </div>
      {
        [...Array(top_k).keys()].map(
          i => <div>{top_k_probs[i].toFixed(4)} @ {vocab[top_k_ids[i]]}</div>
        )
      }
      <div className='sans'>
        Here is some sans text as well
      </div>
      <div class="carousel">{Array.from({length: 20}).map((e,i) => 
        <a class="carouselitem" href="https://www.scribd.com/document/38880370" target="_blank">
          <img className='thumb' src="https://imgv2-1-f.scribdassets.com/img/document/38880370/149x198/499a77aa62/0?v=1"/>
          <div className="title sans jumbo">Chinchilla Facts: 10 Facts You Would Never Guess about Chinchillas</div>
          <div className="author sans">{i} {i} Jessica Harrison the Greatest Author she is so cool</div>
        </a>
      )}
      </div>
    </div>
  }
}

export default App;
