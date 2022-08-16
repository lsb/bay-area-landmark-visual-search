import './App.css';
import React from 'react';
import Webcam from 'react-webcam';

// const ort = require('onnxruntime-web');
/*global ort */

const vocab = ['228 Memorial Park',
'Alcatraz Federal Penitentiary',
'Alcatraz Island',
'Almaden Quicksilver County Park',
'Altamont Pass Wind Farm',
'Alum Rock Park',
'Angel Island',
'Aquarium of the Bay',
'Ardenwood Historic Farm',
'Arizona Cactus Garden',
'Año Nuevo State Park',
'Bayview Park (San Francisco)',
'Beringer Vineyards',
'Berkeley Square',
'Big Basin Redwoods State Park',
'Black Diamond Mines Regional Preserve',
'Broadway',
'Butano State Park',
'CEFCU Stadium',
'California Automobile Museum',
'California Memorial Stadium',
'California State Capitol',
'California State Railroad Museum',
'Carlin Canyon',
'Casa de Estudillo',
'Castello di Amorosa',
'Cathedral Building',
'Central Park',
'Coit Tower',
'Computer History Museum',
'Cow Palace',
'Crissy Field',
'Davis Dam',
'Deep Cove',
'Dragon Gate, San Francisco',
'East Brother Island Light',
'Exploratorium',
'First Unitarian Church of Oakland',
'Fort Mason',
'Fort Ross',
'Ghirardelli Square',
'Glen Canyon Park',
'Golden Gate Bridge',
'Golden Gate National Cemetery',
'Golden Gate Park',
'Golden State Model Railroad Museum',
'Googleplex',
'Grand Lake',
'Half Moon Bay State Beach',
'Happy Valley',
'Henry Cowell Redwoods State Park',
'Kezar Stadium',
'Lake Merced',
'Lake Merritt',
'Lake Washington',
"Levi's Stadium",
'Liberty Island',
'Lloyd Lake',
'Loch Lomond',
'Marin County Civic Center',
'Marin Headlands',
'Milagra Ridge',
'Miller Park',
'Mission San Juan Bautista',
'Monterey Bay',
'Monticello',
'Mori Point',
'Moscone Center',
'Mount Tamalpais',
'Muir Woods National Monument',
'Museum of Art and Digital Entertainment',
'Musée Mécanique',
'Napa Valley Wine Train',
'Natural Bridges State Beach',
'Oak Hill Memorial Park',
'Oakland Aviation Museum',
'Old Sacramento State Historic Park',
'One Montgomery Tower',
'Our Lady of Peace Shrine',
'Pacific Coast Air Museum',
'Pacific Park',
"People's Park",
'Pescadero State Beach',
'Piedmont Park',
'Pier 39 (San Francisco)',
'Pigeon Point Lighthouse',
'Point Reyes Lighthouse',
'Port of Oakland',
'Port of San Francisco',
'Presidio of San Francisco',
'Quarry Lakes Regional Recreation Area',
'Redwood Valley Railway',
'Ruth Bancroft Garden',
'San Bruno Mountain',
'San Francisco Bay',
'San Francisco Botanical Garden',
'San Francisco City Hall',
'San Francisco Museum of Modern Art',
'San Francisco National Cemetery',
'San Francisco Railway Museum',
'San Francisco–Oakland Bay Bridge',
'San Jose Museum of Art',
'San Mateo – Hayward Bridge',
'Santa Cruz Beach Boardwalk',
'Santa Cruz Breakwater Light',
'Santa Cruz Mountains',
'Santana Row',
'Sather Tower',
'Shoreline Park, Mountain View',
'Skyline Park',
'Sonoma Raceway',
'South Beach',
'Sterling Vineyards',
'Sutro Tower',
'The 42',
'Tilden Regional Park',
'Tower 42',
'Tower Bridge',
'Tower Hill',
'Transamerica Pyramid',
'Tribune Tower',
'USS Hornet Museum',
'University of California Botanical Garden',
'University of California Museum of Paleontology',
'University of California, Berkeley',
'V. Sattui Winery',
'Westfield Valley Fair',
'Yerba Buena Island'];

const SQSZ = 256 // this square size is everywhere

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
    };
    this.webcamref = React.createRef();
  }
  componentDidMount() {
    this.handleWebcam();
    this.handleOnnx();
  }
  async handleOnnx() {
    const ortsession = await ort.InferenceSession.create('./landmarks-mnv3.onnx');
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
    const startDate = Date.now();
    const {bchw, ortsession} = this.state;
    unclampAndTranspose201(imagedata, bchw);
    const xposeDate = Date.now();
    const input = new ort.Tensor("float32", bchw, [1, 3, SQSZ, SQSZ]);
    const {output} = await ortsession.run({input});
    const inference_result = output.data;
    const inferDate = Date.now();
    const timing = `${inferDate - startDate}`;
    this.setState({inference_result, timing});
    requestAnimationFrame(() => this.handleWebcam());
  }
  render() {
    const top_k = 5;
    const {inference_result, timing} = this.state;
    const top_k_probs = new Float32Array([...inference_result]).sort().reverse().slice(0, top_k);
    const top_k_ids = top_k_probs.map(prob => inference_result.indexOf(prob));
    return <div className="App">
      <center>
        via your&nbsp;
        <select id="facingMode" onChange={e => this.setState({facingMode: e.target.value})} value={this.state.facingMode}>
          <option value="user">selfie</option>
          <option value="environment">photo</option>
          <option value=""></option>
        </select>&nbsp;webcam
      </center>
      <Webcam
        audio={false}
        height={videoConstraints.height}
        width={videoConstraints.width}
        ref={this.webcamref}
        videoConstraints={{...videoConstraints, facingMode: this.state.facingMode}}
      />
      <div className="millis">
        {timing} ms: {this.state.ortsession === null ? " --- no session :(" : ""}
        {JSON.stringify(top_k_probs.map(p => Math.floor(p * 10000)))}
        {JSON.stringify(top_k_ids)}
      </div>
      {
        [...Array(top_k).keys()].map(
          i => <div>{top_k_probs[i].toFixed(4)} @ {vocab[top_k_ids[i]]}</div>
        )
      }
    </div>
  }
}

export default App;
