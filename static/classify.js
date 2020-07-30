let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var class1=0, class2=0, class3=0, class4=0, class5=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);
    

  model = tf.sequential({
    layers: [
        tf.layers.flatten({inputShape:
                          mobilenet.outputs[0].shape.slice(1)}),
        tf.layers.dense({ units: 100, activation: 'relu'}),
        tf.layers.dense({ units: 5, activation: 'softmax'})
    ]
  });
    
   
  const optimizer = tf.train.adam(0.0001);
    

  model.compile({optimizer: optimizer, loss:'categoricalCrossentropy'});
 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
			class1++;
			document.getElementById("class1").innerText = "Class 1 Samples: " + class1;
			break;
		case "1":
			class2++;
			document.getElementById("class2").innerText = "Class 2 Samples: " + class2;
			break;
		case "2":
			class3++;
			document.getElementById("class3").innerText = "Class 3 Samples: " + class3;
			break;  
		case "3":
			class4++;
			document.getElementById("class4").innerText = "Class 4 Samples: " + class4;
			break;
        case "4":
            class5++;
            document.getElementById("class5").innerText = "Class 5 Samples: " + class5;
		
            
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "Prediction: Class 1";
			break;
		case 1:
			predictionText = "Prediction: Class 2";
			break;
		case 2:
			predictionText = "Prediction: Class 3";
			break;
		case 3:
			predictionText = "Prediction: Class 4";
			break;
        case 4:
            predictionText = "Prediction: Class 5";
            break;
	
            
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
	alert("Training Done!")
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}


async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}


init();