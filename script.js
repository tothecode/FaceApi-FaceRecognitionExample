
const image = document.getElementById("image");

Promise.all([
    //faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
])
.then(startDetection)

  async function startDetection(){
    const labeledFaceDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);


    const canvas = faceapi.createCanvasFromMedia(image);
    document.body.append(canvas);
  
    const faces = await faceapi.detectAllFaces( image, new faceapi.SsdMobilenetv1Options() ).withFaceLandmarks().withFaceDescriptors();

    const displaySize = {width: image.width, height: image.height};
    const resizedFaces = faceapi.resizeResults(faces, displaySize);

    canvas.getContext('2d').clearRect(0,0, canvas.width, canvas.height);
    //faceapi.draw.drawDetections(canvas, resizedFaces);
    //faceapi.draw.drawFaceExpressions(canvas,resizedFaces);

    const results = resizedFaces.map(d => faceMatcher.findBestMatch(d.descriptor));
    results.forEach((result, i) => {
      const box = resizedFaces[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
      drawBox.draw(canvas);
    })

    document.getElementById("status").innerText = "Done";
  }

  function loadLabeledImages() {
    const labels = ['Wonder Woman', 'Batman', 'Superman', 'Cyborg', 'Flash', 'Aquaman'];

    return Promise.all(
      labels.map(async label => {
        const descriptions = []
        for (let i = 1; i < 2; i++) {
          const img = await faceapi.fetchImage(`img/labeled/${label}/${i}.jpg`);
          const face = await faceapi.detectSingleFace(img)
                                     .withFaceLandmarks()
                                     .withFaceDescriptor();
          descriptions.push(face.descriptor);
        }
  
        return new faceapi.LabeledFaceDescriptors(label, descriptions)
      })
    )
  }


