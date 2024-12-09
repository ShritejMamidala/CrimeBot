const video = document.getElementById('webcam');
const startButton = document.getElementById('startCapture');
const stopButton = document.getElementById('stopCapture');
const themeToggle = document.getElementById('themeToggle');
const personSummary = document.getElementById('personSummary');
const canvasOverlay = document.createElement('canvas');
let websocket = null;
let captureInterval = null;

canvasOverlay.style.position = 'absolute';
canvasOverlay.style.pointerEvents = 'none'; // non-interactive
document.body.appendChild(canvasOverlay);

// align
function alignCanvasWithVideo() {
  const rect = video.getBoundingClientRect();
  canvasOverlay.style.width = `${rect.width}px`;
  canvasOverlay.style.height = `${rect.height}px`;
  canvasOverlay.style.top = `${rect.top}px`;
  canvasOverlay.style.left = `${rect.left}px`;
  canvasOverlay.width = rect.width;
  canvasOverlay.height = rect.height;
}
window.addEventListener('resize', alignCanvasWithVideo);

// Dark/Light mode
themeToggle.addEventListener('click', () => {
  const theme = document.body.getAttribute('data-theme');
  document.body.setAttribute('data-theme', theme === 'dark' ? 'light' : 'dark');
});

//  Webcam Stream
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      alignCanvasWithVideo(); // align canvas once  metadata loads
    };
  } catch (err) {
    console.error('Error accessing webcam:', err);
    alert('Could not access the webcam.');
  }
}

// get frames Frames and send them to WebSocket
function startFrameCapture() {
  const captureCanvas = document.createElement('canvas');
  const captureContext = captureCanvas.getContext('2d');
  captureCanvas.width = video.videoWidth || 640;
  captureCanvas.height = video.videoHeight || 480;

  captureInterval = setInterval(() => {
    captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCanvas.toBlob((blob) => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        blob.arrayBuffer().then((buffer) => {
          websocket.send(buffer); // binary frame
        });
      }
    }, 'image/jpeg', 0.8);
  }, 100); // 10 FPS
}

//  WebSocket
function connectWebSocket() {
  websocket = new WebSocket("ws://localhost:8000/process");
  websocket.onopen = () => {
    console.log('WebSocket connected');
    startFrameCapture();
  };
  websocket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "detection_results") {
      console.log("Detections received:", data.data);

      // Clear overlay from the canvas
      const ctx = canvasOverlay.getContext('2d');
      ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

      //  scaling 
      const scaleX = canvasOverlay.width / video.videoWidth;
      const scaleY = canvasOverlay.height / video.videoHeight;

      //  bbox on the canvas
      data.data.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const confidence = (detection.confidence * 100).toFixed(2);
        const id = detection.id;

        // scale bbox coords to match canvas
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        const scaledX2 = x2 * scaleX;
        const scaledY2 = y2 * scaleY;

        //  bbox
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

        //  confidence and ID data
        ctx.fillStyle = 'red';
        ctx.font = '16px Arial';
        ctx.fillText(`ID: ${id}`, scaledX1, scaledY1 - 20);
        ctx.fillText(`${confidence}%`, scaledX1, scaledY1 - 5);
      });
    }

    else if (data.type === "gpt_response") {
      console.log("GPT Response received:", data);
      const { id, content } = data;

      // check if gpt alr ID already exists
      let summaryElement = document.getElementById(`summary-id-${id}`);
      if (!summaryElement) {
        // create a  summary for the ID
        summaryElement = document.createElement('div');
        summaryElement.id = `summary-id-${id}`;
        summaryElement.innerHTML = `<p><strong>Suspect ID ${id}:</strong> ${content}</p>`;
        personSummary.appendChild(summaryElement);
      }
    }
  };
  websocket.onclose = () => console.log('WebSocket disconnected');
  websocket.onerror = (err) => console.error('WebSocket error:', err);
}

// Frame Capture
function stopFrameCapture() {
  clearInterval(captureInterval);
  const ctx = canvasOverlay.getContext('2d');
  ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height); // clear  canvas
  if (websocket) {
    websocket.close();
    websocket = null;
  }
}

// button listeners
startButton.addEventListener('click', () => {
  connectWebSocket();
  startButton.disabled = true;
  stopButton.disabled = false;
});

stopButton.addEventListener('click', () => {
  stopFrameCapture();
  startButton.disabled = false;
  stopButton.disabled = true;
});

//  Webcam on Page Load so immediatly
startWebcam();

//  event listener for toggle the textbox on clicking the logo
function toggleTextbox() {
  const textboxContainer = document.getElementById('textboxContainer');
  textboxContainer.style.display = textboxContainer.style.display === 'block' ? 'none' : 'block';
}

//  show the textbox
document.querySelector('.robot-logo').addEventListener('click', toggleTextbox);


document.getElementById('saveAboutMe').addEventListener('click', () => {
  const aboutMeText = document.getElementById('aboutMeTextbox').value;
  console.log('About Me:', aboutMeText); 
  alert('Your information has been saved!');
  document.getElementById('textboxContainer').style.display = 'none';
});