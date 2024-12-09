const video = document.getElementById('webcam');
const startButton = document.getElementById('startCapture');
const stopButton = document.getElementById('stopCapture');
const themeToggle = document.getElementById('themeToggle');
const personSummary = document.getElementById('personSummary');
const canvasOverlay = document.createElement('canvas');
let websocket = null;
let captureInterval = null;

// Add canvas overlay on top of the video
canvasOverlay.style.position = 'absolute';
canvasOverlay.style.pointerEvents = 'none'; // Make the canvas non-interactive
document.body.appendChild(canvasOverlay);

// Align canvas overlay with the video element
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

// Dark/Light Mode Toggle
themeToggle.addEventListener('click', () => {
  const theme = document.body.getAttribute('data-theme');
  document.body.setAttribute('data-theme', theme === 'dark' ? 'light' : 'dark');
});

// Start Webcam Stream
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      alignCanvasWithVideo(); // Align canvas once video metadata is loaded
    };
  } catch (err) {
    console.error('Error accessing webcam:', err);
    alert('Could not access the webcam.');
  }
}

// Capture Frames and Send to WebSocket
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
          websocket.send(buffer); // Send binary frame
        });
      }
    }, 'image/jpeg', 0.8);
  }, 100); // ~10 FPS
}

// Initialize WebSocket
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

      // Clear the canvas overlay
      const ctx = canvasOverlay.getContext('2d');
      ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

      // Calculate scaling factors
      const scaleX = canvasOverlay.width / video.videoWidth;
      const scaleY = canvasOverlay.height / video.videoHeight;

      // Draw bounding boxes on the overlay canvas
      data.data.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const confidence = (detection.confidence * 100).toFixed(2);
        const id = detection.id;

        // Scale bounding box coordinates to match the canvas dimensions
        const scaledX1 = x1 * scaleX;
        const scaledY1 = y1 * scaleY;
        const scaledX2 = x2 * scaleX;
        const scaledY2 = y2 * scaleY;

        // Draw rectangle
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

        // Add confidence and ID text
        ctx.fillStyle = 'red';
        ctx.font = '16px Arial';
        ctx.fillText(`ID: ${id}`, scaledX1, scaledY1 - 20);
        ctx.fillText(`${confidence}%`, scaledX1, scaledY1 - 5);
      });
    }

    else if (data.type === "gpt_response") {
      console.log("GPT Response received:", data);
      const { id, content } = data;

      // Check if the summary for this ID already exists
      let summaryElement = document.getElementById(`summary-id-${id}`);
      if (!summaryElement) {
        // Create a new summary for the ID if it doesn't exist
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

// Stop Frame Capture
function stopFrameCapture() {
  clearInterval(captureInterval);
  const ctx = canvasOverlay.getContext('2d');
  ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height); // Clear the canvas
  if (websocket) {
    websocket.close();
    websocket = null;
  }
}

// Button Event Listeners
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

// Start Webcam on Page Load
startWebcam();

// Add event listener for robot logo click to toggle the textbox
function toggleTextbox() {
  const textboxContainer = document.getElementById('textboxContainer');
  textboxContainer.style.display = textboxContainer.style.display === 'block' ? 'none' : 'block';
}

// Update the robot logo click event to show the textbox
document.querySelector('.robot-logo').addEventListener('click', toggleTextbox);

// Event listener for save button to store the "About Me" text
document.getElementById('saveAboutMe').addEventListener('click', () => {
  const aboutMeText = document.getElementById('aboutMeTextbox').value;
  console.log('About Me:', aboutMeText); // You can save this to a server or local storage if needed
  alert('Your information has been saved!');
  document.getElementById('textboxContainer').style.display = 'none'; // Hide the textbox after saving
});