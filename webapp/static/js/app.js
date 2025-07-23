// DOM Elements
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const cameraBtn = document.getElementById('cameraBtn');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const cancelBtn = document.getElementById('cancelBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const resultsLoader = document.getElementById('resultsLoader');
const resultsData = document.getElementById('resultsData');
const errorMessage = document.getElementById('errorMessage');
const primaryDiseaseName = document.getElementById('primaryDiseaseName');
const primaryConfidence = document.getElementById('primaryConfidence');
const primaryDescription = document.getElementById('primaryDescription');
const treatmentInfo = document.getElementById('treatmentInfo');
const preventionInfo = document.getElementById('preventionInfo');
const otherPossibilities = document.getElementById('otherPossibilities');
const inferenceTime = document.getElementById('inferenceTime');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const errorRetryBtn = document.getElementById('errorRetryBtn');
const offlineStatus = document.getElementById('offlineStatus');
const offlineNotification = document.getElementById('offlineNotification');
const installBtn = document.getElementById('installBtn');

// Camera Modal Elements
const cameraModal = document.getElementById('cameraModal');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const cameraFeed = document.getElementById('cameraFeed');
const cameraCanvas = document.getElementById('cameraCanvas');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const captureBtn = document.getElementById('captureBtn');

// Global variables
let selectedFile = null;
let deferredPrompt = null;
let stream = null;
let facingMode = 'environment'; // 'environment' for back camera, 'user' for front camera

// Check if the browser is online
function updateOnlineStatus() {
    const isOnline = navigator.onLine;
    const statusIndicator = offlineStatus.querySelector('.status-indicator');
    const statusText = offlineStatus.querySelector('.status-text');
    
    if (isOnline) {
        statusIndicator.classList.remove('offline');
        statusIndicator.classList.add('online');
        statusText.textContent = 'You are currently online';
        offlineNotification.hidden = true;
    } else {
        statusIndicator.classList.remove('online');
        statusIndicator.classList.add('offline');
        statusText.textContent = 'You are currently offline';
        offlineNotification.hidden = false;
    }
}

// Event listeners for online/offline status
window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);

// Initialize online status
updateOnlineStatus();

// File Upload Handlers
function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    
    // Check if the file is an image
    if (!file.type.match('image.*')) {
        alert('Please select an image file (JPEG, PNG).');
        return;
    }
    
    selectedFile = file;
    
    // Display image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewContainer.hidden = false;
        dropArea.querySelector('.upload-instructions').hidden = true;
        dropArea.querySelector('.camera-option').hidden = true;
    };
    reader.readAsDataURL(file);
}

// Drag and drop handlers
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

// Browse button handler
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
});

// Camera button handler
cameraBtn.addEventListener('click', () => {
    openCamera();
});

// Cancel button handler
cancelBtn.addEventListener('click', () => {
    resetUploadArea();
});

// Analyze button handler
analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    
    // Show results section and loader
    resultsSection.hidden = false;
    resultsLoader.hidden = false;
    resultsData.hidden = true;
    errorMessage.hidden = true;
    resultImage.src = imagePreview.src;
    
    // Scroll to results section
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Upload the image for analysis
    uploadImage(selectedFile);
});

// New analysis button handler
newAnalysisBtn.addEventListener('click', () => {
    resetUploadArea();
    resultsSection.hidden = true;
});

// Error retry button handler
errorRetryBtn.addEventListener('click', () => {
    resetUploadArea();
    resultsSection.hidden = true;
});

// Reset upload area
function resetUploadArea() {
    selectedFile = null;
    previewContainer.hidden = true;
    dropArea.querySelector('.upload-instructions').hidden = false;
    dropArea.querySelector('.camera-option').hidden = false;
    fileInput.value = '';
}

// Upload image for analysis
function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'An error occurred during analysis.');
            });
        }
        return response.json();
    })
    .then(data => {
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        resultsLoader.hidden = true;
        errorMessage.hidden = false;
        errorMessage.querySelector('p').textContent = error.message;
    });
}

// Display analysis results
function displayResults(data) {
    // Hide loader
    resultsLoader.hidden = true;
    
    // Enhanced error handling for missing or empty data
    if (!data) {
        errorMessage.hidden = false;
        errorMessage.querySelector('p').textContent = 'No response data received. Please try again.';
        return;
    }
    
    // Check if predictions exist and are valid
    if (!data.predictions || !Array.isArray(data.predictions) || data.predictions.length === 0) {
        errorMessage.hidden = false;
        errorMessage.querySelector('p').textContent = data.error || 'No predictions found. Please try again with a different image.';
        return;
    }
    
    // Show results data
    resultsData.hidden = false;
    errorMessage.hidden = true;
    
    // Get the primary prediction (first one) with fallbacks for all properties
    const primaryPrediction = data.predictions[0] || {};
    
    // Display primary prediction with robust fallbacks
    primaryDiseaseName.textContent = primaryPrediction.display_name || 'Unknown Disease';
    primaryConfidence.textContent = `Confidence: ${primaryPrediction.confidence_percent || '0%'}`;
    primaryDescription.textContent = primaryPrediction.description || 'No description available for this disease.';
    
    // Display treatment and prevention with fallbacks
    treatmentInfo.textContent = primaryPrediction.treatment || 'No specific treatment information available. Please consult with an agricultural expert.';
    preventionInfo.textContent = primaryPrediction.prevention || 'No specific prevention information available. Consider general best practices for plant health.';
    
    // Display other possibilities with robust error handling
    otherPossibilities.innerHTML = '';
    
    if (data.predictions.length > 1) {
        for (let i = 1; i < data.predictions.length; i++) {
            const prediction = data.predictions[i] || {};
            const li = document.createElement('li');
            li.innerHTML = `<strong>${prediction.display_name || 'Unknown'}</strong> (${prediction.confidence_percent || '0%'})`;
            otherPossibilities.appendChild(li);
        }
    } else {
        const li = document.createElement('li');
        li.textContent = 'No alternative possibilities detected';
        otherPossibilities.appendChild(li);
    }
    
    // Display inference time with fallback
    inferenceTime.textContent = data.inference_time || 'N/A';
}

// Camera functionality
function openCamera() {
    // Check if the browser supports getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support camera access. Please use a modern browser or upload an image instead.');
        return;
    }
    
    // Show the camera modal
    cameraModal.classList.add('show');
    
    // Get camera access
    startCamera();
}

// Start camera
async function startCamera() {
    try {
        // Stop any existing stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        // Get camera stream
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facingMode }
        });
        
        // Display the stream
        cameraFeed.srcObject = stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access the camera. Please check your camera permissions or use the upload option instead.');
        closeCameraModal();
    }
}

// Switch camera
switchCameraBtn.addEventListener('click', () => {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    startCamera();
});

// Capture photo
captureBtn.addEventListener('click', () => {
    if (!stream) return;
    
    // Set canvas dimensions to match video
    cameraCanvas.width = cameraFeed.videoWidth;
    cameraCanvas.height = cameraFeed.videoHeight;
    
    // Draw the current video frame to the canvas
    const context = cameraCanvas.getContext('2d');
    context.drawImage(cameraFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);
    
    // Convert canvas to blob
    cameraCanvas.toBlob(blob => {
        // Create a File object from the blob
        selectedFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        
        // Display the captured image
        imagePreview.src = cameraCanvas.toDataURL('image/jpeg');
        previewContainer.hidden = false;
        dropArea.querySelector('.upload-instructions').hidden = true;
        dropArea.querySelector('.camera-option').hidden = true;
        
        // Close the camera modal
        closeCameraModal();
    }, 'image/jpeg', 0.9);
});

// Close camera modal
closeCameraBtn.addEventListener('click', closeCameraModal);

function closeCameraModal() {
    cameraModal.classList.remove('show');
    
    // Stop the camera stream
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// PWA Installation
window.addEventListener('beforeinstallprompt', (e) => {
    // Prevent the mini-infobar from appearing on mobile
    e.preventDefault();
    // Stash the event so it can be triggered later
    deferredPrompt = e;
    // Update UI to notify the user they can install the PWA
    installBtn.hidden = false;
});

installBtn.addEventListener('click', async () => {
    if (!deferredPrompt) return;
    
    // Show the install prompt
    deferredPrompt.prompt();
    
    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice;
    
    // We no longer need the prompt
    deferredPrompt = null;
    
    // Hide the install button
    installBtn.hidden = true;
});

// When the app is installed, hide the install button
window.addEventListener('appinstalled', () => {
    installBtn.hidden = true;
    deferredPrompt = null;
});