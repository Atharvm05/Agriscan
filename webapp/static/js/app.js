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
    const offlineStatusFooter = document.getElementById('offlineStatusFooter');
    
    // Update the main header offline status
    if (offlineStatus) {
        const statusIndicator = offlineStatus.querySelector('.status-indicator');
        const statusText = offlineStatus.querySelector('.status-text');
        
        if (statusIndicator && statusText) {
            if (isOnline) {
                statusIndicator.classList.remove('offline');
                statusIndicator.classList.add('online');
                statusText.textContent = 'You are currently online';
            } else {
                statusIndicator.classList.remove('online');
                statusIndicator.classList.add('offline');
                statusText.textContent = 'You are currently offline';
            }
        } else {
            console.warn('Status indicator or text elements not found in header');
        }
    } else {
        console.warn('Header offline status element not found');
    }
    
    // Update the footer offline status
    if (offlineStatusFooter) {
        const footerStatusIndicator = offlineStatusFooter.querySelector('.status-indicator');
        const footerStatusText = offlineStatusFooter.querySelector('.status-text');
        
        if (footerStatusIndicator && footerStatusText) {
            if (isOnline) {
                footerStatusIndicator.classList.remove('offline');
                footerStatusIndicator.classList.add('online');
                footerStatusText.textContent = 'You are currently online';
            } else {
                footerStatusIndicator.classList.remove('online');
                footerStatusIndicator.classList.add('offline');
                footerStatusText.textContent = 'You are currently offline';
            }
        }
    }
    
    // Update the offline notification banner
    if (offlineNotification) {
        offlineNotification.hidden = isOnline;
    }
}

// Event listeners for online/offline status
// window is always available in browser environment, so no null check needed
window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);

// Initialize online status
// Call updateOnlineStatus only after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    updateOnlineStatus();
});

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
if (dropArea) {
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
}

// Browse button handler
if (browseBtn) {
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });
}

// File input change handler
if (fileInput) {
    fileInput.addEventListener('change', () => {
        handleFiles(fileInput.files);
    });
}

// Camera button handler
if (cameraBtn) {
    cameraBtn.addEventListener('click', () => {
        openCamera();
    });
}

// Cancel button handler
if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
        resetUploadArea();
    });
}

// Analyze button handler
if (analyzeBtn) {
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
}

// New analysis button handler
if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener('click', () => {
        resetUploadArea();
        resultsSection.hidden = true;
    });
}

// Error retry button handler
if (errorRetryBtn) {
    errorRetryBtn.addEventListener('click', () => {
        resetUploadArea();
        resultsSection.hidden = true;
    });
}

// Reset upload area
function resetUploadArea() {
    selectedFile = null;
    if (previewContainer) {
        previewContainer.hidden = true;
    }
    if (dropArea) {
        const uploadInstructions = dropArea.querySelector('.upload-instructions');
        const cameraOption = dropArea.querySelector('.camera-option');
        if (uploadInstructions) {
            uploadInstructions.hidden = false;
        }
        if (cameraOption) {
            cameraOption.hidden = false;
        }
    }
    if (fileInput) {
        fileInput.value = '';
    }
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
    if (resultsLoader) {
        resultsLoader.hidden = true;
    }
    
    // Enhanced error handling for missing or empty data
    if (!data) {
        if (errorMessage) {
            errorMessage.hidden = false;
            const errorParagraph = errorMessage.querySelector('p');
            if (errorParagraph) {
                errorParagraph.textContent = 'No response data received. Please try again.';
            }
        }
        return;
    }
    
    // Check if predictions exist and are valid
    if (!data.predictions || !Array.isArray(data.predictions) || data.predictions.length === 0) {
        if (errorMessage) {
            errorMessage.hidden = false;
            const errorParagraph = errorMessage.querySelector('p');
            if (errorParagraph) {
                errorParagraph.textContent = data.error || 'No predictions found. Please try again with a different image.';
            }
        }
        return;
    }
    
    // Show results data
    if (resultsData) {
        resultsData.hidden = false;
    }
    if (errorMessage) {
        errorMessage.hidden = true;
    }
    
    // Get the primary prediction (first one) with fallbacks for all properties
    const primaryPrediction = data.predictions[0] || {};
    
    // Display primary prediction with robust fallbacks
    if (primaryDiseaseName) {
        primaryDiseaseName.textContent = primaryPrediction.display_name || 'Unknown Disease';
    }
    if (primaryConfidence) {
        primaryConfidence.textContent = `Confidence: ${primaryPrediction.confidence_percent || '0%'}`;
    }
    if (primaryDescription) {
        primaryDescription.textContent = primaryPrediction.description || 'No description available for this disease.';
    }
    
    // Display treatment and prevention with fallbacks
    if (treatmentInfo) {
        treatmentInfo.textContent = primaryPrediction.treatment || 'No specific treatment information available. Please consult with an agricultural expert.';
    }
    if (preventionInfo) {
        preventionInfo.textContent = primaryPrediction.prevention || 'No specific prevention information available. Consider general best practices for plant health.';
    }
    
    // Display other possibilities with robust error handling
    if (otherPossibilities) {
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
    }
    
    // Display inference time with fallback
    if (inferenceTime) {
        inferenceTime.textContent = data.inference_time || 'N/A';
    }
}

// Camera functionality
function openCamera() {
    // Check if the browser supports getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support camera access. Please use a modern browser or upload an image instead.');
        return;
    }
    
    // Show the camera modal
    if (cameraModal) {
        cameraModal.classList.add('show');
    } else {
        console.error('Camera modal element not found');
        return;
    }
    
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
        if (cameraFeed) {
            cameraFeed.srcObject = stream;
        } else {
            console.error('Camera feed element not found');
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            closeCameraModal();
        }
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access the camera. Please check your camera permissions or use the upload option instead.');
        closeCameraModal();
    }
}

// Switch camera
if (switchCameraBtn) {
    switchCameraBtn.addEventListener('click', () => {
        facingMode = facingMode === 'environment' ? 'user' : 'environment';
        startCamera();
    });
}

// Capture photo
if (captureBtn) {
    captureBtn.addEventListener('click', () => {
        if (!stream) return;
        
        // Set canvas dimensions to match video
        if (!cameraCanvas || !cameraFeed) {
            console.error('Camera canvas or feed element not found');
            return;
        }
        
        cameraCanvas.width = cameraFeed.videoWidth;
        cameraCanvas.height = cameraFeed.videoHeight;
        
        // Draw the current video frame to the canvas
        const context = cameraCanvas.getContext('2d');
        if (!context) {
            console.error('Could not get canvas context');
            return;
        }
        
        context.drawImage(cameraFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);
        
        // Convert canvas to blob
        cameraCanvas.toBlob(blob => {
            // Create a File object from the blob
            selectedFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            
            // Display the captured image
            if (imagePreview) {
                imagePreview.src = cameraCanvas.toDataURL('image/jpeg');
            }
            
            if (previewContainer) {
                previewContainer.hidden = false;
            }
            
            if (dropArea) {
                const uploadInstructions = dropArea.querySelector('.upload-instructions');
                const cameraOption = dropArea.querySelector('.camera-option');
                
                if (uploadInstructions) {
                    uploadInstructions.hidden = true;
                }
                
                if (cameraOption) {
                    cameraOption.hidden = true;
                }
            }
            
            // Close the camera modal
            closeCameraModal();
        }, 'image/jpeg', 0.9);
    });
}

// Close camera modal
if (closeCameraBtn) {
    closeCameraBtn.addEventListener('click', closeCameraModal);
}

function closeCameraModal() {
    if (cameraModal) {
        cameraModal.classList.remove('show');
    }
    
    // Stop the camera stream
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// PWA Installation
const installBtnFooter = document.getElementById('installBtnFooter');

window.addEventListener('beforeinstallprompt', (e) => {
    // Prevent the mini-infobar from appearing on mobile
    e.preventDefault();
    // Stash the event so it can be triggered later
    deferredPrompt = e;
    // Update UI to notify the user they can install the PWA
    if (installBtn) {
        installBtn.hidden = false;
    }
    if (installBtnFooter) {
        installBtnFooter.hidden = false;
    }
});

// Add click event listener to the main install button
if (installBtn) {
    installBtn.addEventListener('click', async () => {
        if (!deferredPrompt) return;
        
        // Show the install prompt
        deferredPrompt.prompt();
        
        // Wait for the user to respond to the prompt
        const { outcome } = await deferredPrompt.userChoice;
        
        // We no longer need the prompt
        deferredPrompt = null;
        
        // Hide all install buttons
        if (installBtn) installBtn.hidden = true;
        if (installBtnFooter) installBtnFooter.hidden = true;
    });
}

// Add click event listener to the footer install button
if (installBtnFooter) {
    installBtnFooter.addEventListener('click', async () => {
        if (!deferredPrompt) return;
        
        // Show the install prompt
        deferredPrompt.prompt();
        
        // Wait for the user to respond to the prompt
        const { outcome } = await deferredPrompt.userChoice;
        
        // We no longer need the prompt
        deferredPrompt = null;
        
        // Hide all install buttons
        if (installBtn) installBtn.hidden = true;
        if (installBtnFooter) installBtnFooter.hidden = true;
    });
}

// When the app is installed, hide all install buttons
window.addEventListener('appinstalled', () => {
    if (installBtn) installBtn.hidden = true;
    if (installBtnFooter) installBtnFooter.hidden = true;
    deferredPrompt = null;
});