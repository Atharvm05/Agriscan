// Offline inference using TensorFlow.js for AgriScan

class OfflineInference {
  constructor() {
    this.model = null;
    this.labels = null;
    this.isModelLoaded = false;
    this.modelId = 'agriscan_model_tflite';
    this.labelsId = 'agriscan_labels';
    this.db = agriscanDB; // Reference to the database singleton
    this.diseaseInfo = null; // Will store disease information
  }

  // Initialize the offline inference system
  async init() {
    try {
      console.log('Initializing offline inference...');
      
      // Load disease information
      await this.loadDiseaseInfo();
      
      // Check if model is already in IndexedDB
      const modelExists = await this.db.modelExists(this.modelId);
      
      if (modelExists) {
        console.log('Model found in database, loading...');
        await this.loadModelFromIndexedDB();
      } else {
        console.log('Model not found in database, downloading...');
        await this.downloadAndSaveModel();
      }
      
      // Load labels
      await this.loadLabels();
      
      this.isModelLoaded = true;
      console.log('Offline inference initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize offline inference:', error);
      return false;
    }
  }

  // Load the model from IndexedDB
  async loadModelFromIndexedDB() {
    try {
      const modelData = await this.db.getModel(this.modelId);
      if (!modelData) {
        throw new Error('Model not found in database');
      }
      
      // Load the model using TensorFlow.js
      this.model = await tf.loadGraphModel(
        tf.io.fromMemory(modelData.model)
      );
      
      console.log('Model loaded from IndexedDB successfully');
    } catch (error) {
      console.error('Error loading model from IndexedDB:', error);
      throw error;
    }
  }

  // Download the model from the server and save to IndexedDB
  async downloadAndSaveModel() {
    try {
      // Check if online
      if (!navigator.onLine) {
        throw new Error('Cannot download model while offline');
      }
      
      // Download the model
      console.log('Downloading model from server...');
      this.model = await tf.loadGraphModel('/static/model/model.json');
      
      // Save the model to IndexedDB
      const modelArtifacts = await this.model.save(tf.io.withSaveHandler(async (artifacts) => {
        // Save model to IndexedDB
        await this.db.saveModel(this.modelId, artifacts, {
          format: 'graph-model',
          version: '1.0.0',
          date: new Date().toISOString()
        });
        return { modelArtifactsInfo: { dateSaved: new Date() } };
      }));
      
      console.log('Model downloaded and saved to IndexedDB:', modelArtifacts);
    } catch (error) {
      console.error('Error downloading and saving model:', error);
      throw error;
    }
  }

  // Load class labels
  async loadLabels() {
    try {
      // Check if labels exist in IndexedDB
      let labelsData = await this.db.getModel(this.labelsId);
      
      if (labelsData) {
        try {
          // Load labels from IndexedDB
          this.labels = JSON.parse(new TextDecoder().decode(labelsData.model));
          console.log('Labels loaded from IndexedDB:', this.labels);
        } catch (parseError) {
          console.error('Error parsing labels from IndexedDB:', parseError);
          // If parsing fails, we'll try to download fresh labels
          labelsData = null;
        }
      }
      
      if (!labelsData || !this.labels) {
        // Download labels from server
        if (!navigator.onLine) {
          throw new Error('Cannot download labels while offline');
        }
        
        // Try label_map.json first
        try {
          const response = await fetch('/static/model/label_map.json');
          if (response.ok) {
            this.labels = await response.json();
          } else {
            // Fallback to labels.json
            const fallbackResponse = await fetch('/static/model/labels.json');
            if (!fallbackResponse.ok) {
              throw new Error(`Failed to fetch labels: ${fallbackResponse.status} ${fallbackResponse.statusText}`);
            }
            this.labels = await fallbackResponse.json();
          }
        } catch (fetchError) {
          console.error('Error fetching labels:', fetchError);
          // Create a default empty labels object as fallback
          this.labels = {};
          throw fetchError;
        }
        
        // Save labels to IndexedDB
        try {
          const labelsBuffer = new TextEncoder().encode(JSON.stringify(this.labels));
          await this.db.saveModel(this.labelsId, labelsBuffer, {
            format: 'json',
            version: '1.0.0',
            date: new Date().toISOString()
          });
          
          console.log('Labels downloaded and saved to IndexedDB:', this.labels);
        } catch (saveError) {
          console.error('Error saving labels to IndexedDB:', saveError);
          // Continue even if saving fails
        }
      }
    } catch (error) {
      console.error('Error loading labels:', error);
      throw error;
    }
  }

  // Load disease information
  async loadDiseaseInfo() {
    try {
      // Check if disease info exists in IndexedDB
      const diseaseInfoData = await this.db.getModel('disease_info');
      
      if (diseaseInfoData) {
        // Load disease info from IndexedDB
        this.diseaseInfo = JSON.parse(new TextDecoder().decode(diseaseInfoData.model));
        console.log('Disease info loaded from IndexedDB');
      } else {
        // Download disease info from server
        if (!navigator.onLine) {
          throw new Error('Cannot download disease info while offline');
        }
        
        const response = await fetch('/static/data/disease_info.json');
        if (!response.ok) {
          throw new Error(`Failed to fetch disease info: ${response.status} ${response.statusText}`);
        }
        
        this.diseaseInfo = await response.json();
        
        // Save disease info to IndexedDB
        const diseaseInfoBuffer = new TextEncoder().encode(JSON.stringify(this.diseaseInfo));
        await this.db.saveModel('disease_info', diseaseInfoBuffer, {
          format: 'json',
          version: '1.0.0',
          date: new Date().toISOString()
        });
        
        console.log('Disease info downloaded and saved to IndexedDB');
      }
    } catch (error) {
      console.error('Error loading disease info:', error);
      throw error;
    }
  }

  // Preprocess the image for the model
  preprocessImage(img) {
    return tf.tidy(() => {
      // Convert the image to a tensor
      let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224]) // Resize to model input size
        .toFloat();
      
      // Normalize the image
      tensor = tensor.div(tf.scalar(255));
      
      // Add batch dimension [1, 224, 224, 3]
      tensor = tensor.expandDims(0);
      
      return tensor;
    });
  }

  // Run inference on an image
  async runInference(imageElement) {
    if (!this.isModelLoaded) {
      throw new Error('Model not loaded');
    }
    
    try {
      console.log('Running offline inference...');
      
      // Start timing
      const startTime = performance.now();
      
      // Preprocess the image
      const tensor = this.preprocessImage(imageElement);
      
      // Run the model
      const predictions = await this.model.predict(tensor);
      
      // Get the prediction data
      const data = await predictions.data();
      
      // Cleanup tensors
      tensor.dispose();
      predictions.dispose();
      
      // End timing
      const inferenceTime = performance.now() - startTime;
      
      // Process results
      const results = this.processResults(data, inferenceTime);
      
      // Save results to IndexedDB
      await this.db.saveResults(results);
      
      return results;
    } catch (error) {
      console.error('Error running inference:', error);
      throw error;
    }
  }

  // Process the raw prediction results
  processResults(predictionData, inferenceTime) {
    // Create an array of prediction objects with label and confidence
    const predictionArray = Array.from(predictionData).map((confidence, index) => {
      const label = this.labels[index];
      return {
        label,
        confidence,
        confidence_percent: `${(confidence * 100).toFixed(2)}%`,
        display_name: this.formatDisplayName(label),
        description: this.getDiseaseDescription(label),
        treatment: this.getDiseaseTreatment(label),
        prevention: this.getDiseasePrevention(label)
      };
    });
    
    // Sort by confidence (highest first)
    predictionArray.sort((a, b) => b.confidence - a.confidence);
    
    // Format the inference time
    const formattedInferenceTime = `${inferenceTime.toFixed(2)} ms`;
    
    // Return the formatted results
    return {
      predictions: predictionArray,
      inference_time: formattedInferenceTime,
      timestamp: new Date().toISOString(),
      offline: true
    };
  }

  // Format the label for display
  formatDisplayName(label) {
    // Example: 'Tomato_Late_blight' -> 'Tomato Late Blight'
    return label
      .replace(/_/g, ' ')
      .replace(/\b\w/g, char => char.toUpperCase());
  }

  // Get disease description from the disease info
  getDiseaseDescription(label) {
    if (!this.diseaseInfo || !this.diseaseInfo[label]) {
      return 'No description available for this disease.';
    }
    return this.diseaseInfo[label].description || 'No description available.';
  }

  // Get disease treatment from the disease info
  getDiseaseTreatment(label) {
    if (!this.diseaseInfo || !this.diseaseInfo[label]) {
      return 'No treatment information available for this disease.';
    }
    return this.diseaseInfo[label].treatment || 'No treatment information available.';
  }

  // Get disease prevention from the disease info
  getDiseasePrevention(label) {
    if (!this.diseaseInfo || !this.diseaseInfo[label]) {
      return 'No prevention information available for this disease.';
    }
    return this.diseaseInfo[label].prevention || 'No prevention information available.';
  }
}

// Create and export a singleton instance
const offlineInference = new OfflineInference();

// Initialize when the page loads if offline mode is enabled
// document is always available in browser environment, so no null check needed
document.addEventListener('DOMContentLoaded', () => {
  const offlineEnabled = localStorage.getItem('offlineEnabled') === 'true';
  if (offlineEnabled) {
    offlineInference.init().then(success => {
      if (success) {
        console.log('Offline inference ready');
      }
    });
  }
});