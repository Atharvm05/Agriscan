// IndexedDB for AgriScan - Handles offline storage

class AgriScanDB {
  constructor() {
    this.dbName = 'agriscan-db';
    this.dbVersion = 1;
    this.db = null;
    this.pendingUploadsStore = 'pending-uploads';
    this.resultsStore = 'analysis-results';
    this.modelStore = 'models';
  }

  // Initialize the database
  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create object stores if they don't exist
        if (!db.objectStoreNames.contains(this.pendingUploadsStore)) {
          db.createObjectStore(this.pendingUploadsStore, { keyPath: 'id' });
        }

        if (!db.objectStoreNames.contains(this.resultsStore)) {
          db.createObjectStore(this.resultsStore, { keyPath: 'id' });
        }

        if (!db.objectStoreNames.contains(this.modelStore)) {
          db.createObjectStore(this.modelStore, { keyPath: 'id' });
        }
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        console.log('Database initialized successfully');
        resolve();
      };

      request.onerror = (event) => {
        console.error('Error opening database:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Add a pending upload to the database
  async addPendingUpload(file, metadata = {}) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.pendingUploadsStore], 'readwrite');
      const store = transaction.objectStore(this.pendingUploadsStore);

      const id = 'upload_' + Date.now();
      const item = {
        id,
        file,
        timestamp: Date.now(),
        status: 'pending',
        metadata
      };

      const request = store.add(item);

      request.onsuccess = () => {
        console.log('Pending upload added to database');
        resolve(id);
      };

      request.onerror = (event) => {
        console.error('Error adding pending upload:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Get all pending uploads
  async getPendingUploads() {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.pendingUploadsStore], 'readonly');
      const store = transaction.objectStore(this.pendingUploadsStore);
      const request = store.getAll();

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = (event) => {
        console.error('Error getting pending uploads:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Remove a pending upload
  async removePendingUpload(id) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.pendingUploadsStore], 'readwrite');
      const store = transaction.objectStore(this.pendingUploadsStore);
      const request = store.delete(id);

      request.onsuccess = () => {
        console.log('Pending upload removed from database');
        resolve();
      };

      request.onerror = (event) => {
        console.error('Error removing pending upload:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Save analysis results
  async saveResults(results) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.resultsStore], 'readwrite');
      const store = transaction.objectStore(this.resultsStore);

      const id = 'result_' + Date.now();
      const item = {
        id,
        ...results,
        timestamp: Date.now()
      };

      const request = store.add(item);

      request.onsuccess = () => {
        console.log('Analysis results saved to database');
        resolve(id);
      };

      request.onerror = (event) => {
        console.error('Error saving analysis results:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Get analysis results by ID
  async getResultById(id) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.resultsStore], 'readonly');
      const store = transaction.objectStore(this.resultsStore);
      const request = store.get(id);

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = (event) => {
        console.error('Error getting analysis result:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Get all analysis results
  async getAllResults() {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.resultsStore], 'readonly');
      const store = transaction.objectStore(this.resultsStore);
      const request = store.getAll();

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = (event) => {
        console.error('Error getting all analysis results:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Save or update model
  async saveModel(modelId, modelBuffer, metadata = {}) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.modelStore], 'readwrite');
      const store = transaction.objectStore(this.modelStore);

      const item = {
        id: modelId,
        model: modelBuffer,
        metadata,
        timestamp: Date.now()
      };

      const request = store.put(item); // Use put to update if exists

      request.onsuccess = () => {
        console.log('Model saved to database');
        resolve();
      };

      request.onerror = (event) => {
        console.error('Error saving model:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Get model by ID
  async getModel(modelId) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.modelStore], 'readonly');
      const store = transaction.objectStore(this.modelStore);
      const request = store.get(modelId);

      request.onsuccess = () => {
        resolve(request.result);
      };

      request.onerror = (event) => {
        console.error('Error getting model:', event.target.error);
        reject(event.target.error);
      };
    });
  }

  // Check if model exists
  async modelExists(modelId) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([this.modelStore], 'readonly');
      const store = transaction.objectStore(this.modelStore);
      const request = store.count(modelId);

      request.onsuccess = () => {
        resolve(request.result > 0);
      };

      request.onerror = (event) => {
        console.error('Error checking if model exists:', event.target.error);
        reject(event.target.error);
      };
    });
  }
}

// Create and export a singleton instance
const agriscanDB = new AgriScanDB();

// Initialize the database when the script loads
agriscanDB.init().catch(error => {
  console.error('Failed to initialize database:', error);
});