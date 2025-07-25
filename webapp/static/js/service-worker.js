// Service Worker for AgriScan - Crop Disease Detection App

const CACHE_NAME = 'agriscan-cache-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/offline',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/images/logo.svg',
  '/static/images/hero-image.jpg',
  '/static/images/upload-icon.svg',
  '/static/images/camera-icon.svg',
  '/static/images/offline-icon.svg',
  '/static/images/favicon.ico'
];

// Install event - cache assets
// self is always available in service worker context, so no null check needed
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(ASSETS_TO_CACHE);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
// self is always available in service worker context, so no null check needed
self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache or network
// self is always available in service worker context, so no null check needed
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return the response from the cached version
        if (response) {
          return response;
        }
        
        // Not in cache - return the result from the live server
        // Clone the request because it's a one-time use stream
        return fetch(event.request.clone())
          .then(response => {
            // Check if we received a valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }
            
            // Clone the response because it's a one-time use stream
            const responseToCache = response.clone();
            
            // Only cache GET requests
            if (event.request.method === 'GET') {
              caches.open(CACHE_NAME)
                .then(cache => {
                  cache.put(event.request, responseToCache);
                });
            }
            
            return response;
          });
      })
      .catch(error => {
        // If both cache and network fail (offline), serve the offline page
        if (event.request.mode === 'navigate') {
          return caches.match('/offline');
        }
        
        // For image requests, return a placeholder
        if (event.request.destination === 'image') {
          return caches.match('/static/images/offline-icon.svg');
        }
        
        // For other resources, just return an error response
        return new Response('Network error happened', {
          status: 408,
          headers: { 'Content-Type': 'text/plain' }
        });
      })
  );
});

// Background sync for offline image uploads
// self is always available in service worker context, so no null check needed
self.addEventListener('sync', event => {
  if (event.tag === 'sync-images') {
    event.waitUntil(syncImages());
  }
});

// Function to sync images when back online
async function syncImages() {
  try {
    // Get all pending uploads from IndexedDB
    const db = await openDB();
    const pendingUploads = await db.getAll('pending-uploads');
    
    // Process each pending upload
    for (const upload of pendingUploads) {
      try {
        // Create a FormData object
        const formData = new FormData();
        formData.append('file', upload.file);
        
        // Send the request
        const response = await fetch('/upload', {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          // If successful, remove from pending uploads
          await db.delete('pending-uploads', upload.id);
          
          // Notify the client if possible
          const clients = await self.clients.matchAll();
          clients.forEach(client => {
            client.postMessage({
              type: 'SYNC_SUCCESS',
              id: upload.id
            });
          });
        }
      } catch (error) {
        console.error('Error syncing image:', error);
      }
    }
  } catch (error) {
    console.error('Error in syncImages:', error);
  }
}

// Helper function to open IndexedDB
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('agriscan-db', 1);
    
    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending-uploads')) {
        db.createObjectStore('pending-uploads', { keyPath: 'id' });
      }
    };
    
    request.onsuccess = event => resolve(event.target.result);
    request.onerror = event => reject(event.target.error);
  });
}