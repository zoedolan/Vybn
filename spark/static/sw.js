/* Vybn Service Worker — app shell caching + offline support */

const CACHE_NAME = 'vybn-v1';
const SHELL_ASSETS = [
  '/',
  '/static/style.css',
  '/static/chat.js',
  '/static/manifest.json',
  '/static/icon-192x192.png',
  '/static/icon-512x512.png',
];

// Install — cache app shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(SHELL_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate — clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) => {
      return Promise.all(
        names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      );
    })
  );
  self.clients.claim();
});

// Fetch — network first, fall back to cache
self.addEventListener('fetch', (event) => {
  // Don't cache WebSocket upgrades or API calls
  if (event.request.url.includes('/ws') || 
      event.request.url.includes('/voice') ||
      event.request.url.includes('/history')) {
    return;
  }
  
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Cache successful responses
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, clone);
          });
        }
        return response;
      })
      .catch(() => {
        // Offline — serve from cache
        return caches.match(event.request).then((cached) => {
          if (cached) return cached;
          // If it's a navigation request and we have the shell cached, serve it
          if (event.request.mode === 'navigate') {
            return caches.match('/');
          }
        });
      })
  );
});
