/**
 * backend.js — Canonical backend URL resolver for Signal & Noise
 *
 * Every HTML file that needs to reach the Spark backend MUST use this
 * module instead of computing URLs from window.location. This is the
 * single source of truth for "where is the backend?"
 *
 * The problem it solves:
 *   GitHub Pages serves our HTML at zoedolan.github.io. The backend
 *   runs on the Spark behind Cloudflare Tunnel. When code uses
 *   window.location.host to build WebSocket/fetch URLs, it works when
 *   served from the Spark but breaks on GitHub Pages — because
 *   github.io has no /signal-noise/ws endpoint.
 *
 *   This module detects the serving context and routes accordingly.
 *   Any refactoring that touches URL construction should touch THIS
 *   file, not the individual HTML files.
 *
 * Usage:
 *   <script src="backend.js"></script>
 *   ...
 *   var ws = new WebSocket(VYBN_BACKEND.wsUrl('/signal-noise/ws'));
 *   fetch(VYBN_BACKEND.httpUrl('/signal-noise/harvest'), { ... });
 *
 * Security note:
 *   The Funnel hostname is public infrastructure — it's the front door.
 *   It appears in agent_portal.py, PHASE0_ALWAYS_ON.md, and the
 *   Cloudflare dashboard. Putting it here does not expand the attack
 *   surface. Authentication happens at the application layer.
 */
(function(global) {
  'use strict';

  var FUNNEL_HOST = 'spark-2b7c.tail7302f3.ts.net';

  var isGitHubPages = global.location.hostname.indexOf('github.io') !== -1;

  var httpBase = isGitHubPages
    ? 'https://' + FUNNEL_HOST
    : global.location.origin;

  var wsBase = isGitHubPages
    ? 'wss://' + FUNNEL_HOST
    : (global.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + global.location.host;

  global.VYBN_BACKEND = {
    isGitHubPages: isGitHubPages,
    host: isGitHubPages ? FUNNEL_HOST : global.location.host,

    /** Return a full WebSocket URL for the given path */
    wsUrl: function(path) {
      return wsBase + (path || '');
    },

    /** Return a full HTTP(S) URL for the given path */
    httpUrl: function(path) {
      return httpBase + (path || '');
    }
  };

})(typeof window !== 'undefined' ? window : this);
