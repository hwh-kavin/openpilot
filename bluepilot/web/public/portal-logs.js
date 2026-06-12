(function () {
  'use strict';

  // Runtime patch for production bundle: replace WebSocket log streaming with
  // lightweight HTTP polling against /api/manager-logs (swaglog files).
  if (!window.location.pathname.startsWith('/logs')) {
    return;
  }

  const POLL_MS = 2000;
  const NativeWebSocket = window.WebSocket;
  let onMessage = null;
  let seenLines = new Set();
  let bootstrapped = false;
  let pollTimer = null;

  function isPaused() {
    const status = document.querySelector('.console-status-value');
    return status && status.classList.contains('paused');
  }

  function deliverLine(line) {
    if (!onMessage || isPaused()) {
      return;
    }
    onMessage({
      data: JSON.stringify({
        type: 'log_line',
        data: { line: line },
      }),
    });
  }

  async function pollLogs() {
    if (isPaused()) {
      return;
    }

    try {
      const response = await fetch('/api/manager-logs', { cache: 'no-store' });
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      if (!data.success || !data.output) {
        return;
      }

      const lines = data.output.split('\n').filter(function (line) {
        return line.trim();
      });

      if (!bootstrapped) {
        lines.forEach(function (line) {
          seenLines.add(line);
        });
        bootstrapped = true;
        hideWsError();
        return;
      }

      let added = false;
      lines.forEach(function (line) {
        if (seenLines.has(line)) {
          return;
        }
        seenLines.add(line);
        deliverLine(line);
        added = true;
      });

      if (added) {
        hideWsError();
      }
    } catch (error) {
      console.warn('[portal-logs] poll failed', error);
    }
  }

  function hideWsError() {
    document.querySelectorAll('.console-error').forEach(function (el) {
      const text = (el.textContent || '').trim();
      if (text.includes('WebSocket') || text.includes('连接错误')) {
        el.textContent = '';
        el.style.display = 'none';
      }
    });
  }

  function startPolling() {
    if (pollTimer !== null) {
      return;
    }
    pollLogs();
    pollTimer = window.setInterval(pollLogs, POLL_MS);
  }

  function stopPolling() {
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  window.WebSocket = function (url, protocols) {
    if (typeof url === 'string' && url.indexOf(':8089') !== -1) {
      const stub = {
        readyState: 1,
        close: function () {
          this.readyState = 3;
          stopPolling();
        },
        send: function () {},
      };

      Object.defineProperty(stub, 'onmessage', {
        get: function () {
          return onMessage;
        },
        set: function (fn) {
          onMessage = fn;
        },
      });

      Object.defineProperty(stub, 'onopen', {
        set: function (fn) {
          if (typeof fn === 'function') {
            window.setTimeout(fn, 0);
          }
          startPolling();
        },
      });

      Object.defineProperty(stub, 'onerror', {
        set: function () {},
      });

      Object.defineProperty(stub, 'onclose', {
        set: function () {},
      });

      return stub;
    }

    return new NativeWebSocket(url, protocols);
  };

  Object.assign(window.WebSocket, NativeWebSocket);
  window.WebSocket.prototype = NativeWebSocket.prototype;

  document.addEventListener('click', function (event) {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const button = target.closest('button');
    if (!button) {
      return;
    }
    const label = (button.textContent || '').trim().toLowerCase();
    if (label === 'clear' || label === '清除') {
      seenLines.clear();
      bootstrapped = false;
    }
  });
})();
