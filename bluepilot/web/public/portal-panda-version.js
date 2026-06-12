(function () {
  'use strict';

  const PANDA_ATTR = 'data-bp-panda-version-pill';
  const PANDA_LABELS = ['Panda Version', 'Panda 版本'];
  const AGNOS_LABELS = ['Agnos Version', 'Agnos 版本'];
  const locale = window.__PORTAL_LOCALE__ || 'en';
  const label = String(locale).startsWith('zh') ? 'Panda 版本' : 'Panda Version';

  let cachedVersion = null;

  function findVersionRow() {
    const rows = document.querySelectorAll('.status-pills-container .status-pills-row');
    for (const row of rows) {
      const labels = Array.from(row.querySelectorAll('.pill-label')).map(function (el) {
        return el.textContent.trim();
      });
      if (labels.some(function (t) { return AGNOS_LABELS.indexOf(t) >= 0; })) {
        return row;
      }
    }
    return null;
  }

  function findExistingPandaPill(row) {
    const marked = row.querySelector('[' + PANDA_ATTR + ']');
    if (marked) {
      return marked;
    }
    const pills = row.querySelectorAll('.status-pill');
    for (const pill of pills) {
      const pillLabel = pill.querySelector('.pill-label');
      if (pillLabel && PANDA_LABELS.indexOf(pillLabel.textContent.trim()) >= 0) {
        pill.setAttribute(PANDA_ATTR, '1');
        return pill;
      }
    }
    return null;
  }

  function ensurePandaPill(version) {
    const row = findVersionRow();
    if (!row) {
      return false;
    }

    let pill = findExistingPandaPill(row);
    if (!pill) {
      pill = document.createElement('div');
      pill.className = 'status-pill';
      pill.setAttribute(PANDA_ATTR, '1');
      pill.innerHTML =
        '<span class="pill-label">' + label + '</span>' +
        '<span class="pill-value"></span>';
      row.appendChild(pill);
    }

    const valueEl = pill.querySelector('.pill-value');
    const display = version || 'N/A';
    if (valueEl) {
      valueEl.textContent = display;
    }
    pill.title = version || '';
    return true;
  }

  function refresh() {
    return fetch('/api/system/device-info', { cache: 'no-store' })
      .then(function (resp) {
        if (!resp.ok) {
          return null;
        }
        return resp.json();
      })
      .then(function (data) {
        if (!data) {
          return;
        }
        cachedVersion = data.panda_version || null;
        ensurePandaPill(cachedVersion);
      })
      .catch(function () {});
  }

  function start() {
    refresh();
    window.setInterval(refresh, 30000);

    const root = document.getElementById('root');
    if (!root) {
      return;
    }

    const observer = new MutationObserver(function () {
      const row = findVersionRow();
      if (!row) {
        return;
      }
      if (!findExistingPandaPill(row)) {
        ensurePandaPill(cachedVersion);
      }
    });

    observer.observe(root, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
