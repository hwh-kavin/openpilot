(function () {
  'use strict';

  const locale = window.__PORTAL_LOCALE__ || 'en';
  if (!String(locale).startsWith('zh')) {
    return;
  }

  let phrases = [];
  let paramLabels = null;
  let phrasesReady = false;

  const loaders = [
    fetch('/i18n/phrases-zh-CHS.json', { cache: 'no-store' })
      .then((response) => {
        if (!response.ok) {
          throw new Error('failed to load portal phrases');
        }
        return response.json();
      })
      .then((data) => {
        phrases = Object.entries(data).sort((a, b) => b[0].length - a[0].length);
        phrasesReady = true;
      }),
    fetch('/api/param-labels', { cache: 'no-store' })
      .then((response) => {
        if (!response.ok) {
          throw new Error('failed to load param labels');
        }
        return response.json();
      })
      .then((data) => {
        if (data && data.success) {
          paramLabels = data.labels || null;
        }
      }),
  ];

  Promise.allSettled(loaders)
    .then(() => {
      applyAll();
      observe();
    })
    .catch((error) => {
      console.warn('[portal-i18n]', error);
      observe();
    });

  function translate(text) {
    if (!phrasesReady || !text) {
      return text;
    }
    let out = text;
    for (const [en, zh] of phrases) {
      if (out.includes(en)) {
        out = out.split(en).join(zh);
      }
    }
    return out;
  }

  function applyParamLabels() {
    if (!paramLabels || !window.location.pathname.startsWith('/parameters')) {
      return;
    }

    document.querySelectorAll('.param-key').forEach((el) => {
      const key = (el.getAttribute('data-param-key') || el.getAttribute('title') || el.textContent || '').trim();
      const label = paramLabels[key];
      if (!label || !label.title) {
        return;
      }
      el.setAttribute('data-param-key', key);
      el.textContent = label.title;
      el.setAttribute('title', key);
    });

    document.querySelectorAll('.param-description p').forEach((el) => {
      const row = el.closest('.param-row');
      if (!row) {
        return;
      }
      const keyEl = row.querySelector('.param-key');
      const key = keyEl && keyEl.getAttribute('data-param-key');
      if (!key || !paramLabels[key] || !paramLabels[key].description) {
        return;
      }
      const zh = paramLabels[key].description;
      if (el.textContent !== zh) {
        el.textContent = zh;
      }
    });

    document.querySelectorAll('.param-description .description-label').forEach((el) => {
      if (el.textContent === 'Description') {
        el.textContent = '说明';
      }
    });
  }

  function patchElement(el) {
    if (!(el instanceof HTMLElement)) {
      return;
    }

    for (const attr of ['title', 'placeholder', 'aria-label']) {
      const value = el.getAttribute(attr);
      if (!value) {
        continue;
      }
      const next = translate(value);
      if (next !== value) {
        el.setAttribute(attr, next);
      }
    }
  }

  function patchTextNode(node) {
    const text = node.textContent;
    if (!text || !text.trim()) {
      return;
    }
    const next = translate(text);
    if (next !== text) {
      node.textContent = next;
    }
  }

  function walk(root) {
    if (!root) {
      return;
    }

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
      if (node.nodeType === Node.TEXT_NODE) {
        patchTextNode(node);
      } else {
        patchElement(node);
      }
      node = walker.nextNode();
    }
  }

  let scheduled = false;
  function applyAll() {
    if (scheduled) {
      return;
    }
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      walk(document.getElementById('root') || document.body);
      applyParamLabels();
    });
  }

  function observe() {
    const root = document.getElementById('root') || document.body;
    if (!root) {
      return;
    }

    new MutationObserver(() => applyAll()).observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ['title', 'placeholder', 'aria-label'],
    });

    applyAll();
    window.setInterval(applyAll, 1500);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', observe);
  } else {
    observe();
  }
})();
