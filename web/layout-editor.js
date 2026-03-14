(function () {
  var chars = Object.keys(window.MONGOLIAN_TOKEN_MAP || {}).sort(function (a, b) {
    return window.MONGOLIAN_TOKEN_MAP[a] - window.MONGOLIAN_TOKEN_MAP[b];
  });
  var tokenMap = window.MONGOLIAN_TOKEN_MAP || {};
  var NUMBER_ROW_CODES = new Set([
    "Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6",
    "Digit7", "Digit8", "Digit9", "Digit0", "Minus", "Equal"
  ]);
  var ASSIGNABLE_CODES = new Set([
    "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU", "KeyI", "KeyO", "KeyP",
    "BracketLeft", "BracketRight", "Backslash",
    "KeyA", "KeyS", "KeyD", "KeyF", "KeyG", "KeyH", "KeyJ", "KeyK", "KeyL",
    "Semicolon", "Quote",
    "KeyZ", "KeyX", "KeyC", "KeyV", "KeyB", "KeyN", "KeyM", "Comma", "Period", "Slash",
    "Enter", "Space"
  ]);

  var fontSelect = document.getElementById("font-select");
  var fontStatus = document.getElementById("font-status");
  var charPalette = document.getElementById("char-palette");
  var layoutKeyboard = document.getElementById("layout-keyboard");
  var btnSaveLayout = document.getElementById("btn-save-layout");
  var btnReloadLayout = document.getElementById("btn-reload-layout");
  var editorStatus = document.getElementById("editor-status");
  var selectedCharSymbol = document.getElementById("selected-char-symbol");
  var selectedCharMeta = document.getElementById("selected-char-meta");

  var fontStyleEl = null;
  var fontFamilyByName = {};
  var layout = null;
  var selectedChar = null;

  function sanitizeFontFamily(name) {
    return "MongolianFont_" + String(name || "").replace(/[^a-zA-Z0-9_-]/g, "_");
  }

  function ensureFontStyleElement() {
    if (!fontStyleEl) {
      fontStyleEl = document.createElement("style");
      fontStyleEl.id = "dynamic-font-faces";
      document.head.appendChild(fontStyleEl);
    }
    return fontStyleEl;
  }

  function installFontFaces(fonts) {
    var css = fonts.map(function (font) {
      var family = sanitizeFontFamily(font.name);
      var format = font.format === "ttf" ? "truetype" : font.format;
      fontFamilyByName[font.name] = family;
      return "@font-face { font-family: \"" + family + "\"; src: url(\"" + font.url + "\") format(\"" + format + "\"); }";
    }).join("\n");
    ensureFontStyleElement().textContent = css;
  }

  function applyFontSelection(fontName) {
    var family = fontFamilyByName[fontName];
    if (!family) {
      return;
    }
    document.documentElement.style.setProperty("--font-mongolian", "\"" + family + "\", \"Noto Sans Mongolian\", sans-serif");
    if (fontStatus) {
      fontStatus.textContent = "Using " + fontName;
    }
    try {
      window.localStorage.setItem("mongolianKeyboardFont", fontName);
    } catch (e) {}
  }

  function setupFontSwitcher() {
    if (!fontSelect) {
      return;
    }
    fetch("/api/fonts")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        var fonts = data.fonts || [];
        if (!fonts.length) {
          fontSelect.innerHTML = "<option value=\"\">No fonts found</option>";
          if (fontStatus) fontStatus.textContent = "No project fonts found.";
          return;
        }

        installFontFaces(fonts);
        fontSelect.innerHTML = "";
        fonts.forEach(function (font) {
          var option = document.createElement("option");
          option.value = font.name;
          option.textContent = font.label;
          fontSelect.appendChild(option);
        });

        var saved = "";
        try {
          saved = window.localStorage.getItem("mongolianKeyboardFont") || "";
        } catch (e) {}
        var defaultFont = fonts.some(function (font) { return font.name === saved; }) ? saved : fonts[0].name;
        fontSelect.value = defaultFont;
        applyFontSelection(defaultFont);
      })
      .catch(function () {
        fontSelect.innerHTML = "<option value=\"\">Font loading failed</option>";
        if (fontStatus) fontStatus.textContent = "Could not load project fonts.";
      });

    fontSelect.addEventListener("change", function () {
      applyFontSelection(fontSelect.value);
    });
  }

  function displayChar(ch) {
    if (ch === " ") {
      return "Space";
    }
    if (ch === "\n") {
      return "↵";
    }
    return ch || "Empty";
  }

  function displayToken(ch) {
    if (!ch && ch !== "") {
      return "";
    }
    if (!Object.prototype.hasOwnProperty.call(tokenMap, ch)) {
      return "";
    }
    return String(tokenMap[ch]);
  }

  function isAssignableCode(code) {
    return ASSIGNABLE_CODES.has(code);
  }

  function getAssignmentIndex() {
    var index = {};
    if (!layout || !Array.isArray(layout.rows)) {
      return index;
    }
    layout.rows.forEach(function (row, rowIndex) {
      row.forEach(function (key, keyIndex) {
        ["base", "shift"].forEach(function (layer) {
          var value = key[layer];
          if (value) {
            index[value] = {
              rowIndex: rowIndex,
              keyIndex: keyIndex,
              layer: layer,
              code: key.code
            };
          }
        });
      });
    });
    return index;
  }

  function updateSelectedCharPanel() {
    if (!selectedChar && selectedChar !== "") {
      selectedCharSymbol.textContent = "None";
      selectedCharMeta.textContent = "Choose a character below.";
      return;
    }
    selectedCharSymbol.textContent = displayChar(selectedChar);
    var token = displayToken(selectedChar);
    selectedCharMeta.textContent = token ? "Token " + token : "Not currently in token map.";
  }

  function renderPalette() {
    var assignmentIndex = getAssignmentIndex();
    charPalette.innerHTML = "";
    chars.forEach(function (ch) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "char-button";
      if (selectedChar === ch) {
        btn.classList.add("char-button-selected");
      }
      if (assignmentIndex[ch]) {
        btn.classList.add("char-button-used");
      } else {
        btn.classList.add("char-button-unused");
      }

      var charSpan = document.createElement("span");
      charSpan.className = "char-button-symbol" + (ch === " " ? " char-button-space" : "");
      charSpan.textContent = displayChar(ch);

      var tokenSpan = document.createElement("span");
      tokenSpan.className = "char-button-token";
      tokenSpan.textContent = displayToken(ch);

      btn.appendChild(charSpan);
      btn.appendChild(tokenSpan);
      btn.setAttribute("aria-label", "Choose character " + displayChar(ch));
      btn.addEventListener("click", function () {
        selectedChar = ch;
        updateSelectedCharPanel();
        renderPalette();
      });
      charPalette.appendChild(btn);
    });
  }

  function createKeyEntry(value, layer, keyCode) {
    var entry = document.createElement("span");
    entry.className = "key-entry key-entry-" + layer;
    entry.setAttribute("data-layer", layer);
    entry.setAttribute("data-code", keyCode);

    var charSpan = document.createElement("span");
    charSpan.className = "key-char" + (value === " " ? " key-char-space" : "");
    charSpan.textContent = displayChar(value);

    var tokenSpan = document.createElement("span");
    tokenSpan.className = "key-token";
    tokenSpan.textContent = displayToken(value);

    entry.appendChild(charSpan);
    entry.appendChild(tokenSpan);
    return entry;
  }

  function renderKeyboard() {
    layoutKeyboard.innerHTML = "";
    if (!layout || !Array.isArray(layout.rows)) {
      return;
    }
    layout.rows.forEach(function (row) {
      var rowEl = document.createElement("div");
      rowEl.className = "keyboard-row";
      row.forEach(function (key) {
        var assignable = isAssignableCode(key.code);
        var keyEl = document.createElement("button");
        keyEl.type = "button";
        keyEl.className = "key";
        if (key.widthClass) {
          keyEl.classList.add(key.widthClass);
        }
        if (!assignable || key.passive) {
          keyEl.classList.add("key-passive");
        } else {
          keyEl.classList.add("editor-key");
        }

        var label = document.createElement("span");
        label.className = "key-label";
        label.textContent = key.label;

        var content = document.createElement("span");
        content.className = "key-content";
        var entries = document.createElement("span");
        entries.className = "key-entries";
        entries.appendChild(createKeyEntry(key.base || "", "base", key.code));
        entries.appendChild(createKeyEntry(key.shift || "", "shift", key.code));
        content.appendChild(entries);

        keyEl.appendChild(label);
        keyEl.appendChild(content);
        if (NUMBER_ROW_CODES.has(key.code)) {
          keyEl.title = "Number-row keys are locked for Mongolian character editing.";
        }
        if (assignable && !key.passive) {
          keyEl.addEventListener("click", function (event) {
            assignSelectedCharToKey(key.code, event.shiftKey ? "shift" : "base");
          });
        }
        rowEl.appendChild(keyEl);
      });
      layoutKeyboard.appendChild(rowEl);
    });
  }

  function findKeyByCode(code) {
    if (!layout || !Array.isArray(layout.rows)) {
      return null;
    }
    for (var rowIndex = 0; rowIndex < layout.rows.length; rowIndex += 1) {
      var row = layout.rows[rowIndex];
      for (var keyIndex = 0; keyIndex < row.length; keyIndex += 1) {
        if (row[keyIndex].code === code) {
          return row[keyIndex];
        }
      }
    }
    return null;
  }

  function assignSelectedCharToKey(code, layer) {
    if (!selectedChar && selectedChar !== "") {
      editorStatus.textContent = "Select a character first.";
      return;
    }
    if (!isAssignableCode(code)) {
      editorStatus.textContent = "This key is locked. Mongolian characters can only be assigned to non-number keys.";
      return;
    }

    var key = findKeyByCode(code);
    if (!key) {
      return;
    }

    var targetLayer = layer === "shift" ? "shift" : "base";
    var currentValue = key[targetLayer] || "";
    if (currentValue === selectedChar) {
      editorStatus.textContent = "That character is already assigned to " + code + " (" + targetLayer + ").";
      return;
    }

    var assignmentIndex = getAssignmentIndex();
    var selectedLocation = assignmentIndex[selectedChar];
    key[targetLayer] = selectedChar;

    if (
      selectedLocation &&
      !(selectedLocation.code === code && selectedLocation.layer === targetLayer)
    ) {
      var sourceKey = findKeyByCode(selectedLocation.code);
      if (sourceKey) {
        sourceKey[selectedLocation.layer] = isAssignableCode(selectedLocation.code) ? currentValue : "";
      }
    }

    renderKeyboard();
    renderPalette();
    editorStatus.textContent = "Assigned " + displayChar(selectedChar) + " to " + code + " (" + targetLayer + ").";
  }

  function loadLayout() {
    editorStatus.textContent = "Loading keyboard layout…";
    return fetch("/api/keyboard-layout", { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) {
          throw new Error("Could not load keyboard layout.");
        }
        return res.json();
      })
      .then(function (data) {
        layout = data;
        editorStatus.textContent = "Keyboard layout loaded.";
        renderKeyboard();
        renderPalette();
      })
      .catch(function (err) {
        editorStatus.textContent = "Error: " + (err.message || "Could not load keyboard layout.");
      });
  }

  function saveLayout() {
    if (!layout) {
      return;
    }
    editorStatus.textContent = "Saving keyboard layout…";
    btnSaveLayout.disabled = true;
    fetch("/api/keyboard-layout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(layout),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) {
            throw new Error(data.error || "Could not save keyboard layout.");
          }
          return data;
        });
      })
      .then(function () {
        editorStatus.textContent = "Keyboard layout saved to web/keyboard-layout.json.";
      })
      .catch(function (err) {
        editorStatus.textContent = "Error: " + (err.message || "Could not save keyboard layout.");
      })
      .finally(function () {
        btnSaveLayout.disabled = false;
      });
  }

  btnSaveLayout.addEventListener("click", function () {
    saveLayout();
  });

  btnReloadLayout.addEventListener("click", function () {
    loadLayout();
  });

  setupFontSwitcher();
  updateSelectedCharPanel();
  loadLayout();
})();
