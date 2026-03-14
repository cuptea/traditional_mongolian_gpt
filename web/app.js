(function () {
  var chars = window.MONGOLIAN_CHARS;
  var tokenMap = window.MONGOLIAN_TOKEN_MAP || {};
  if (!chars || !chars.length) {
    document.getElementById("keyboard").textContent = "No character set loaded.";
    return;
  }
  var SPACE_CHAR = chars.indexOf(" ") !== -1 ? " " : "";
  var NEWLINE_CHAR = Object.prototype.hasOwnProperty.call(tokenMap, "\n") ? "\n" : "";
  var DEFAULT_PHYSICAL_KEY_ROWS = [
    [
      { code: "Backquote", label: "`" },
      { code: "Digit1", label: "1" },
      { code: "Digit2", label: "2" },
      { code: "Digit3", label: "3" },
      { code: "Digit4", label: "4" },
      { code: "Digit5", label: "5" },
      { code: "Digit6", label: "6" },
      { code: "Digit7", label: "7" },
      { code: "Digit8", label: "8" },
      { code: "Digit9", label: "9" },
      { code: "Digit0", label: "0" },
      { code: "Minus", label: "-" },
      { code: "Equal", label: "=" },
      { code: "Backspace", label: "Backspace", widthClass: "key-backspace" },
    ],
    [
      { code: "Tab", label: "Tab", widthClass: "key-tab", passive: true },
      { code: "KeyQ", label: "Q" },
      { code: "KeyW", label: "W" },
      { code: "KeyE", label: "E" },
      { code: "KeyR", label: "R" },
      { code: "KeyT", label: "T" },
      { code: "KeyY", label: "Y" },
      { code: "KeyU", label: "U" },
      { code: "KeyI", label: "I" },
      { code: "KeyO", label: "O" },
      { code: "KeyP", label: "P" },
      { code: "BracketLeft", label: "[" },
      { code: "BracketRight", label: "]" },
      { code: "Backslash", label: "\\" },
    ],
    [
      { code: "CapsLock", label: "Caps", widthClass: "key-caps", passive: true },
      { code: "KeyA", label: "A" },
      { code: "KeyS", label: "S" },
      { code: "KeyD", label: "D" },
      { code: "KeyF", label: "F" },
      { code: "KeyG", label: "G" },
      { code: "KeyH", label: "H" },
      { code: "KeyJ", label: "J" },
      { code: "KeyK", label: "K" },
      { code: "KeyL", label: "L" },
      { code: "Semicolon", label: ";" },
      { code: "Quote", label: "'" },
      { code: "Enter", label: "Enter", widthClass: "key-enter" },
    ],
    [
      { code: "ShiftLeft", label: "Shift", widthClass: "key-shift", passive: true },
      { code: "KeyZ", label: "Z" },
      { code: "KeyX", label: "X" },
      { code: "KeyC", label: "C" },
      { code: "KeyV", label: "V" },
      { code: "KeyB", label: "B" },
      { code: "KeyN", label: "N" },
      { code: "KeyM", label: "M" },
      { code: "Comma", label: "," },
      { code: "Period", label: "." },
      { code: "Slash", label: "/" },
      { code: "ShiftRight", label: "Shift", widthClass: "key-shift", passive: true },
    ],
    [
      { code: "Space", label: "Space", widthClass: "key-space" },
    ],
  ];
  var CHARACTER_KEY_CODES = [
    "Backquote", "Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6",
    "Digit7", "Digit8", "Digit9", "Digit0", "Minus", "Equal",
    "KeyQ", "KeyW", "KeyE", "KeyR", "KeyT", "KeyY", "KeyU", "KeyI", "KeyO", "KeyP",
    "BracketLeft", "BracketRight", "Backslash",
    "KeyA", "KeyS", "KeyD", "KeyF", "KeyG", "KeyH", "KeyJ", "KeyK", "KeyL",
    "Semicolon", "Quote",
    "KeyZ", "KeyX", "KeyC", "KeyV", "KeyB", "KeyN", "KeyM", "Comma", "Period", "Slash",
  ];

  var inputField = document.getElementById("input-field");
  var inputDisplay = document.getElementById("input-display");
  var inputText = document.getElementById("input-text");
  var inputPlaceholder = document.getElementById("input-placeholder");
  var keyboardEl = document.getElementById("keyboard");
  var btnBackspace = document.getElementById("btn-backspace");
  var btnClear = document.getElementById("btn-clear");
  var btnCopy = document.getElementById("btn-copy");
  var btnExportPdf = document.getElementById("btn-export-pdf");
  var btnRefreshSuggestions = document.getElementById("btn-refresh-suggestions");
  var fontSelect = document.getElementById("font-select");
  var fontStatus = document.getElementById("font-status");
  var currentFontName = document.getElementById("current-font-name");
  var suggestionsEl = document.getElementById("suggestions");
  var suggestionsHint = document.getElementById("suggestions-hint");
  var fontStyleEl = null;
  var fontFamilyByName = {};
  var suggestTimer = null;
  var keyElementsByCode = {};
  var boundaryChars = new Set(
    Object.keys(tokenMap).filter(function (ch) {
      return tokenMap[ch] <= 14;
    })
  );
  var keyboardLayout = buildDefaultKeyboardLayout();
  var physicalKeyMap = buildPhysicalKeyMap(keyboardLayout);

  function buildDefaultKeyboardLayout() {
    var layout = { version: 1, rows: [] };
    var map = {};
    var remainingChars = chars.filter(function (ch) {
      return ch !== SPACE_CHAR;
    });

    CHARACTER_KEY_CODES.forEach(function (code) {
      map[code] = { base: "", shift: "" };
    });

    remainingChars.forEach(function (ch, idx) {
      var code = CHARACTER_KEY_CODES[idx % CHARACTER_KEY_CODES.length];
      if (!map[code]) {
        map[code] = { base: "", shift: "" };
      }
      if (idx < CHARACTER_KEY_CODES.length) {
        map[code].base = ch;
      } else {
        map[code].shift = ch;
      }
    });

    if (SPACE_CHAR) {
      map.Space = { base: SPACE_CHAR, shift: SPACE_CHAR };
    }
    if (NEWLINE_CHAR) {
      map.Enter = { base: NEWLINE_CHAR, shift: NEWLINE_CHAR };
    }
    DEFAULT_PHYSICAL_KEY_ROWS.forEach(function (rowSpec) {
      var row = rowSpec.map(function (spec) {
        var item = {
          code: spec.code,
          label: spec.label,
        };
        if (spec.widthClass) {
          item.widthClass = spec.widthClass;
        }
        if (spec.passive) {
          item.passive = true;
        }
        if (map[spec.code]) {
          item.base = map[spec.code].base || "";
          item.shift = map[spec.code].shift || "";
        }
        return item;
      });
      layout.rows.push(row);
    });
    return layout;
  }

  function buildPhysicalKeyMap(layout) {
    var map = {};
    var rows = layout && Array.isArray(layout.rows) ? layout.rows : [];
    rows.forEach(function (row) {
      row.forEach(function (spec) {
        if (spec.code) {
          map[spec.code] = {
            base: spec.base || "",
            shift: spec.shift || "",
          };
        }
      });
    });
    return map;
  }

  function loadKeyboardLayout() {
    return fetch("keyboard-layout.json", { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) {
          throw new Error("Keyboard layout file not found.");
        }
        return res.json();
      })
      .then(function (data) {
        if (!data || !Array.isArray(data.rows)) {
          throw new Error("Keyboard layout JSON is invalid.");
        }
        return data;
      });
  }

  function getMappedChar(code, useShift) {
    var entry = physicalKeyMap[code];
    if (!entry) {
      return "";
    }
    if (useShift && entry.shift) {
      return entry.shift;
    }
    return entry.base || "";
  }

  function displayChar(ch) {
    if (ch === " ") {
      return " ";
    }
    if (ch === "\n") {
      return "↵";
    }
    return ch;
  }

  function setPressedKeyState(code, isPressed) {
    var el = keyElementsByCode[code];
    if (!el) {
      return;
    }
    el.classList.toggle("key-active", !!isPressed);
  }

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
    if (currentFontName) {
      currentFontName.textContent = fontName.replace(/\.[^.]+$/, "");
    }
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

  function getValue() {
    return inputField.value || "";
  }

  function isBoundaryChar(ch) {
    return boundaryChars.has(ch);
  }

  function extractCurrentWordPrefix(text) {
    if (!text) {
      return "";
    }
    var charsList = Array.from(text);
    if (charsList.length === 0) {
      return "";
    }
    if (isBoundaryChar(charsList[charsList.length - 1])) {
      return "";
    }

    var idx = charsList.length - 1;
    while (idx >= 0 && !isBoundaryChar(charsList[idx])) {
      idx -= 1;
    }
    return charsList.slice(idx + 1).join("");
  }

  function collapseRepeatedSpaces(text) {
    if (!SPACE_CHAR || !text) {
      return text || "";
    }
    var escapedSpace = SPACE_CHAR.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    var repeatedSpace = new RegExp(escapedSpace + "{2,}", "g");
    return text.replace(repeatedSpace, SPACE_CHAR);
  }

  function setValue(s) {
    var normalized = collapseRepeatedSpaces(s || "");
    inputField.value = normalized;
    updateDisplay(normalized);
    scheduleSuggestions(normalized);
  }

  function updateDisplay(text) {
    var t = text.replace(/\n/g, "\n");
    inputText.textContent = t;
    inputPlaceholder.style.display = t.length ? "none" : "inline";
  }

  inputField.addEventListener("input", function () {
    updateDisplay(getValue());
    scheduleSuggestions(getValue());
  });

  inputField.addEventListener("keydown", function (e) {
    if (e.key === "Backspace" && !getValue() && e.preventDefault) e.preventDefault();
  });

  function appendChar(char) {
    if (!char) {
      return;
    }
    setValue(getValue() + char);
    inputField.focus();
  }

  function backspaceOneChar() {
    var v = getValue();
    if (v.length) {
      setValue(v.slice(0, -1));
    }
    inputField.focus();
  }

  function applyPhysicalKey(code, useShift) {
    if (code === "Backspace") {
      backspaceOneChar();
      return true;
    }
    var mapped = getMappedChar(code, useShift);
    if (!mapped) {
      return false;
    }
    appendChar(mapped);
    return true;
  }

  function createKeyEntry(char, layerClass) {
    var tokenId = tokenMap[char];
    var entry = document.createElement("span");
    entry.className = "key-entry " + layerClass;
    var charSpan = document.createElement("span");
    charSpan.className = "key-char" + (char === SPACE_CHAR ? " key-char-space" : "");
    charSpan.textContent = displayChar(char);
    var tokenSpan = document.createElement("span");
    tokenSpan.className = "key-token";
    tokenSpan.textContent = String(tokenId);
    entry.appendChild(charSpan);
    entry.appendChild(tokenSpan);
    return entry;
  }

  function renderKeyboard() {
    keyboardEl.innerHTML = "";
    keyElementsByCode = {};
    (keyboardLayout.rows || []).forEach(function (rowSpec) {
      var row = document.createElement("div");
      row.className = "keyboard-row";
      rowSpec.forEach(function (spec) {
        var mappedBase = getMappedChar(spec.code, false);
        var mappedShift = getMappedChar(spec.code, true);
        var isClickable = spec.code === "Backspace" || !!mappedBase || !!mappedShift;
        var keyEl = document.createElement(isClickable ? "button" : "div");
        keyEl.className = "key";
        keyEl.setAttribute("data-code", spec.code);
        if (spec.widthClass) {
          keyEl.classList.add(spec.widthClass);
        }
        if (!isClickable || spec.passive) {
          keyEl.classList.add("key-passive");
        }
        if (isClickable) {
          keyEl.type = "button";
        }

        var label = document.createElement("span");
        label.className = "key-label";
        label.textContent = spec.label;

        var content = document.createElement("span");
        content.className = "key-content";
        var entries = document.createElement("span");
        entries.className = "key-entries";
        if (mappedBase) {
          entries.appendChild(createKeyEntry(mappedBase, "key-entry-base"));
        }
        if (mappedShift && mappedShift !== mappedBase) {
          entries.appendChild(createKeyEntry(mappedShift, "key-entry-shift"));
        }
        content.appendChild(entries);

        keyEl.appendChild(label);
        keyEl.appendChild(content);
        keyElementsByCode[spec.code] = keyEl;

        if (isClickable) {
          keyEl.setAttribute("aria-label", spec.label);
          keyEl.addEventListener("click", function (event) {
            applyPhysicalKey(spec.code, !!event.shiftKey);
          });
        }

        row.appendChild(keyEl);
      });
      keyboardEl.appendChild(row);
    });
  }

  btnBackspace.addEventListener("click", function () {
    backspaceOneChar();
  });

  btnClear.addEventListener("click", function () {
    setValue("");
    inputField.focus();
  });

  document.addEventListener("keydown", function (event) {
    var tagName = event.target && event.target.tagName ? event.target.tagName.toUpperCase() : "";
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
    if (tagName === "SELECT" || tagName === "OPTION") {
      return;
    }

    if (event.code === "ShiftLeft" || event.code === "ShiftRight") {
      setPressedKeyState(event.code, true);
      return;
    }

    if (event.code === "Tab") {
      return;
    }

    if (applyPhysicalKey(event.code, !!event.shiftKey)) {
      event.preventDefault();
      setPressedKeyState(event.code, true);
    }
  });

  document.addEventListener("keyup", function (event) {
    setPressedKeyState(event.code, false);
  });

  window.addEventListener("blur", function () {
    Object.keys(keyElementsByCode).forEach(function (code) {
      setPressedKeyState(code, false);
    });
  });

  function showSuggestions(suggestions) {
    suggestionsEl.innerHTML = "";
    suggestionsHint.textContent = "";
    if (!suggestions || suggestions.length === 0) {
      suggestionsHint.textContent = "No suggestions.";
      return;
    }
    suggestions.slice(0, 9).forEach(function (suggestion, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-secondary btn-suggestion";
      var num = document.createElement("span");
      num.className = "suggestion-num";
      num.textContent = (i + 1) + ": ";
      var textSpan = document.createElement("span");
      textSpan.className = "suggestion-text";
      textSpan.textContent = suggestion.displayText;
      btn.appendChild(num);
      btn.appendChild(textSpan);
      btn.setAttribute("aria-label", "Use suggestion " + (i + 1));
      btn.addEventListener("click", function () {
        setValue(suggestion.nextValue);
        inputField.focus();
      });
      suggestionsEl.appendChild(btn);
    });
  }

  function fetchAndShowSuggestions(text) {
    if (text == null) {
      suggestionsHint.textContent = "No suggestions.";
      suggestionsEl.innerHTML = "";
      return;
    }
    if (text.length === 0) {
      suggestionsHint.textContent = "No suggestions.";
      suggestionsEl.innerHTML = "";
      return;
    }
    var context = text;
    suggestionsHint.textContent = "Loading…";
    suggestionsEl.innerHTML = "";
    fetch("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: context }),
    })
      .then(function (res) {
        return res.text().then(function (raw) {
          var trimmed = raw.trim();
          if (trimmed.charAt(0) === "<") {
            throw new Error("Autocomplete API not available. Run the app with the Flask server from the project root: python web/server.py");
          }
          var data;
          try {
            data = JSON.parse(raw);
          } catch (e) {
            throw new Error("Autocomplete API not available. Run the app with: python web/server.py");
          }
          if (!res.ok) throw new Error(data.error || "Request failed");
          return data;
        });
      })
      .then(function (data) {
        var completions = data.completions || [];
        var currentWordPrefix = extractCurrentWordPrefix(text);
        var suggestions = completions.map(function (completion) {
          return {
            displayText: currentWordPrefix + completion,
            nextValue: text + completion,
          };
        }).sort(function (a, b) {
          return a.displayText.localeCompare(b.displayText);
        });
        showSuggestions(suggestions);
        if (suggestions.length === 0) {
          suggestionsHint.textContent = "No suggestions.";
        }
      })
      .catch(function (err) {
        suggestionsHint.textContent = "Error: " + (err.message || "Could not get suggestions.");
        suggestionsEl.innerHTML = "";
      });
  }

  function scheduleSuggestions(text) {
    if (suggestTimer) {
      window.clearTimeout(suggestTimer);
    }
    suggestTimer = window.setTimeout(function () {
      fetchAndShowSuggestions(text);
    }, 200);
  }

  function refreshSuggestionsNow() {
    if (suggestTimer) {
      window.clearTimeout(suggestTimer);
      suggestTimer = null;
    }
    fetchAndShowSuggestions(getValue());
  }

  function exportPdf() {
    var text = getValue();
    if (!text || !btnExportPdf) {
      return;
    }

    var originalLabel = btnExportPdf.textContent;
    btnExportPdf.disabled = true;
    btnExportPdf.textContent = "Exporting…";

    fetch("/api/export/pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        font_name: fontSelect ? fontSelect.value : "",
      }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.text().then(function (raw) {
            var message = "Could not export PDF.";
            try {
              var data = JSON.parse(raw);
              message = data.error || message;
            } catch (e) {}
            throw new Error(message);
          });
        }
        return res.blob().then(function (blob) {
          var disposition = res.headers.get("Content-Disposition") || "";
          var match = disposition.match(/filename="?([^"]+)"?/i);
          return {
            blob: blob,
            filename: match ? match[1] : "mongolian-text.pdf",
          };
        });
      })
      .then(function (result) {
        var url = window.URL.createObjectURL(result.blob);
        var link = document.createElement("a");
        link.href = url;
        link.download = result.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        btnExportPdf.textContent = "PDF ready";
        setTimeout(function () {
          btnExportPdf.textContent = originalLabel;
        }, 1500);
      })
      .catch(function () {
        btnExportPdf.textContent = "Export failed";
        setTimeout(function () {
          btnExportPdf.textContent = originalLabel;
        }, 1500);
      })
      .finally(function () {
        btnExportPdf.disabled = false;
      });
  }

  btnCopy.addEventListener("click", function () {
    var v = getValue();
    if (!v) return;
    navigator.clipboard.writeText(v).then(
      function () {
        var old = btnCopy.textContent;
        btnCopy.textContent = "Copied!";
        setTimeout(function () { btnCopy.textContent = old; }, 1500);
      },
      function () {
        btnCopy.textContent = "Copy failed";
        setTimeout(function () { btnCopy.textContent = "Copy"; }, 1500);
      }
    );
  });

  if (btnExportPdf) {
    btnExportPdf.addEventListener("click", function () {
      exportPdf();
    });
  }

  if (btnRefreshSuggestions) {
    btnRefreshSuggestions.addEventListener("click", function () {
      refreshSuggestionsNow();
    });
  }

  loadKeyboardLayout()
    .then(function (layout) {
      keyboardLayout = layout;
      physicalKeyMap = buildPhysicalKeyMap(keyboardLayout);
      renderKeyboard();
    })
    .catch(function () {
      keyboardLayout = buildDefaultKeyboardLayout();
      physicalKeyMap = buildPhysicalKeyMap(keyboardLayout);
      renderKeyboard();
    });

  setupFontSwitcher();
  updateDisplay("");
  scheduleSuggestions(getValue());
})();
