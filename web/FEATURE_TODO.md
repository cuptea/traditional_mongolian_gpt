# Rich Editor Feature TODO (zmongol_docs)

## Goals
- Keep the autocomplete suggestion API integration path intact.
- Keep keyboard layout editing (`/layout-editor.html`) available.
- Add a hide/show toggle for the virtual keyboard section.
- Upgrade editing UX toward a document-style editor (toolbar, headings, formatting, lists, alignment, undo/redo).
- Preserve PDF download support.

## Implementation checklist
- [x] Add a phased TODO document before implementation.
- [x] Add rich-text toolbar controls (bold/italic/underline, headings, lists, quote, align, undo/redo, clear formatting).
- [x] Add a contenteditable editor surface as the primary editing area.
- [x] Keep hidden plaintext field synchronized for keyboard input, suggestions, and API/PDF payloads.
- [x] Add show/hide virtual keyboard toggle without removing keyboard layout editor link.
- [x] Keep export-to-PDF button and behavior operational.
- [ ] Add future enhancements: collaborative editing, comments, revision history, pagination.

## Polish pass
- Keep control labels concise and consistent.
- Keep defaults keyboard-visible for first-time users.
- Make toolbar responsive and accessible with ARIA labels.
- Ensure suggestions and PDF export read from plain-text value derived from editor content.
