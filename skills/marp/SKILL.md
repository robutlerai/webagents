---
name: marp
description: "Use when creating slide decks, presentations, product requirement docs with diagrams, or converting markdown to PDF/PPTX/HTML slides. Triggers on \"make a deck\", \"create slides\", \"presentation\", \"export to PDF\", \"requirement doc with diagrams\", or any Marp-related request."
license: MIT-0
metadata:
  author: "patoo0x"
  source: "https://clawhub.ai/patoo0x/skills/marp"
  version: "1.0.0"
---

# Marp: Markdown Presentation Ecosystem

Create slide decks from plain Markdown. Export to HTML, PDF, PPTX, or images via CLI.

## Quick Start

```bash
# Verify installed
marp --version

# Convert to HTML (default)
marp slides.md -o slides.html

# Convert to PDF (requires Chrome/Edge installed)
marp slides.md --pdf -o slides.pdf

# Convert to PowerPoint
marp slides.md --pptx -o slides.pptx

# Convert to PNG (one image per slide)
marp slides.md --images png

# Watch mode (auto-rebuild on save)
marp -w slides.md

# Serve with live preview
marp -s ./slides-dir
```

## Slide Syntax

Slides are separated by `---` horizontal rulers. Front-matter sets global directives.

```markdown
---
marp: true
theme: default
paginate: true
---

# Title Slide

Subtitle text here

---

## Second Slide

- Bullet points
- Work normally
- Standard **markdown**

---

## Third Slide

Code blocks, tables, images: all standard markdown.
```

**Key rule:** `marp: true` in front-matter is required for Marp CLI to process the file.

## Directives Reference

### Global (whole deck)

| Directive | Values | Purpose |
|-----------|--------|---------|
| `theme` | `default`, `gaia`, `uncover` | Built-in themes |
| `paginate` | `true`/`false` | Page numbers |
| `size` | `16:9` (default), `4:3` | Slide aspect ratio |
| `style` | CSS string | Custom styles for whole deck |
| `headingDivider` | `1`-`6` or array | Auto-split slides at headings |

### Local (per slide, in HTML comments)

```markdown
<!-- backgroundColor: #1a1a2e -->
<!-- color: white -->
<!-- class: lead -->
<!-- _class: invert -->   ← underscore prefix = this slide only (spot directive)
<!-- header: Flash Product Spec -->
<!-- footer: Confidential -->
```

| Directive | Purpose |
|-----------|---------|
| `backgroundColor` | Slide background color |
| `backgroundImage` | Background image URL |
| `color` | Text color |
| `class` | CSS class (persists to following slides) |
| `_class` | CSS class (this slide only) |
| `header` | Header text |
| `footer` | Footer text |
| `paginate` | `true`/`false`/`hold`/`skip` |

## Built-in Themes

### default
GitHub markdown style, content vertically centered. Clean for technical docs.

### gaia
Classic Marp look. Supports `lead` class (centered) and `gaia` class (alternate colors).
```markdown
<!-- _class: lead -->      ← centered title layout
<!-- _class: gaia -->       ← alternate color scheme
<!-- _class: lead invert --> ← dark centered
```

### uncover
Always centered, minimal. Good for talks.

### Common to all themes
```markdown
<!-- class: invert -->  ← dark mode variant
<!-- size: 4:3 -->      ← traditional aspect ratio (default is 16:9)
```

## Image Syntax (Extended)

Marp extends standard markdown image syntax with keywords in alt text.

### Inline images
```markdown
![w:200](image.png)              <!-- width 200px -->
![h:150](image.png)              <!-- height 150px -->
![w:200 h:150](image.png)       <!-- both -->
![blur:3px](image.png)           <!-- CSS filter -->
![grayscale](image.png)          <!-- grayscale filter -->
```

### Background images
```markdown
![bg](image.png)                 <!-- full slide background -->
![bg contain](image.png)        <!-- fit inside slide -->
![bg cover](image.png)          <!-- cover entire slide -->
![bg fit](image.png)            <!-- alias for contain -->
![bg 50%](image.png)            <!-- 50% size -->
![bg right](image.png)          <!-- split: image right, content left -->
![bg left:40%](image.png)       <!-- split: image left 40%, content right -->
```

### Multiple backgrounds
```markdown
![bg](image1.png)
![bg](image2.png)
![bg](image3.png)
<!-- Shows side-by-side -->
```

### Split backgrounds (content + image)
```markdown
![bg right:60%](diagram.png)

## Architecture

- Component A talks to B
- B stores in database
- C reads from cache

<!-- Image takes 60% right, text takes 40% left -->
```

## Custom CSS

### Inline in front-matter
```markdown
---
marp: true
theme: default
style: |
  section {
    font-family: 'Inter', sans-serif;
  }
  section.brand {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }
  h1 { color: #2d3748; }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
---
```

### Using custom classes
```markdown
<!-- _class: brand -->

# Branded Slide

This slide uses the custom `brand` class defined in the style block.
```

### Custom theme file
```bash
# Create theme CSS file, then use with --theme flag
marp slides.md --theme ./custom-theme.css --pdf
```

## Diagrams in Slides

Marp doesn't natively render Mermaid. Render the diagram to SVG first (for example with the Mermaid CLI, if it is installed) and place the SVG as an image:

```bash
mmdc -i diagram.mmd -o diagram.svg
```

```markdown
![bg right:50%](diagram.svg)
```

## Presenter Notes

Add notes below `---` comment blocks: visible in presenter view, hidden in slides.

```markdown
## Slide Title

Content here

<!-- This is a presenter note. Won't appear on slide. -->
<!-- Use marp -s to serve and press P for presenter view. -->
```

## CLI Reference

```bash
# Basic conversion
marp input.md                     # → input.html
marp input.md -o output.pdf       # → PDF
marp input.md --pptx              # → PPTX
marp input.md --images png        # → PNG per slide

# Options
marp --pdf --allow-local-files slides.md   # Allow local image paths in PDF (decks you trust only)
marp --html slides.md                       # Enable HTML tags in markdown (decks you trust only)
marp --theme ./theme.css slides.md          # Custom theme
marp --title "My Deck" slides.md            # Set HTML title

# Batch conversion
marp ./slides-dir/                 # Convert all .md in directory

# Server + watch
marp -s ./slides-dir              # HTTP server with live reload
marp -w slides.md                 # Watch and rebuild on change

# Configuration file (marp.config.mjs)
# Place in project root: auto-detected by CLI
```

### marp.config.mjs example
```javascript
export default {
  allowLocalFiles: true,
  html: true,
  themeSet: './themes',
  output: './output',
};
```

## Patterns for Product Docs

### Requirement doc template
```markdown
---
marp: true
theme: default
paginate: true
header: "Flash: Product Spec"
footer: "Confidential"
style: |
  section { font-size: 24px; }
  section.cover { text-align: center; }
  section.cover h1 { font-size: 48px; }
---

<!-- _class: cover -->

# Feature: Cashu Flashcards

**Version:** 1.0 · **Date:** 2026-03-19 · **Author:** Product Team

---

## Overview

Brief description of what this feature does and why.

---

## User Flow

![bg right:55% contain](./diagrams/user-flow.svg)

1. User opens Flashcard tab
2. Selects amount
3. Confirms creation
4. Receives Cashu token

---

## Architecture

![bg contain](./diagrams/architecture.svg)

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Create flashcard from USD wallet | P0 |
| R2 | Redeem flashcard to any Flash user | P0 |
| R3 | Expire after 30 days | P1 |

---

## Open Questions

- [ ] Max flashcard value?
- [ ] Offline redemption support?
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgot `marp: true` | Add to front-matter: CLI ignores file without it |
| PDF export fails | Needs Chrome/Chromium/Edge installed. Use `--allow-local-files` for local images |
| Mermaid not rendering | Pre-render to SVG with `mmdc`, embed as image |
| Background image not showing | Use `![bg](path)`: the `bg` keyword is required |
| Styles not applying | Check CSS selector: slides are `<section>` elements |
| Directives not working | Must be in HTML comment `<!-- key: value -->` or front-matter |
| Split background wrong side | `![bg right](img)` = image right; `![bg left](img)` = image left |

## Links

- Docs: https://marp.app
- Marpit (framework): https://marpit.marp.app
- CLI: https://github.com/marp-team/marp-cli
- VS Code extension: marp-team.marp-vscode
