---
name: pdf
description: Work with PDF files: read their text, fill forms, merge and split pages. Use when a task mentions a .pdf file.
license: MIT
metadata:
  version: 1.0
  author: fixture
allowed-tools: Bash(python3:*) Read
---

# PDF skill

Read `forms.md` before filling a form and `reference.md` for the page API.

## Filling a form

Run the bundled script with the input file and the values:

```
scripts/fill_form.py input.pdf name=Ada
```

Current date: !`date`
