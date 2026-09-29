# Pandoc Supported Formats Reference

## Table of Contents
1. [Input Formats](#input-formats)
2. [Output Formats](#output-formats)
3. [Format-to-Extension Mapping](#format-to-extension-mapping)
4. [PDF Engines](#pdf-engines)
5. [Useful Pandoc Options](#useful-pandoc-options)

---

## Input Formats

| Format | Pandoc name | Typical extensions |
|---|---|---|
| CommonMark | `commonmark` | .md |
| CSV | `csv` | .csv |
| DocBook | `docbook` | .xml, .dbk |
| DOCX (Word) | `docx` | .docx |
| DokuWiki | `dokuwiki` | .txt |
| EPUB | `epub` | .epub |
| FB2 | `fb2` | .fb2 |
| GitHub-Flavored Markdown | `gfm` | .md |
| Haddock | `haddock` | .hs |
| HTML | `html` | .html, .htm |
| Jupyter Notebook | `ipynb` | .ipynb |
| JATS (XML) | `jats` | .xml |
| Jira wiki | `jira` | - |
| JSON (pandoc AST) | `json` | .json |
| LaTeX | `latex` | .tex, .latex |
| Man page | `man` | .1, .man |
| Markdown (pandoc) | `markdown` | .md, .markdown |
| Markdown (strict) | `markdown_strict` | .md |
| Markdown (PHP Extra) | `markdown_phpextra` | .md |
| MediaWiki | `mediawiki` | .wiki |
| Muse | `muse` | .muse |
| Native (pandoc AST) | `native` | .native |
| ODT | `odt` | .odt |
| OPML | `opml` | .opml |
| Org mode | `org` | .org |
| reStructuredText | `rst` | .rst |
| txt2tags | `t2t` | .t2t |
| TikiWiki | `tikiwiki` | - |
| TWiki | `twiki` | - |
| Creole | `creole` | - |

## Output Formats

| Format | Pandoc name | Typical extensions |
|---|---|---|
| AsciiDoc | `asciidoc` / `asciidoctor` | .adoc |
| Beamer (PDF slides) | `beamer` | .pdf |
| CommonMark | `commonmark` | .md |
| ConTeXt | `context` | .ctx |
| DocBook 4/5 | `docbook4` / `docbook5` | .xml |
| DOCX (Word) | `docx` | .docx |
| DokuWiki | `dokuwiki` | .txt |
| DZSlides (HTML slides) | `dzslides` | .html |
| EPUB 2/3 | `epub2` / `epub3` | .epub |
| FB2 | `fb2` | .fb2 |
| GitHub-Flavored Markdown | `gfm` | .md |
| Haddock | `haddock` | .hs |
| HTML 4/5 | `html4` / `html5` | .html |
| ICML (InDesign) | `icml` | .icml |
| JATS (XML) | `jats` | .xml |
| Jira wiki | `jira` | - |
| JSON (pandoc AST) | `json` | .json |
| Jupyter Notebook | `ipynb` | .ipynb |
| LaTeX | `latex` | .tex |
| Man page | `man` | .1 |
| Markdown (pandoc) | `markdown` | .md |
| Markdown (strict) | `markdown_strict` | .md |
| MediaWiki | `mediawiki` | .wiki |
| MS PowerPoint | `pptx` | .pptx |
| Muse | `muse` | .muse |
| Native (pandoc AST) | `native` | .native |
| ODT | `odt` | .odt |
| OPML | `opml` | .opml |
| Org mode | `org` | .org |
| PDF | (via a PDF engine) | .pdf |
| Plain text | `plain` | .txt |
| reStructuredText | `rst` | .rst |
| Reveal.js (HTML slides) | `revealjs` | .html |
| RTF | `rtf` | .rtf |
| S5 (HTML slides) | `s5` | .html |
| Slideous (HTML slides) | `slideous` | .html |
| TEI | `tei` | .xml |
| Texinfo | `texinfo` | .texi |
| Textile | `textile` | .textile |
| ZimWiki | `zimwiki` | .txt |

## Format-to-Extension Mapping

When the user provides a filename, map the extension to the pandoc format:

| Extension | Input format | Output format |
|---|---|---|
| .md, .markdown | `markdown` | `markdown` |
| .html, .htm | `html` | `html5` |
| .tex, .latex | `latex` | `latex` |
| .docx | `docx` | `docx` |
| .odt | `odt` | `odt` |
| .epub | `epub` | `epub3` |
| .pdf | - | (via pdf-engine) |
| .pptx | - | `pptx` |
| .rst | `rst` | `rst` |
| .org | `org` | `org` |
| .ipynb | `ipynb` | `ipynb` |
| .csv | `csv` | - |
| .json | `json` | `json` |
| .xml | `docbook` or `jats` | `docbook5` or `jats` |
| .rtf | - | `rtf` |
| .txt | `plain` or `markdown` | `plain` |
| .wiki | `mediawiki` | `mediawiki` |
| .adoc | - | `asciidoctor` |

## PDF Engines

| Engine | Command | Best for | Notes |
|---|---|---|---|
| XeLaTeX | `--pdf-engine=xelatex` | Unicode, custom fonts | Recommended default; supports system fonts via `mainfont` |
| LuaLaTeX | `--pdf-engine=lualatex` | Complex typography | Slower but very flexible |
| Tectonic | `--pdf-engine=tectonic` | Auto-downloading packages | Self-contained TeX engine |
| pdfLaTeX | `--pdf-engine=pdflatex` | Speed, ASCII-heavy docs | Fastest LaTeX engine; limited Unicode |
| WeasyPrint | `--pdf-engine=weasyprint` | CSS-based PDF | Python-based; good CSS Paged Media |
| wkhtmltopdf | `--pdf-engine=wkhtmltopdf` | HTML/CSS rendering | Renders via WebKit; respects CSS |
| Prince | `--pdf-engine=prince` | Professional HTML→PDF | Commercial; excellent CSS support |
 
When using `scripts/convert.sh`, PDF engine auto-selection prefers HTML/CSS engines for HTML input
and LaTeX engines for other document types. Override this explicitly with `--pdf-engine=<engine>`
when needed.

## Useful Pandoc Options

### Document structure
| Option | Description |
|---|---|
| `-s` / `--standalone` | Produce a complete document (not a fragment) |
| `--toc` | Generate table of contents |
| `--toc-depth=N` | Depth of TOC headings (default: 3) |
| `--number-sections` | Number section headings |
| `--shift-heading-level-by=N` | Shift heading levels up or down |

### Metadata
| Option | Description |
|---|---|
| `-M key=value` | Set metadata (title, author, date, lang, etc.) |
| `--metadata-file=FILE` | Read metadata from YAML file |

### Styling and templates
| Option | Description |
|---|---|
| `--css=URL` | Link CSS stylesheet (HTML output) |
| `--template=FILE` | Use custom template |
| `--reference-doc=FILE` | Styling reference for DOCX/ODT/PPTX |
| `-V key=value` | Set template variables |
| `--highlight-style=STYLE` | Code highlighting (pygments, kate, monochrome, etc.) |

### Media and resources
| Option | Description |
|---|---|
| `--self-contained` | Embed all resources (images, CSS) as data URIs |
| `--extract-media=DIR` | Extract images to directory |
| `--resource-path=DIRS` | Search path for images and resources |
| `--epub-cover-image=FILE` | Cover image for EPUB output |

### Formatting control
| Option | Description |
|---|---|
| `--wrap=auto|none|preserve` | Text wrapping mode |
| `--columns=N` | Line width for wrapping |
| `--tab-stop=N` | Tab stop width |
| `--strip-comments` | Remove HTML comments |
| `--no-highlight` | Disable syntax highlighting |

### Filters
| Option | Description |
|---|---|
| `--filter=PROG` | JSON filter program |
| `--lua-filter=FILE` | Lua filter script |
| `--citeproc` | Process citations |
| `--bibliography=FILE` | Bibliography file (BibTeX, etc.) |
| `--csl=FILE` | Citation style file |

### Common LaTeX/PDF variables
| Variable | Description | Example |
|---|---|---|
| `geometry:margin` | Page margins | `-V geometry:margin=1in` |
| `fontsize` | Base font size | `-V fontsize=12pt` |
| `mainfont` | Main document font | `-V mainfont="DejaVu Serif"` |
| `monofont` | Code font | `-V monofont="DejaVu Sans Mono"` |
| `documentclass` | LaTeX document class | `-V documentclass=report` |
| `papersize` | Paper size | `-V papersize=a4` |
| `colorlinks` | Color hyperlinks | `-V colorlinks=true` |
| `linkcolor` | Internal link color | `-V linkcolor=blue` |
| `urlcolor` | URL link color | `-V urlcolor=blue` |
