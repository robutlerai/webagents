#!/usr/bin/env bash
# pandoc-convert helper script
# Usage: convert.sh <input-file> <output-file> [extra pandoc options...]
#
# Auto-detects formats from file extensions, applies sensible defaults,
# and selects the appropriate PDF engine when needed.

set -euo pipefail

command_exists() {
    command -v "$1" &>/dev/null
}

pdf_engine_command() {
    case "$1" in
        xelatex|pdflatex|lualatex|tectonic|wkhtmltopdf|weasyprint|prince)
            printf '%s\n' "$1"
            ;;
        *)
            return 1
            ;;
    esac
}

pdf_engine_hint() {
    case "$1" in
        xelatex)
            echo "Install XeLaTeX (for example: texlive-xetex)."
            ;;
        pdflatex)
            echo "Install pdfLaTeX (for example: texlive-latex-base)."
            ;;
        lualatex)
            echo "Install LuaLaTeX (for example: texlive-luatex)."
            ;;
        tectonic)
            echo "Install Tectonic."
            ;;
        wkhtmltopdf)
            echo "Install wkhtmltopdf for WebKit-based HTML/CSS PDF rendering."
            ;;
        weasyprint)
            echo "Install WeasyPrint for CSS Paged Media PDF rendering."
            ;;
        prince)
            echo "Install Prince XML for commercial-grade HTML/CSS PDF rendering."
            ;;
        *)
            echo "Install the requested PDF engine or choose another with --pdf-engine=<engine>."
            ;;
    esac
}

latex_pdf_engine() {
    case "$1" in
        xelatex|pdflatex|lualatex|tectonic)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

html_pdf_engine() {
    case "$1" in
        wkhtmltopdf|weasyprint|prince)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

requested_pdf_engine() {
    local previous_is_engine=false

    for arg in "$@"; do
        if [[ "$previous_is_engine" == true ]]; then
            printf '%s\n' "$arg"
            return 0
        fi

        case "$arg" in
            --pdf-engine=*)
                printf '%s\n' "${arg#--pdf-engine=}"
                return 0
                ;;
            --pdf-engine)
                previous_is_engine=true
                ;;
            *)
                previous_is_engine=false
                ;;
        esac
    done

    return 1
}

select_pdf_engine() {
    local input_ext="$1"

    if [[ "$input_ext" == "html" || "$input_ext" == "htm" ]]; then
        for engine in weasyprint wkhtmltopdf prince xelatex lualatex tectonic pdflatex; do
            if command_exists "$(pdf_engine_command "$engine")"; then
                printf '%s\n' "$engine"
                return 0
            fi
        done
    else
        for engine in xelatex lualatex tectonic pdflatex weasyprint wkhtmltopdf prince; do
            if command_exists "$(pdf_engine_command "$engine")"; then
                printf '%s\n' "$engine"
                return 0
            fi
        done
    fi

    return 1
}

show_pdf_recovery_guidance() {
    local engine="$1"

    echo "PDF conversion failed. Recovery options:" >&2

    if [[ -n "$engine" ]]; then
        echo "  - Retry with a different engine via --pdf-engine=<engine>." >&2
        if latex_pdf_engine "$engine"; then
            echo "  - If CSS is important, prefer weasyprint, wkhtmltopdf, or prince." >&2
            echo "  - If Unicode or fonts are failing, prefer xelatex or lualatex." >&2
        elif html_pdf_engine "$engine"; then
            echo "  - If layout is unstable, try xelatex or lualatex instead." >&2
            echo "  - If local assets are missing, add --resource-path=<dir>." >&2
        fi
    fi

    echo "  - Verify fonts, templates, CSS, and image paths referenced by the document." >&2
}

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input-file> <output-file> [pandoc options...]"
    echo ""
    echo "Examples:"
    echo "  $0 report.md report.pdf"
    echo "  $0 page.html output.docx"
    echo "  $0 notes.md notes.html --css=style.css --toc"
    echo "  $0 paper.md paper.pdf --toc -V geometry:margin=1in"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

if ! command_exists pandoc; then
    echo "Error: pandoc is not installed or not on PATH." >&2
    echo "Install pandoc and retry." >&2
    exit 1
fi

# Validate input file exists
if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file '$INPUT' not found."
    exit 1
fi

OUTPUT_DIR="$(dirname "$OUTPUT")"
if [[ "$OUTPUT_DIR" != "." && ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist." >&2
    echo "Create it first or choose a different output path." >&2
    exit 1
fi

# Extract output extension (lowercase)
OUT_EXT="${OUTPUT##*.}"
OUT_EXT="$(echo "$OUT_EXT" | tr '[:upper:]' '[:lower:]')"

# Extract input extension (lowercase)
IN_EXT="${INPUT##*.}"
IN_EXT="$(echo "$IN_EXT" | tr '[:upper:]' '[:lower:]')"

# Build pandoc arguments
PANDOC_ARGS=()
PDF_ENGINE=""
REQUESTED_ENGINE=""
HAS_CSS_ARG=false

for arg in "$@"; do
    if [[ "$arg" == --css=* || "$arg" == "--css" || "$arg" == -c ]]; then
        HAS_CSS_ARG=true
        break
    fi
done

# Always produce standalone output for formats that support it
case "$OUT_EXT" in
    html|html5|htm|epub|epub3)
        PANDOC_ARGS+=("-s")
        ;;
    pdf|docx|odt|pptx|rtf)
        PANDOC_ARGS+=("-s")
        ;;
esac

# PDF engine selection
if [[ "$OUT_EXT" == "pdf" ]]; then
    if REQUESTED_ENGINE="$(requested_pdf_engine "$@")"; then
        PDF_ENGINE="$REQUESTED_ENGINE"

        if engine_command="$(pdf_engine_command "$PDF_ENGINE")"; then
            if ! command_exists "$engine_command"; then
                echo "Error: Requested PDF engine '$PDF_ENGINE' is not installed or not on PATH." >&2
                pdf_engine_hint "$PDF_ENGINE" >&2
                exit 1
            fi
        fi
    else
        if ! PDF_ENGINE="$(select_pdf_engine "$IN_EXT")"; then
            echo "Error: No supported PDF engine found on PATH." >&2
            echo "Install one of: xelatex, lualatex, tectonic, pdflatex, weasyprint, wkhtmltopdf, prince." >&2
            exit 1
        fi

        PANDOC_ARGS+=("--pdf-engine=$PDF_ENGINE")
    fi

    if latex_pdf_engine "$PDF_ENGINE"; then
        # Set sensible default margins if not specified for LaTeX-based engines.
        GEOMETRY_SET=false
        for arg in "$@"; do
            if [[ "$arg" == *geometry* ]]; then
                GEOMETRY_SET=true
                break
            fi
        done
        if [[ "$GEOMETRY_SET" == false ]]; then
            PANDOC_ARGS+=("-V" "geometry:margin=1in")
        fi
    fi

    if [[ "$HAS_CSS_ARG" == true ]] && latex_pdf_engine "$PDF_ENGINE"; then
        echo "Warning: CSS options are usually ignored by LaTeX PDF engines like '$PDF_ENGINE'." >&2
        echo "Consider --pdf-engine=weasyprint, --pdf-engine=wkhtmltopdf, or --pdf-engine=prince for CSS-driven PDF output." >&2
    fi
fi

# For HTML input going to PDF, hint the input format explicitly
# (pandoc sometimes misdetects HTML fragments)
if [[ "$IN_EXT" == "html" || "$IN_EXT" == "htm" ]]; then
    PANDOC_ARGS+=("-f" "html")
fi

# Run pandoc
echo "Converting: $INPUT → $OUTPUT"
if [[ -n "$PDF_ENGINE" ]]; then
    echo "PDF engine: $PDF_ENGINE"
fi
echo "Command: pandoc \"$INPUT\" -o \"$OUTPUT\" ${PANDOC_ARGS[*]} $*"

if ! pandoc "$INPUT" -o "$OUTPUT" "${PANDOC_ARGS[@]}" "$@"; then
    if [[ "$OUT_EXT" == "pdf" ]]; then
        show_pdf_recovery_guidance "$PDF_ENGINE"
    fi
    exit 1
fi

# Verify output was created
if [[ -f "$OUTPUT" ]]; then
    SIZE=$(stat --printf="%s" "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null || echo "unknown")
    echo "Success: Created $OUTPUT ($SIZE bytes)"
else
    echo "Error: Output file was not created."
    exit 1
fi
