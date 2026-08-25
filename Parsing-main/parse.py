"""
STEP 1: PARSING (LOCAL VERSION)
--------------------------------
Input  : first PDF found in ./input/
Output : ./output/parsed/sample_doc.pkl   (docling Document object)
         ./output/parsed/result.md
         ./output/parsed/result.json

Usage:
    python parse.py
    (or import run_parsing() from pipeline.py to chain it with the rest
    of the pipeline)

NOTE: install dependencies first (see requirements.txt):
    pip install -r requirements.txt

LangSmith Integration:
    Set the following environment variables (or add to .env):
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY=<your_langsmith_api_key>
        LANGSMITH_PROJECT=pdf-rag-pipeline
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import pickle
from pathlib import Path

from dotenv import load_dotenv
from langsmith import traceable, trace

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    PictureDescriptionVlmOptions,
    EasyOcrOptions,
)
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Paths
# -----------------------------
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output/parsed")

PICTURE_DESCRIPTION_PROMPT = """
Analyze this image from a PDF document and generate a detailed, factual description.

Instructions:
1. Identify the type of visual (chart, graph, pie chart, table, diagram, flowchart, infographic, photograph, illustration, map, etc.).
2. Describe the overall purpose of the visual.
3. Extract all visible titles, labels, legends, axis names, and important text.
4. If it is a chart or graph:
   - Identify the chart type.
   - Describe the variables being compared.
   - Explain the overall trends and patterns.
   - Mention the highest, lowest, increases, decreases, peaks, and outliers.
   - Mention any numerical values that are clearly visible.
5. If it is a pie chart:
   - Identify each category.
   - Mention percentages or proportions if visible.
   - Describe the largest and smallest segments.
6. If it is a table:
   - Summarize the table.
   - Mention important rows, columns, and values.
7. If it is a diagram or flowchart:
   - Explain the components.
   - Describe relationships and flow between elements.
8. If it is an infographic or illustration:
   - Describe the important visual elements.
   - Explain the main message being communicated.
9. Do not invent or estimate values that are not visible.
10. Produce a concise but information-rich description suitable for semantic search and Retrieval-Augmented Generation (RAG).

Return only the description.
"""


@traceable(run_type="chain", name="find_input_pdf")
def find_input_pdf(input_dir: Path = INPUT_DIR) -> Path:
    """Picks the PDF to parse from the input folder."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder '{input_dir}' does not exist. "
            f"Create it and drop a PDF file inside."
        )

    pdfs = sorted(input_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"No PDF found in '{input_dir}'. Drop a PDF file in there and re-run."
        )

    if len(pdfs) == 1:
        print(f"Found 1 PDF in '{input_dir}': '{pdfs[0].name}'")
        return pdfs[0]

    print(f"\nMultiple PDFs found in '{input_dir}':")
    for idx, pdf in enumerate(pdfs, start=1):
        print(f"  [{idx}] {pdf.name}")

    selection = input(f"\nSelect a PDF to parse (1-{len(pdfs)} or type the filename): ").strip()

    if selection.isdigit() and 1 <= int(selection) <= len(pdfs):
        return pdfs[int(selection) - 1]

    for pdf in pdfs:
        if pdf.name.lower() in (selection.lower(), f"{selection.lower()}.pdf"):
            return pdf

    raise ValueError(
        f"'{selection}' is not a valid index or filename. "
        f"Valid options: {[p.name for p in pdfs]}"
    )


@traceable(run_type="chain", name="build_converter")
def build_converter(enable_vlm: bool = True) -> DocumentConverter:
    """Configures the docling pipeline: OCR, table structure extraction,
    and optional VLM-based picture description."""
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = True
    options.generate_page_images = True
    options.generate_picture_images = True
    options.images_scale = 1.5
    options.ocr_options = EasyOcrOptions(lang=["en"], force_full_page_ocr=False)

    if enable_vlm:
        try:
            options.do_picture_description = True
            options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
                prompt=PICTURE_DESCRIPTION_PROMPT,
                picture_area_threshold=0.05,  # default: 0.05 (5% of page area). Set to 0.0 to describe ALL images.
                generation_config={
                    "max_new_tokens": 256,
                    "do_sample": False,
                },
            )
        except Exception as e:
            print(f"[Notice] Could not attach VLM options ({e}). Continuing with standard layout parsing.")
            options.do_picture_description = False
    else:
        options.do_picture_description = False

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


@traceable(run_type="chain", name="run_parsing")
def run_parsing(pdf_path: Path = None, output_dir: Path = OUTPUT_DIR):
    """Runs the full parsing pipeline on a single PDF and saves the results.
    Includes automatic fallback to standard layout parsing if VLM execution fails."""
    pdf_path = Path(pdf_path) if pdf_path else find_input_pdf()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing '{pdf_path}' ...")

    # --- Build converter and run PDF conversion (traced) ---
    try:
        converter = build_converter(enable_vlm=True)
        with trace(name="pdf_conversion", run_type="chain", inputs={"pdf": str(pdf_path)}):
            result = converter.convert(str(pdf_path))
    except Exception as e:
        print(f"\n[Notice] VLM execution failed ({e}). Switching to standard parsing...")
        converter = build_converter(enable_vlm=False)
        with trace(name="pdf_conversion_fallback", run_type="chain", inputs={"pdf": str(pdf_path)}):
            result = converter.convert(str(pdf_path))

    document = result.document

    # Print picture descriptions for visibility
    for pic in document.pictures:
        if pic.meta and pic.meta.description:
            print(pic.meta.description.text)
            print("---------")

    # --- Save outputs (traced) ---
    with trace(name="save_outputs", run_type="chain", inputs={"output_dir": str(output_dir)}):
        pkl_path = output_dir / "sample_doc.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(document, f)

        document.save_as_markdown(output_dir / "result.md", image_mode=ImageRefMode.REFERENCED)
        document.save_as_json(output_dir / "result.json")

    print(f"[SUCCESS] Saved parsed document to '{pkl_path}'.")
    return document


if __name__ == "__main__":
    run_parsing()
