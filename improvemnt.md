# OMR Processing Performance Improvements

This document summarizes potential optimizations to improve image/PDF processing speed **without reducing accuracy** in the OMR application.

---

## 1. OCR Optimizations

### 1.1 Early-exit in `extract_code_with_ocr`

Current behavior:
- For each sheet, the code runs Tesseract many times with different page segmentation modes (PSMs) on several preprocessed images (`morph`, `resized`, and full `gray`).
- This can result in 10+ Tesseract calls per sheet just to read a single code (e.g., `C456712`).

Possible improvement:
- Keep the list of PSMs ordered from strongest to weakest (e.g. `[6, 7, 11, 12, 13]`).
- After each `pytesseract.image_to_string` call, immediately run the regex to detect the code.
- As soon as a valid match is found, **return early** and skip the remaining PSMs and image variants.

Impact:
- On "easy" sheets (most cases), this stops after 1–2 Tesseract calls while preserving the same recognition logic and accuracy.

### 1.2 Early-exit in `extract_ocr_field`

Current behavior:
- For each OCR field, multiple PSMs are tried on the preprocessed region.

Possible improvement:
- Start with only the 1–2 best PSMs (often 6 and 7 for single-line text).
- After each OCR call, immediately apply the provided regex pattern for that field.
- If there is a match, return the result and stop trying additional PSMs.

Impact:
- Reduces redundant Tesseract calls while still allowing fallback PSMs when the first attempts fail.

### 1.3 Disable OCR when not needed

Current behavior:
- The `OMRProcessor` supports an `ocr_enabled` flag and only processes `ocr_fields` when it is `True`.

Possible improvement:
- In the GUI, expose a clear option (e.g., checkbox) to enable/disable OCR.
- When the user only cares about OMR bubbles and barcodes, run with `ocr_enabled=False`.

Impact:
- Completely avoids Tesseract overhead when OCR is not required.

---

## 2. PDF Conversion Optimizations

Current behavior:
- PDFs are converted to images using:

  ```python
  images = convert_from_path(pdf_path, dpi=300)
  ```

Possible improvements:

### 2.1 Use multi-threaded conversion

- Use the `thread_count` argument to speed up multi-page PDF rendering:

  ```python
  images = convert_from_path(pdf_path, dpi=300, thread_count=4)
  ```

- This parallelizes page rendering and improves throughput on multi-core CPUs without changing image content.

### 2.2 Carefully tuning DPI (optional)

- If scanners produce clean, high-contrast inputs, it may be possible to reduce DPI slightly (e.g. 250) to speed up conversion and downstream processing.
- Any DPI change should be validated on a representative sample to ensure bubble detection and OCR accuracy remain acceptable.

---

## 3. Bubble Detection Optimizations

Current behavior:
- `detect_filled_bubble` creates a full-image mask (`np.zeros((h, w))`) for each bubble.
- It draws a circle on the entire image, then uses `bitwise_and` on full-size images and computes statistics.

Possible improvements:

### 3.1 Use a tight ROI around each bubble

- Instead of allocating a full-size mask, crop a small square ROI around the bubble center `(x, y)` with side length about `2 * radius`.
- Maintain a precomputed circular mask matching this ROI size.
- Apply operations (mean intensity, black pixel ratio, etc.) only within this ROI.

Impact:
- Reduces the number of pixels processed per bubble dramatically.
- Preserves the same decision logic and thresholds, so recognition accuracy stays the same.

---

## 4. Preprocessing (Denoising) Optimizations

Current behavior:
- `preprocess_image` applies CLAHE and then `cv2.fastNlMeansDenoising`, which is relatively expensive for full A4 pages at 300 DPI.

Possible improvements:

### 4.1 Conditional denoising

- Analyze the image to detect whether it is noisy (e.g., via global variance or simple noise metrics).
- Only apply `fastNlMeansDenoising` when the noise level exceeds a threshold.

### 4.2 Fast-mode denoising

- Introduce a "Fast mode" (or "Standard" vs. "High quality"):
  - **High quality mode**: current pipeline with full `fastNlMeansDenoising` for maximum robustness.
  - **Standard mode**: lighter blur such as `cv2.medianBlur(enhanced, 3)` or `cv2.GaussianBlur` to reduce noise cheaply.

Impact:
- Significant speed-up on clean scans.
- Allows the user to choose a trade-off between speed and robustness without fundamentally changing the bubble detection logic.

---

## 5. Parallel Processing of Multiple Sheets

Current behavior:
- `process_pdf` and `process_images` iterate over pages/images one by one in Python.

Possible improvement:
- Use `concurrent.futures.ProcessPoolExecutor` (or similar) to process different pages/images in parallel:
  - Each worker calls `processor.process_sheet(...)` on a separate image.
  - Collect results and preserve page/image order.

Impact:
- Substantial throughput improvement for large batches of sheets.
- Per-sheet logic and accuracy remain the same; only execution order across CPU cores changes.

---

## 6. Operational Recommendations

- **Debug mode**:
  - Keep "Debug Mode (slower)" disabled for production runs to avoid extra drawing and file writes (e.g., debug images in `debug_output`).
- **Tesseract configuration**:
  - Set `pytesseract.pytesseract.tesseract_cmd` once at startup (as already done) and avoid modifying it in per-sheet code.

---

## Suggested Implementation Order

1. Add early-exit logic for OCR in `extract_code_with_ocr` and `extract_ocr_field`.
2. Enable multi-threaded PDF conversion via `thread_count`.
3. Parallelize multi-image / multi-page processing at the sheet level.
4. Optimize `detect_filled_bubble` by using a small ROI and cached circular masks.
5. Add a fast/quality mode toggle for denoising and preprocessing.

These steps focus on improving performance while preserving (or carefully validating) the current recognition accuracy.
