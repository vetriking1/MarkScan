# MarkScan OMR

A comprehensive Optical Mark Recognition (OMR) and Optical Character Recognition (OCR) system for processing scanned answer sheets and forms.

## Features

- **Template Creation**: Create custom OMR templates with a visual GUI
- **Sheet Processing**: Process multiple sheets from images or PDF files
- **Bubble Detection**: Advanced multi-method bubble detection with confidence scoring
- **Barcode Recognition**: Automatic barcode detection and decoding
- **OCR Integration**: Extract alphanumeric codes (letter + 6-digit format) using Tesseract OCR
- **Batch Processing**: Process multiple sheets at once
- **Excel Export**: Export results directly to Excel files
- **Debug Mode**: Visual debugging with image output for troubleshooting
- **Preprocessing Pipeline**: 
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Denoising
  - Multiple thresholding methods (Otsu, Adaptive, Simple)

## Requirements

- Python 3.7+
- PyQt5
- OpenCV (cv2)
- NumPy
- Pandas
- pytesseract
- pdf2image
- pyzbar
- Tesseract OCR (must be installed separately)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OMR
```

2. Install Python dependencies:
```bash
pip install opencv-python numpy pandas pyqt5 pytesseract pdf2image pyzbar openpyxl
```

3. Install Tesseract OCR:
   - **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. Update the Tesseract path in `main.py` (line 16):
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\path\to\tesseract.exe"
```

## Usage

### Running the Application

```bash
python main.py
```

### Creating a Template

1. Open the "Template Creator" tab
2. Load a reference image (blank OMR sheet)
3. Configure field settings:
   - Field name (e.g., "register_number", "semester", "marks")
   - Type: "vertical" for single column, "grid" for multiple columns
   - Number of columns and rows
   - Bubble radius and spacing
4. Click "Click First Bubble Position" and click on the first bubble location
5. Repeat for all fields
6. Save the template as a JSON file

### Processing Sheets

1. Open the "Sheet Processor" tab
2. Load your template
3. Choose processing method:
   - "Process PDF": Process a multi-page PDF file
   - "Process Images": Select multiple image files
4. Enable "Debug Mode" for visual output (creates debug images in a folder)
5. View results in the table
6. Export to Excel for further analysis

## Project Structure

```
OMR/
├── main.py              # Main application with GUI and processing logic
├── LICENSE              # MIT License
└── README.md            # This file
```

## Architecture

### Core Components

- **OMRProcessor**: Core processing logic (independent of GUI)
  - Image preprocessing
  - Bubble detection with multi-method scoring
  - Barcode detection
  - OCR code extraction
  - Field value extraction
  - PDF and image processing
  - Excel export

- **TemplateCreator**: GUI for creating OMR templates
  - Visual template editor
  - Field configuration
  - Bubble positioning
  - Template save/load

- **SheetProcessor**: GUI for processing sheets
  - Template loading
  - Batch processing
  - Results display
  - Debug mode
  - Excel export

## Bubble Detection Algorithm

The system uses a weighted scoring approach combining multiple detection methods:

1. **Darkness Score** (40% weight): Analyzes grayscale intensity in bubble region
2. **Otsu Score** (30% weight): Black pixel ratio after Otsu thresholding
3. **Adaptive Score** (20% weight): Black pixel ratio after adaptive thresholding
4. **Standard Deviation Score** (10% weight): Measures uniformity (filled bubbles have lower std)

A bubble is considered "filled" if the weighted score exceeds the threshold (default: 0.25).

## OCR Code Extraction

The system extracts alphanumeric codes in the format `[A-Z]\d{6,}` (one letter followed by 6+ digits):

- Uses multiple preprocessing techniques (resize, bilateral filter, morphological operations)
- Tries multiple PSM modes (Page Segmentation Modes)
- Applies pattern matching with regex
- Includes fallback for missing letters

## Debug Mode

Enable debug mode to generate intermediate images:
- Original, grayscale, enhanced, denoised versions
- Otsu, adaptive, and simple threshold results
- Bubble detection visualization with scores

Debug images are saved to a `debug_output` folder next to the processed file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for image processing
- Tesseract OCR for text recognition
- PyQt5 for the GUI framework
- pyzbar for barcode detection

## Troubleshooting

**Issue**: "Tesseract not found"
- **Solution**: Update the Tesseract path in main.py line 16

**Issue**: Poor bubble detection
- **Solution**: Enable debug mode and adjust threshold values in `detect_filled_bubble` method

**Issue**: OCR accuracy issues
- **Solution**: Ensure the image is scanned at 300 DPI or higher with good lighting

**Issue**: PDF processing fails
- **Solution**: Ensure poppler is installed (required by pdf2image):
  - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows)
  - Mac: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`
