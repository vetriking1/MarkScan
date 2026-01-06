# MarkScan OMR

A comprehensive Optical Mark Recognition (OMR) and Optical Character Recognition (OCR) system for processing scanned answer sheets and forms.

## Features

- **Template Creation**: Create custom OMR templates with a visual GUI
- **Interactive Rectangle Positioning**: Drag-and-drop rectangle for grid/horizontal/MCQ fields
  - Automatically calculates row/column gaps and bubble radius
  - Real-time bubble preview
  - Rectangle remains draggable and resizable after confirmation (green state)
- **Multiple Field Types**:
  - **Horizontal**: Single row of bubbles (values 1, 2, 3, ... from left)
  - **Grid**: Multi-row, multi-column (values 0-9 per column)
  - **MCQ**: Multiple choice questions with A, B, C, D options
    - Configurable correct answers (e.g., A,B,C,D,A,B,C,D)
    - Automatic scoring and correctness check
- **Sheet Processing**: Process multiple sheets from images or PDF files
- **Bubble Detection**: Advanced multi-method bubble detection with confidence scoring
- **Barcode Recognition**: Automatic barcode detection and decoding
- **OCR Integration**: Extract text with custom regex patterns
  - User-definable OCR fields
  - Region-based OCR with visual rectangle selection
  - Toggle OCR on/off for performance
  - Support for multiple OCR fields with different regex patterns
- **Batch Processing**: Process multiple sheets at once
- **Excel Export**: Export results directly to Excel files
- **Debug Mode**: Visual debugging with image output for troubleshooting
- **Preprocessing Pipeline**: 
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Denoising
  - Multiple thresholding methods (Otsu, Adaptive, Simple)

## Requirements

### Installation

#### Option 1: Install from requirements.txt
```bash
pip install -r requirements.txt
```

#### Option 2: Install dependencies manually
```bash
pip install opencv-python numpy pandas pyqt5 pytesseract pdf2image pyzbar openpyxl
```

### Tesseract OCR

Tesseract OCR must be installed separately:

- **Windows**: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

Update the Tesseract path in `main.py` (line 16):
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
   - Type: "horizontal", "grid", or "mcq"
   - Number of columns and rows
   - Bubble radius and spacing
   - For MCQ: Enter correct answers (e.g., "A,B,C,D,A,B,C,D")
4. **Interactive Rectangle Positioning** (for grid/horizontal/mcq):
   - Click "Click First Bubble Position"
   - Drag rectangle over the bubble area
   - Adjust corners to resize
   - Drag inside rectangle to move
   - Bubbles appear in real-time
   - Press Enter or click "Confirm Rectangle" to confirm
   - Rectangle turns green but remains draggable/resizable
5. For MCQ fields: Enter correct answers as comma-separated values
6. Repeat for all fields
7. Save template as a JSON file

### Configuring OCR Fields

1. Open the "OCR Configuration" tab
2. Load a reference image
3. Click "Add OCR Field"
4. Configure:
   - Field name (e.g., "student_code", "exam_id")
   - Regex pattern (e.g., `[A-Z]\d{6,}` for letter+6 digits)
5. Click "Select Region on Image"
6. Drag rectangle to select text region
7. Adjust rectangle as needed
8. Save OCR fields to JSON

### Processing Sheets

1. Open the "Sheet Processor" tab
2. Load your template
3. Load OCR fields (optional - can use saved file)
4. Toggle "Enable OCR" checkbox to enable/disable OCR processing
5. Choose processing method:
   - "Process PDF": Process a multi-page PDF file
   - "Process Images": Select multiple image files
6. Enable "Debug Mode" for visual output (creates debug images in a folder)
7. View results in the table
8. Export to Excel for further analysis

## Project Structure

```
OMR/
├── main.py              # Main application with GUI and processing logic
├── requirements.txt       # Python package dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## Architecture

### Core Components

- **OMRProcessor**: Core processing logic (independent of GUI)
  - Image preprocessing
  - Bubble detection with multi-method scoring
  - Barcode detection
  - OCR code extraction with custom fields
  - Field value extraction (horizontal, grid, MCQ)
  - PDF and image processing
  - Excel export

- **TemplateCreator**: GUI for creating OMR templates
  - Visual template editor
  - Interactive rectangle positioning
  - Field configuration
  - Bubble positioning
  - Template save/load

- **SheetProcessor**: GUI for processing sheets
  - Template loading
  - Batch processing
  - OCR field management
  - Results display
  - Debug mode
  - Excel export

- **OCRConfigWidget**: GUI for configuring OCR fields
  - Add/remove OCR fields
  - Visual region selection
  - Regex pattern configuration
  - OCR field save/load

## Bubble Detection Algorithm

The system uses a weighted scoring approach combining multiple detection methods:

1. **Darkness Score** (40% weight): Analyzes grayscale intensity in bubble region
2. **Otsu Score** (30% weight): Black pixel ratio after Otsu thresholding
3. **Adaptive Score** (20% weight): Black pixel ratio after adaptive thresholding
4. **Standard Deviation Score** (10% weight): Measures uniformity (filled bubbles have lower std)

A bubble is considered "filled" if the weighted score exceeds the threshold (default: 0.25).

## MCQ Scoring

For MCQ fields, the system automatically:
- Extracts answers from student sheets
- Compares with correct answers
- Marks each question as correct (✓) or incorrect (✗)
- Calculates total score (e.g., "Score: 7/10")

Results display format: `Q1:A(✓) | Q2:B(✗) | Q3:C(✓) | ... | Score: 7/10`

## OCR Code Extraction

For OCR fields, users can define custom regex patterns:

Example patterns:
- `[A-Z]\d{6,}` - Letter followed by 6+ digits (e.g., C456712)
- `\d{4}` - 4-digit numbers (e.g., 2025)
- `[A-Z]{2,4}` - 2-4 letter codes (e.g., ABC, ABDC)
- `\d{2}/\d{2}/\d{4}` - Date format (e.g., 01/06/2025)

The system:
- Extracts text from specified region
- Applies regex pattern matching
- Returns first matching value
- Processes multiple OCR fields independently

## Rectangle-Based Bubble Positioning

For grid, horizontal, and MCQ fields, bubble positions are determined interactively:

1. **Drag** rectangle to cover bubble area
2. **Resize** using yellow corner handles
3. **Move** by dragging inside rectangle
4. **Auto-calculate** gaps and radius from rectangle dimensions
5. **Press Enter** to confirm (rectangle turns green)
6. **Continue editing** - rectangle stays green and fully interactive

Colors:
- **Blue**: Creating/editing state (with yellow resize handles)
- **Green**: Confirmed state (cleaner look, still editable)
- **Green bubbles**: Saved fields
- **Red bubbles**: Current field being created

## Debug Mode

Enable debug mode to generate intermediate images:

- Original image
- Grayscale version
- Enhanced (CLAHE) version
- Denoised version
- Otsu threshold result
- Adaptive threshold result
- Simple threshold result
- Bubble detection visualization with scores

Debug images are saved to a `debug_output` folder next to the processed file.

## Building Standalone Executable

### Using PyInstaller (Recommended)

#### Step 1: Install PyInstaller
```bash
pip install pyinstaller
```

#### Step 2: Create Icon
- Use .ico file for Windows (preferred)
- Can use .png (will be converted)
- Recommended size: 256x256 or 512x512 pixels

Convert PNG to ICO using Python:
```python
from PIL import Image

img = Image.open('logo.png')
img.save('icon.ico', format='ICO', sizes=[(256,256)])
```

#### Step 3: Build Executable
```bash
cd "C:\Users\vetri\OneDrive\Desktop\Python\scripts\OMR"

# Build with icon and data files
pyinstaller --onefile --windowed --name="MarkScan OMR" --icon=icon.ico --add-data="README.md;." --add-data="LICENSE;." main.py

# Or simpler command
pyinstaller --onefile --windowed --icon=icon.ico main.py
```

#### Command Options:
- `--onefile` or `-F`: Create single .exe file
- `--windowed` or `-w`: No console window (GUI app)
- `--icon=icon.ico`: Set application icon
- `--name="App Name"`: Set output executable name
- `--add-data`: Include data files

#### Step 4: Run from dist folder
The executable will be created in `dist/MarkScan OMR/` folder.

### Final Folder Structure After Build:
```
dist/
└── MarkScan OMR/
    └── MarkScan OMR.exe
```

You can distribute the single .exe file or the entire `MarkScan OMR` folder.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for image processing
- Tesseract OCR for text recognition
- PyQt5 for the GUI framework
- pyzbar for barcode detection
- pdf2image for PDF processing

## Troubleshooting

**Issue**: "Tesseract not found"
- **Solution**: Update the Tesseract path in main.py line 16

**Issue**: Poor bubble detection
- **Solution**: Enable debug mode and adjust threshold values in `detect_filled_bubble` method

**Issue**: OCR accuracy issues
- **Solution**: Ensure image is scanned at 300 DPI or higher with good lighting

**Issue**: PDF processing fails
- **Solution**: Ensure poppler is installed (required by pdf2image):
  - Windows: Download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows)
  - Mac: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`

**Issue**: Antivirus flags .exe as suspicious
- **Solution**: Sign executable or add to vendor exclusion list

**Issue**: .exe file is too large
- **Solution**: Use `--onefile` for smaller file (slower startup), or remove it for folder with many smaller files (faster startup)
