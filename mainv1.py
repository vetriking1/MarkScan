"""
OMR Sheet Processing System
Complete system for creating OMR templates and processing scanned sheets
"""

import json
import re
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\vetri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)
from pdf2image import convert_from_path
from PyQt5.QtCore import QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QBrush, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyzbar.pyzbar import decode

# ============================================================================
# CORE OMR PROCESSING LOGIC (Separate from GUI)
# ============================================================================


class OMRProcessor:
    """Core logic for OMR processing - independent of PyQt5"""

    def __init__(self, template_data, debug_mode=False, ocr_enabled=True):
        self.template = template_data
        self.debug_mode = debug_mode
        self.debug_images = []
        self.ocr_enabled = ocr_enabled

    def detect_barcode(self, image):
        """Detect and decode barcode from image"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Try to detect barcodes
            barcodes = decode(gray)

            if barcodes:
                # Return the first barcode found
                barcode_data = barcodes[0].data.decode("utf-8")
                if self.debug_mode:
                    print(f"Barcode detected: {barcode_data}")
                return barcode_data

            # If no barcode found in grayscale, try with original image
            if len(image.shape) == 3:
                barcodes = decode(image)
                if barcodes:
                    barcode_data = barcodes[0].data.decode("utf-8")
                    if self.debug_mode:
                        print(f"Barcode detected: {barcode_data}")
                    return barcode_data

            if self.debug_mode:
                print("No barcode detected")
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"Barcode detection error: {str(e)}")
            return None

    def extract_code_with_ocr(self, image):
        """Extract letter + 6-digit code using OCR (e.g., C456712, C205013)"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Focus on left portion of image where code typically appears
            h, w = gray.shape
            left_portion = gray[:, : int(w * 0.3)]  # Left 30% of image

            # Aggressive preprocessing for better OCR
            # 1. Resize to improve OCR accuracy
            scale_factor = 2
            resized = cv2.resize(
                left_portion,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )

            # 2. Apply bilateral filter to reduce noise while keeping edges
            filtered = cv2.bilateralFilter(resized, 9, 75, 75)

            # 3. Multiple thresholding approaches
            _, thresh1 = cv2.threshold(
                filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Invert if background is dark
            if np.mean(thresh1) < 127:
                thresh1 = cv2.bitwise_not(thresh1)

            # 4. Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

            # Try OCR with multiple PSM modes and images
            all_text = []
            psm_modes = [6, 7, 11, 12, 13]  # Different page segmentation modes

            for psm in psm_modes:
                # Try on preprocessed image
                text = pytesseract.image_to_string(
                    morph,
                    config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
                all_text.append(text)

                # Try on resized only
                text2 = pytesseract.image_to_string(
                    resized,
                    config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                )
                all_text.append(text2)

            # Also try on full image with specific config
            full_text = pytesseract.image_to_string(
                gray,
                config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            all_text.append(full_text)

            # Combine all OCR results
            combined_text = "\n".join(all_text)

            if self.debug_mode:
                print(f"OCR Raw Text Sample: {combined_text[:300]}")

            # Use regex to find pattern: 1 letter followed by 6+ digits
            pattern = r"[A-Z]\d{6,}"
            matches = re.findall(pattern, combined_text)

            if matches:
                # Return the most common match
                from collections import Counter

                code = Counter(matches).most_common(1)[0][0]
                if self.debug_mode:
                    print(f"Code extracted: {code} (from {len(matches)} matches)")
                return code

            # If no match, try more lenient pattern (in case C is misread)
            # Look for standalone 6+ digit numbers and check if preceded by letter
            lenient_pattern = r"[A-Z]?\s*\d{6,}"
            lenient_matches = re.findall(lenient_pattern, combined_text)

            if lenient_matches:
                # Clean up matches and add 'C' if missing
                for match in lenient_matches:
                    cleaned = match.strip().replace(" ", "")
                    if cleaned and cleaned[0].isdigit() and len(cleaned) >= 6:
                        # Likely missing the letter, add 'C' as default
                        code = "C" + cleaned
                        if self.debug_mode:
                            print(f"Code extracted with correction: {code}")
                        return code
                    elif len(cleaned) >= 7:
                        if self.debug_mode:
                            print(f"Code extracted: {cleaned}")
                        return cleaned

            if self.debug_mode:
                print("No code pattern found")
                print(f"All text: {combined_text[:200]}")
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"OCR extraction error: {str(e)}")
            return None

    def extract_ocr_field(self, image, ocr_field_data):
        """Extract text from a specific region using OCR and apply regex"""
        try:
            region = ocr_field_data["region"]
            pattern = ocr_field_data["pattern"]
            
            # Extract region from image
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            region_img = image[y:y+h, x:x+w]
            
            if self.debug_mode:
                print(f"Extracting OCR from region: x={x}, y={y}, w={w}, h={h}")
            
            # Preprocess the region for OCR
            if len(region_img.shape) == 3:
                gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = region_img
            
            # Resize for better OCR accuracy
            scale_factor = 2
            resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_CUBIC)
            
            # Bilateral filter to reduce noise
            filtered = cv2.bilateralFilter(resized, 9, 75, 75)
            
            # Multiple thresholding approaches
            _, thresh1 = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if background is dark
            if np.mean(thresh1) < 127:
                thresh1 = cv2.bitwise_not(thresh1)
            
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            
            # Try OCR with multiple PSM modes
            psm_modes = [6, 7, 11, 12, 13]
            all_text = []
            
            for psm in psm_modes:
                text = pytesseract.image_to_string(
                    morph, 
                    config=f'--psm {psm}'
                )
                all_text.append(text)
            
            # Combine results
            combined_text = '\n'.join(all_text)
            
            if self.debug_mode:
                print(f"OCR text from {ocr_field_data.get('name', 'field')}: {combined_text[:200]}")
            
            # Apply regex pattern
            matches = re.findall(pattern, combined_text)
            
            if matches:
                # Return the first match
                result = matches[0]
                if isinstance(result, tuple):
                    result = result[0] if result else ""
                result = str(result).strip()
                
                if self.debug_mode:
                    print(f"OCR matched with pattern '{pattern}': {result}")
                return result
            
            if self.debug_mode:
                print(f"No match found for pattern: {pattern}")
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"OCR field extraction error: {str(e)}")
            return None

    def preprocess_image(self, image):
        """Preprocess the scanned image with multiple enhancement techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Apply multiple thresholding methods
        # Method 1: Otsu's thresholding
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Method 2: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )

        # Method 3: Simple threshold
        _, simple = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)

        if self.debug_mode:
            self.debug_images = {
                "original": image.copy(),
                "gray": gray,
                "enhanced": enhanced,
                "denoised": denoised,
                "otsu": otsu,
                "adaptive": adaptive,
                "simple": simple,
            }

        return otsu, adaptive, denoised

    def detect_filled_bubble(self, otsu_img, adaptive_img, gray_img, x, y, radius):
        """Enhanced bubble detection with multiple methods"""
        # Create circular mask
        h, w = gray_img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            mask, (x, y), int(radius * 0.8), 255, -1
        )  # Use 80% of radius to avoid edges

        # Extract regions
        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0:
            return False, 0.0

        # Method 1: Check darkness in grayscale (most reliable)
        masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
        gray_values = masked_gray[mask > 0]
        avg_intensity = np.mean(gray_values)
        darkness_score = (255 - avg_intensity) / 255.0  # 0=white, 1=black

        # Method 2: Otsu threshold analysis
        masked_otsu = cv2.bitwise_and(otsu_img, otsu_img, mask=mask)
        otsu_values = masked_otsu[mask > 0]
        black_pixels_otsu = np.sum(otsu_values == 0)
        otsu_score = black_pixels_otsu / total_pixels

        # Method 3: Adaptive threshold analysis
        masked_adaptive = cv2.bitwise_and(adaptive_img, adaptive_img, mask=mask)
        adaptive_values = masked_adaptive[mask > 0]
        black_pixels_adaptive = np.sum(adaptive_values == 0)
        adaptive_score = black_pixels_adaptive / total_pixels

        # Method 4: Standard deviation (filled bubbles have lower std)
        std_dev = np.std(gray_values)
        std_score = 1.0 - (std_dev / 128.0)  # Normalize

        # Weighted scoring system
        final_score = (
            darkness_score * 0.4
            + otsu_score * 0.3
            + adaptive_score * 0.2
            + std_score * 0.1
        )

        # Dynamic threshold based on image quality
        threshold = 0.25  # Lower threshold for better detection
        is_filled = final_score > threshold

        if self.debug_mode:
            print(
                f"Bubble at ({x},{y}): dark={darkness_score:.3f}, otsu={otsu_score:.3f}, "
                f"adaptive={adaptive_score:.3f}, std={std_score:.3f}, "
                f"final={final_score:.3f}, filled={is_filled}"
            )

        return is_filled, final_score

    def extract_field_value(self, otsu_img, adaptive_img, gray_img, field_data):
        """Extract value from a multi-digit field with confidence scoring"""
        field_type = field_data["type"]
        bubbles = field_data["bubbles"]

        if field_type == "horizontal":
            # Single row of bubbles (like semester) - 1 row, multiple columns
            max_score = 0
            selected_value = None

            for bubble in bubbles:
                is_filled, score = self.detect_filled_bubble(
                    otsu_img,
                    adaptive_img,
                    gray_img,
                    bubble["x"],
                    bubble["y"],
                    bubble["radius"],
                )
                if is_filled and score > max_score:
                    max_score = score
                    selected_value = bubble["value"]

            return selected_value

        elif field_type == "mcq":
            # MCQ questions - each row is a question, each column is answer option (A, B, C, D)
            rows = field_data["rows"]
            cols = field_data["cols"]
            correct_answers = field_data.get("correct_answers", [])
            
            results = []
            
            for row in range(rows):
                row_bubbles = [b for b in bubbles if b["row"] == row]
                row_bubbles.sort(key=lambda x: x["col"])
                
                max_score = 0
                selected_value = None
                
                for bubble in row_bubbles:
                    is_filled, score = self.detect_filled_bubble(
                        otsu_img,
                        adaptive_img,
                        gray_img,
                        bubble["x"],
                        bubble["y"],
                        bubble["radius"],
                    )
                    if is_filled and score > max_score:
                        max_score = score
                        selected_value = bubble["value"]
                
                question_num = row + 1
                
                if selected_value is not None:
                    is_correct = False
                    if question_num - 1 < len(correct_answers):
                        is_correct = selected_value == correct_answers[question_num - 1]
                    
                    results.append({
                        "question": question_num,
                        "answer": selected_value,
                        "correct": is_correct
                    })
                else:
                    results.append({
                        "question": question_num,
                        "answer": "",
                        "correct": False
                    })
            
            return results

        elif field_type == "grid":
            # Grid of bubbles (like register number, marks, etc.)
            cols = field_data["cols"]
            rows = field_data["rows"]
            result = []

            for col in range(cols):
                col_bubbles = [b for b in bubbles if b["col"] == col]
                col_bubbles.sort(key=lambda x: x["row"])

                max_score = 0
                col_value = None

                for bubble in col_bubbles:
                    is_filled, score = self.detect_filled_bubble(
                        otsu_img,
                        adaptive_img,
                        gray_img,
                        bubble["x"],
                        bubble["y"],
                        bubble["radius"],
                    )
                    if is_filled and score > max_score:
                        max_score = score
                        col_value = bubble["value"]

                if col_value is not None:
                    result.append(str(col_value))
                else:
                    result.append("")

            return "".join(result) if result else None

        return None

    def process_sheet(self, image_path, save_debug=False):
        """Process a single OMR sheet"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        if image is None:
            raise ValueError("Could not load image")

        # Preprocess - now returns multiple processed versions
        otsu, adaptive, gray = self.preprocess_image(image)

        # Save debug images if requested
        if save_debug and self.debug_mode:
            debug_path = Path(image_path).parent / "debug_output"
            debug_path.mkdir(exist_ok=True)

            for name, img in self.debug_images.items():
                if len(img.shape) == 3:
                    cv2.imwrite(str(debug_path / f"{name}.jpg"), img)
                else:
                    cv2.imwrite(str(debug_path / f"{name}.jpg"), img)

        # Draw bubble locations on debug image
        if self.debug_mode:
            debug_img = image.copy()
            for field_name, field_data in self.template["fields"].items():
                for bubble in field_data["bubbles"]:
                    is_filled, score = self.detect_filled_bubble(
                        otsu, adaptive, gray, bubble["x"], bubble["y"], bubble["radius"]
                    )
                    color = (0, 255, 0) if is_filled else (0, 0, 255)
                    cv2.circle(
                        debug_img,
                        (bubble["x"], bubble["y"]),
                        bubble["radius"],
                        color,
                        2,
                    )
                    cv2.putText(
                        debug_img,
                        f"{score:.2f}",
                        (bubble["x"] - 15, bubble["y"] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

            if save_debug:
                debug_path = Path(image_path).parent / "debug_output"
                cv2.imwrite(str(debug_path / "bubbles_detected.jpg"), debug_img)

        # Detect barcode
        barcode_number = self.detect_barcode(image)

        results = {}
        
        # Add OCR fields if enabled
        if self.ocr_enabled and "ocr_fields" in self.template:
            for ocr_field_name, ocr_field_data in self.template["ocr_fields"].items():
                extracted_value = self.extract_ocr_field(image, ocr_field_data)
                results[ocr_field_name] = extracted_value
        
        # Add barcode to results
        results["barcode_number"] = barcode_number
        
        # Extract all OMR fields
        for field_name, field_data in self.template["fields"].items():
            value = self.extract_field_value(otsu, adaptive, gray, field_data)
            results[field_name] = value

        return results

    def process_pdf(self, pdf_path):
        """Process PDF file with multiple sheets"""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            results = []

            for i, img in enumerate(images):
                # Convert PIL image to OpenCV format
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                sheet_result = self.process_sheet(img_bgr)
                sheet_result["sheet_number"] = i + 1
                results.append(sheet_result)

            return results
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def save_to_excel(self, results, output_path):
        """Save results to Excel file"""
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
        return output_path


# ============================================================================
# TEMPLATE CREATOR GUI
# ============================================================================


class TemplateCreator(QWidget):
    """GUI for creating OMR templates"""

    def __init__(self):
        super().__init__()
        self.image = None
        self.template_data = {"fields": {}, "image_size": None}
        self.current_field = None
        self.bubbles = []
        
        # Interactive rectangle state
        self.field_rect = None  # QRect for field area
        self.field_confirmed = False  # Confirmed with Enter
        self.creating_rect = False  # Creating new rectangle with mouse drag
        self.dragging = False
        self.resizing = False
        self.drag_offset = QPoint()
        self.resize_handle_size = 10
        self.resize_corner = None  # 'tl', 'tr', 'bl', 'br' (top-left, etc.)
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()

        load_btn = QPushButton("Load Reference Image")
        load_btn.clicked.connect(self.load_image)
        controls.addWidget(load_btn)

        save_btn = QPushButton("Save Template")
        save_btn.clicked.connect(self.save_template)
        controls.addWidget(save_btn)

        load_template_btn = QPushButton("Load Template")
        load_template_btn.clicked.connect(self.load_template)
        controls.addWidget(load_template_btn)

        layout.addLayout(controls)

        # Field configuration
        field_config = QGroupBox("Add Field")
        field_layout = QGridLayout()

        field_layout.addWidget(QLabel("Field Name:"), 0, 0)
        self.field_name = QLineEdit()
        field_layout.addWidget(self.field_name, 0, 1)

        field_layout.addWidget(QLabel("Type:"), 1, 0)
        self.field_type = QComboBox()
        self.field_type.addItems(["horizontal", "grid", "mcq"])
        self.field_type.currentTextChanged.connect(self.on_type_changed)
        field_layout.addWidget(self.field_type, 1, 1)

        field_layout.addWidget(QLabel("Columns:"), 2, 0)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(1)
        field_layout.addWidget(self.cols_spin, 2, 1)

        field_layout.addWidget(QLabel("Rows:"), 3, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(10)
        field_layout.addWidget(self.rows_spin, 3, 1)

        field_layout.addWidget(QLabel("Bubble Radius:"), 4, 0)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(5, 50)
        self.radius_spin.setValue(15)
        field_layout.addWidget(self.radius_spin, 4, 1)

        field_layout.addWidget(QLabel("Row Gap (pixels):"), 5, 0)
        self.row_gap_spin = QSpinBox()
        self.row_gap_spin.setRange(10, 200)
        self.row_gap_spin.setValue(40)
        field_layout.addWidget(self.row_gap_spin, 5, 1)

        field_layout.addWidget(QLabel("Column Gap (pixels):"), 6, 0)
        self.col_gap_spin = QSpinBox()
        self.col_gap_spin.setRange(10, 200)
        self.col_gap_spin.setValue(50)
        field_layout.addWidget(self.col_gap_spin, 6, 1)

        field_layout.addWidget(QLabel("Correct Answers (MCQ only):"), 7, 0)
        self.correct_answers = QLineEdit()
        self.correct_answers.setPlaceholderText("e.g., A,B,C,D,A,B,C,D (comma-separated)")
        field_layout.addWidget(self.correct_answers, 7, 1)

        start_field_btn = QPushButton("Click First Bubble Position")
        start_field_btn.clicked.connect(self.start_field)
        field_layout.addWidget(start_field_btn, 8, 0, 1, 2)

        clear_field_btn = QPushButton("Clear Current Field")
        clear_field_btn.clicked.connect(self.clear_field)
        field_layout.addWidget(clear_field_btn, 9, 0, 1, 2)

        confirm_rect_btn = QPushButton("Confirm Rectangle (Enter)")
        confirm_rect_btn.clicked.connect(self.confirm_rectangle)
        field_layout.addWidget(confirm_rect_btn, 10, 0, 1, 2)

        field_config.setLayout(field_layout)
        layout.addWidget(field_config)

        # Image display
        self.scroll = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_image_click
        self.image_label.mouseMoveEvent = self.on_image_move
        self.image_label.mouseReleaseEvent = self.on_image_release
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        # Status
        self.status_label = QLabel("Load an image to start")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def keyPressEvent(self, event):
        if event.key() == 16777220 or event.key() == 16777221:  # Return/Enter keys
            if self.current_field is not None and self.field_rect is not None:
                field_type = self.current_field["type"]
                if field_type in ["grid", "horizontal", "mcq"]:
                    # Confirm rectangle (green)
                    field_name = self.current_field["name"]
                    self.save_current_field()
                    self.field_confirmed = True
                    self.update_display()
                    self.status_label.setText(f"Rectangle confirmed (green). Field '{field_name}' created! Rectangle is still draggable/resizable.")
        super().keyPressEvent(event)

    def on_type_changed(self, type_name):
        if type_name == "horizontal":
            self.rows_spin.setValue(1)
            self.rows_spin.setEnabled(False)
            self.cols_spin.setEnabled(True)
            self.correct_answers.setEnabled(False)
        elif type_name == "mcq":
            self.cols_spin.setValue(4)
            self.cols_spin.setEnabled(True)
            self.rows_spin.setEnabled(True)
            self.correct_answers.setEnabled(True)
        else:
            self.rows_spin.setEnabled(True)
            self.cols_spin.setEnabled(True)
            self.correct_answers.setEnabled(False)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image = QPixmap(file_path)
            self.template_data["image_size"] = (self.image.width(), self.image.height())
            self.update_display()
            self.status_label.setText("Image loaded. Configure and add fields.")

    def start_field(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Load an image first!")
            return

        name = self.field_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a field name!")
            return

        field_type = self.field_type.currentText()
        
        self.current_field = {
            "name": name,
            "type": field_type,
            "cols": self.cols_spin.value(),
            "rows": self.rows_spin.value(),
            "radius": self.radius_spin.value(),
            "row_gap": self.row_gap_spin.value(),
            "col_gap": self.col_gap_spin.value(),
            "bubbles": [],
        }
        
        if field_type == "mcq":
            correct_answers = self.correct_answers.text().strip().upper()
            if correct_answers:
                self.current_field["correct_answers"] = [ans.strip() for ans in correct_answers.split(",")]
            else:
                self.current_field["correct_answers"] = []
        
        self.bubbles = []
        self.field_rect = None
        self.field_confirmed = False
        self.creating_rect = False
        
        # For grid, horizontal, and mcq types, set up interactive rectangle
        field_type = self.current_field["type"]
        if field_type in ["grid", "horizontal", "mcq"]:
            self.status_label.setText(
                f"Click and drag to create rectangle area for '{name}', then adjust to fit bubbles. "
                f"Press Enter to confirm (changes color to green and keeps it resizable)."
            )
        else:
            self.status_label.setText(
                f"Click on the FIRST bubble position (top-left) for '{name}'"
            )

    def on_image_click(self, event):
        if self.current_field is None:
            return

        field_type = self.current_field["type"]
        pos = event.pos()

        # Handle rectangle interaction for grid/horizontal/mcq types
        if field_type in ["grid", "horizontal", "mcq"]:
            if self.field_rect is not None:
                # Check if clicking on resize handle
                if self.is_on_resize_handle(pos):
                    self.resizing = True
                    self.resize_corner = self.get_resize_corner(pos)
                # Check if clicking inside rectangle (for dragging)
                elif self.field_rect.contains(pos):
                    self.dragging = True
                    self.drag_offset = pos - self.field_rect.topLeft()
                return
            else:
                # Start creating rectangle
                self.start_x = pos.x()
                self.start_y = pos.y()
                self.field_rect = QRect(pos.x(), pos.y(), 0, 0)
                self.creating_rect = True
                return
        else:
            # Original behavior for other types
            if len(self.bubbles) == 0:
                start_x, start_y = pos.x(), pos.y()
                self.generate_bubbles_from_position(start_x, start_y)
            return

    def on_image_move(self, event):
        if self.current_field is None or self.field_rect is None:
            return

        field_type = self.current_field["type"]
        if field_type not in ["grid", "horizontal", "mcq"]:
            return

        pos = event.pos()

        if self.resizing:
            # Resize rectangle
            rect = self.field_rect
            if self.resize_corner == "br":
                rect.setBottomRight(pos)
            elif self.resize_corner == "bl":
                rect.setBottomLeft(pos)
            elif self.resize_corner == "tr":
                rect.setTopRight(pos)
            elif self.resize_corner == "tl":
                rect.setTopLeft(pos)
            self.field_rect = rect.normalized()
            
        elif self.dragging:
            # Move rectangle
            new_pos = pos - self.drag_offset
            self.field_rect.moveTo(new_pos)
        elif self.creating_rect:
            # Currently creating rectangle - follows mouse
            self.field_rect = QRect(self.start_x, self.start_y, 
                                pos.x() - self.start_x, 
                                pos.y() - self.start_y).normalized()
        else:
            # Not creating, not dragging, not resizing - do nothing
            return

        # Update bubble positions based on rectangle
        if self.field_rect.width() > 50 and self.field_rect.height() > 50:
            self.update_gaps_from_rectangle()
            self.generate_bubbles_from_rect()

        self.update_display()

    def on_image_release(self, event):
        if self.current_field is None:
            return

        field_type = self.current_field["type"]
        if field_type not in ["grid", "horizontal", "mcq"]:
            return

        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        
        if self.creating_rect:
            self.creating_rect = False
            if self.field_rect.width() > 50 and self.field_rect.height() > 50:
                self.update_gaps_from_rectangle()
                self.generate_bubbles_from_rect()
                self.status_label.setText("Rectangle created. Drag to resize/move, or press Enter to confirm (green).")

    def is_on_resize_handle(self, pos):
        if self.field_rect is None or self.creating_rect:
            return False
        
        handle_size = self.resize_handle_size
        corners = {
            "tl": self.field_rect.topLeft(),
            "tr": self.field_rect.topRight(),
            "bl": self.field_rect.bottomLeft(),
            "br": self.field_rect.bottomRight()
        }
        
        for corner_name, corner_pos in corners.items():
            handle_rect = QRect(corner_pos.x() - handle_size//2, 
                              corner_pos.y() - handle_size//2,
                              handle_size, handle_size)
            if handle_rect.contains(pos):
                return True
        
        return False

    def get_resize_corner(self, pos):
        if self.field_rect is None or self.creating_rect:
            return None
        
        handle_size = self.resize_handle_size
        corners = {
            "tl": self.field_rect.topLeft(),
            "tr": self.field_rect.topRight(),
            "bl": self.field_rect.bottomLeft(),
            "br": self.field_rect.bottomRight()
        }
        
        for corner_name, corner_pos in corners.items():
            handle_rect = QRect(corner_pos.x() - handle_size//2, 
                              corner_pos.y() - handle_size//2,
                              handle_size, handle_size)
            if handle_rect.contains(pos):
                return corner_name
        
        return None

    def update_gaps_from_rectangle(self):
        if self.field_rect is None or self.current_field is None:
            return
        
        cols = self.current_field["cols"]
        rows = self.current_field["rows"]
        
        if cols > 1:
            col_gap = self.field_rect.width() // cols
            self.col_gap_spin.setValue(max(10, col_gap))
        
        if rows > 1:
            row_gap = self.field_rect.height() // rows
            self.row_gap_spin.setValue(max(10, row_gap))
        
        # Calculate radius as ~15% of the smaller gap
        min_gap = min(self.col_gap_spin.value(), self.row_gap_spin.value())
        radius = int(min_gap * 0.35)
        radius = max(5, min(radius, 50))
        self.radius_spin.setValue(radius)

    def generate_bubbles_from_position(self, start_x, start_y):
        cols = self.current_field["cols"]
        rows = self.current_field["rows"]
        row_gap = self.current_field["row_gap"]
        col_gap = self.current_field["col_gap"]
        radius = self.current_field["radius"]
        field_type = self.current_field["type"]

        for col in range(cols):
            for row in range(rows):
                x = start_x + (col * col_gap)
                y = start_y + (row * row_gap)

                if field_type == "horizontal":
                    value = col + 1
                elif field_type == "mcq":
                    value = chr(65 + col)
                else:
                    value = row

                bubble = {
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "col": col,
                    "row": row,
                    "value": value,
                }
                self.bubbles.append(bubble)

        self.save_current_field()

    def generate_bubbles_from_rect(self):
        if self.field_rect is None or self.current_field is None:
            return
        
        cols = self.current_field["cols"]
        rows = self.current_field["rows"]
        radius = self.current_field["radius"]
        field_type = self.current_field["type"]
        
        # Calculate positions based on rectangle
        row_gap = self.field_rect.height() // rows if rows > 1 else self.field_rect.height()
        col_gap = self.field_rect.width() // cols if cols > 1 else self.field_rect.width()

        start_x = self.field_rect.x()
        start_y = self.field_rect.y()
        
        self.bubbles = []
        
        for col in range(cols):
            for row in range(rows):
                # Center bubble in its cell
                x = start_x + (col * col_gap) + (col_gap // 2)
                y = start_y + (row * row_gap) + (row_gap // 2)

                if field_type == "horizontal":
                    value = col + 1
                elif field_type == "mcq":
                    value = chr(65 + col)  # A, B, C, D based on column
                else:
                    value = row

                bubble = {
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "col": col,
                    "row": row,
                    "value": value,
                }
                self.bubbles.append(bubble)

    def save_current_field(self):
        self.current_field["bubbles"] = self.bubbles
        field_name = self.current_field["name"]

        field_data = {
            "type": self.current_field["type"],
            "cols": self.current_field["cols"],
            "rows": self.current_field["rows"],
            "bubbles": self.current_field["bubbles"],
        }
        
        if "correct_answers" in self.current_field:
            field_data["correct_answers"] = self.current_field["correct_answers"]
        
        self.template_data["fields"][field_name] = field_data

        self.update_display()
        self.status_label.setText(
            f"Field '{field_name}' created with {len(self.bubbles)} bubbles! Add another field or save template."
        )

        self.current_field = None
        self.bubbles = []
        self.field_rect = None

    def clear_field(self):
        """Clear/Reset current field being created"""
        if self.current_field is not None or self.bubbles or self.field_rect is not None:
            self.current_field = None
            self.bubbles = []
            self.field_rect = None
            self.field_confirmed = False
            self.creating_rect = False
            self.update_display()
            self.status_label.setText(
                "Current field cleared. Configure and start a new field."
            )
        else:
            self.status_label.setText("No active field to clear.")

    def confirm_rectangle(self):
        """Confirm rectangle (changes color to green but keeps it resizable)"""
        if self.current_field is None or self.field_rect is None:
            self.status_label.setText("No active rectangle to confirm.")
            return
        
        field_type = self.current_field["type"]
        if field_type in ["grid", "horizontal", "mcq"]:
            # Save field name before it gets cleared
            field_name = self.current_field["name"]
            self.save_current_field()
            self.field_confirmed = True
            self.status_label.setText(f"Rectangle confirmed (green). Field '{field_name}' created! Rectangle is still draggable/resizable.")

    def finish_field(self):
        # This method is no longer needed since fields auto-save,
        # but keeping it for compatibility
        if self.current_field is None:
            self.status_label.setText("No active field. Start a new field to continue.")
        else:
            self.status_label.setText("Field already saved automatically!")

    def update_display(self):
        if self.image is None:
            return

        pixmap = self.image.copy()
        painter = QPainter(pixmap)

        # Draw saved fields in green
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for field_name, field_data in self.template_data["fields"].items():
            for bubble in field_data["bubbles"]:
                painter.drawEllipse(
                    QPoint(bubble["x"], bubble["y"]), bubble["radius"], bubble["radius"]
                )

        # Draw current field in red
        if self.bubbles:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            for bubble in self.bubbles:
                painter.drawEllipse(
                    QPoint(bubble["x"], bubble["y"]), bubble["radius"], bubble["radius"]
                )

        # Draw interactive rectangle for grid/horizontal/mcq types
        if self.field_rect is not None:
            if self.field_confirmed:
                # Confirmed rectangle - green color
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                painter.setBrush(QBrush(QColor(0, 255, 0, 50)))
            elif self.creating_rect:
                # Creating rectangle - blue color
                painter.setPen(QPen(QColor(0, 0, 255), 2))
                painter.setBrush(QBrush(QColor(0, 0, 255, 30)))
            else:
                # Unconfirmed rectangle - cyan color
                painter.setPen(QPen(QColor(0, 255, 255), 2))
                painter.setBrush(QBrush(QColor(0, 255, 255, 30)))
            
            painter.drawRect(self.field_rect)
            
            # Draw resize handles (only when not confirmed and not creating)
            if not self.field_confirmed and not self.creating_rect:
                handle_size = self.resize_handle_size
                corners = [
                    self.field_rect.topLeft(),
                    self.field_rect.topRight(),
                    self.field_rect.bottomLeft(),
                    self.field_rect.bottomRight()
                ]
                
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                for corner in corners:
                    painter.drawRect(
                        corner.x() - handle_size//2,
                        corner.y() - handle_size//2,
                        handle_size,
                        handle_size
                    )

        painter.end()
        self.image_label.setPixmap(pixmap)

    def save_template(self):
        if not self.template_data["fields"]:
            QMessageBox.warning(self, "Warning", "No fields to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.template_data, f, indent=2)
            QMessageBox.information(self, "Success", "Template saved!")

    def load_template(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "r") as f:
                self.template_data = json.load(f)
            self.update_display()
            self.status_label.setText("Template loaded")


# ============================================================================
# SHEET PROCESSOR GUI
# ============================================================================


class SheetProcessor(QWidget):
    """GUI for processing OMR sheets"""

    def __init__(self):
        super().__init__()
        self.template_path = None
        self.processor = None
        self.results = []
        self.debug_mode = False
        self.ocr_enabled = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()

        load_template_btn = QPushButton("Load Template")
        load_template_btn.clicked.connect(self.load_template)
        controls.addWidget(load_template_btn)

        process_pdf_btn = QPushButton("Process PDF")
        process_pdf_btn.clicked.connect(self.process_pdf)
        controls.addWidget(process_pdf_btn)

        process_images_btn = QPushButton("Process Images")
        process_images_btn.clicked.connect(self.process_images)
        controls.addWidget(process_images_btn)

        # Debug mode checkbox
        from PyQt5.QtWidgets import QCheckBox

        self.debug_checkbox = QCheckBox("Debug Mode (slower)")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug)
        controls.addWidget(self.debug_checkbox)

        # OCR enable checkbox
        self.ocr_checkbox = QCheckBox("Enable OCR")
        self.ocr_checkbox.setChecked(True)
        self.ocr_checkbox.stateChanged.connect(self.toggle_ocr)
        controls.addWidget(self.ocr_checkbox)

        load_ocr_btn = QPushButton("Load OCR Fields")
        load_ocr_btn.clicked.connect(self.load_ocr_fields)
        controls.addWidget(load_ocr_btn)

        export_btn = QPushButton("Export to Excel")
        export_btn.clicked.connect(self.export_excel)
        controls.addWidget(export_btn)

        layout.addLayout(controls)

        # Status
        self.status_label = QLabel("Load a template to start")
        layout.addWidget(self.status_label)

        # Results table
        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.setLayout(layout)

    def toggle_debug(self, state):
        self.debug_mode = state == Qt.Checked
        if self.processor:
            self.processor.debug_mode = self.debug_mode

    def toggle_ocr(self, state):
        self.ocr_enabled = state == Qt.Checked
        if self.processor:
            self.processor.ocr_enabled = self.ocr_enabled

    def load_template(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    template_data = json.load(f)
                self.processor = OMRProcessor(
                    template_data, 
                    debug_mode=self.debug_mode,
                    ocr_enabled=self.ocr_enabled
                )
                self.template_path = file_path
                self.status_label.setText(f"Template loaded: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load template: {str(e)}"
                )

    def load_ocr_fields(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load OCR Fields", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    ocr_data = json.load(f)
                
                if self.processor:
                    self.processor.template["ocr_fields"] = ocr_data.get("ocr_fields", {})
                    self.status_label.setText("OCR fields loaded successfully!")
                else:
                    QMessageBox.warning(
                        self, "Warning", 
                        "Load a template first before loading OCR fields!"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load OCR fields: {str(e)}"
                )

    def process_pdf(self):
        if self.processor is None:
            QMessageBox.warning(self, "Warning", "Load a template first!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", "", "PDF Files (*.pdf)"
        )
        if file_path:
            try:
                self.status_label.setText("Processing PDF...")
                QApplication.processEvents()

                self.results = self.processor.process_pdf(file_path)
                self.display_results()
                self.status_label.setText(f"Processed {len(self.results)} sheets")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Processing failed: {str(e)}\n\n{traceback.format_exc()}",
                )
                self.status_label.setText("Processing failed")

    def process_images(self):
        if self.processor is None:
            QMessageBox.warning(self, "Warning", "Load a template first!")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_paths:
            try:
                self.status_label.setText("Processing images...")
                QApplication.processEvents()

                self.results = []
                for i, path in enumerate(file_paths):
                    result = self.processor.process_sheet(
                        path, save_debug=self.debug_mode
                    )
                    result["sheet_number"] = i + 1
                    result["file_name"] = Path(path).name
                    self.results.append(result)

                self.display_results()
                msg = f"Processed {len(self.results)} images"
                if self.debug_mode:
                    msg += "\nDebug images saved to 'debug_output' folder"
                self.status_label.setText(msg)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Processing failed: {str(e)}\n\n{traceback.format_exc()}",
                )
                self.status_label.setText("Processing failed")

    def display_results(self):
        if not self.results:
            return

        # Get column names
        columns = list(self.results[0].keys())

        self.table.setRowCount(len(self.results))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)

        for i, result in enumerate(self.results):
            for j, col in enumerate(columns):
                value = result.get(col, "")
                
                # Format MCQ results nicely
                if isinstance(value, list):
                    formatted_answers = []
                    correct_count = 0
                    for item in value:
                        if isinstance(item, dict):
                            answer_str = f"Q{item['question']}:{item['answer']}"
                            if item['correct']:
                                answer_str += "(✓)"
                                correct_count += 1
                            else:
                                answer_str += "(✗)"
                            formatted_answers.append(answer_str)
                    
                    score_str = f"Score: {correct_count}/{len(value)}"
                    value = " | ".join(formatted_answers) + " | " + score_str
                else:
                    value = str(value)
                
                self.table.setItem(i, j, QTableWidgetItem(value))

        self.table.resizeColumnsToContents()

    def export_excel(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", "", "Excel Files (*.xlsx)"
        )
        if file_path:
            try:
                self.processor.save_to_excel(self.results, file_path)
                QMessageBox.information(
                    self, "Success", f"Results exported to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


# ============================================================================
# OCR CONFIGURATION GUI
# ============================================================================


class OCRConfigWidget(QWidget):
    """GUI for configuring OCR fields with region selection"""

    def __init__(self):
        super().__init__()
        self.image = None
        self.ocr_fields = []
        self.current_ocr_field = None
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls = QHBoxLayout()

        load_btn = QPushButton("Load Reference Image")
        load_btn.clicked.connect(self.load_image)
        controls.addWidget(load_btn)

        add_ocr_btn = QPushButton("Add OCR Field")
        add_ocr_btn.clicked.connect(self.add_ocr_field)
        controls.addWidget(add_ocr_btn)

        save_btn = QPushButton("Save OCR Fields")
        save_btn.clicked.connect(self.save_ocr_fields)
        controls.addWidget(save_btn)

        load_ocr_btn = QPushButton("Load OCR Fields")
        load_ocr_btn.clicked.connect(self.load_ocr_fields)
        controls.addWidget(load_ocr_btn)

        clear_rect_btn = QPushButton("Clear Current Rectangle")
        clear_rect_btn.clicked.connect(self.clear_current_rect)
        controls.addWidget(clear_rect_btn)

        layout.addLayout(controls)

        # OCR fields list
        self.ocr_fields_list = QListWidget()
        self.ocr_fields_list.currentRowChanged.connect(self.on_ocr_field_selected)
        layout.addWidget(QLabel("OCR Fields:"))
        layout.addWidget(self.ocr_fields_list)

        # OCR field configuration
        ocr_config = QGroupBox("OCR Field Configuration")
        ocr_layout = QGridLayout()

        ocr_layout.addWidget(QLabel("Field Name:"), 0, 0)
        self.ocr_field_name = QLineEdit()
        ocr_layout.addWidget(self.ocr_field_name, 0, 1)

        ocr_layout.addWidget(QLabel("Regex Pattern:"), 1, 0)
        self.ocr_pattern = QLineEdit()
        self.ocr_pattern.setPlaceholderText("e.g., [A-Z]\\d{6,}")
        ocr_layout.addWidget(self.ocr_pattern, 1, 1)

        ocr_layout.addWidget(QLabel("Region (x, y, width, height):"), 2, 0)
        self.ocr_region = QLineEdit()
        self.ocr_region.setReadOnly(True)
        ocr_layout.addWidget(self.ocr_region, 2, 1)

        select_region_btn = QPushButton("Select Region on Image")
        select_region_btn.clicked.connect(self.start_region_selection)
        ocr_layout.addWidget(select_region_btn, 3, 0, 1, 2)

        remove_ocr_btn = QPushButton("Remove OCR Field")
        remove_ocr_btn.clicked.connect(self.remove_ocr_field)
        ocr_layout.addWidget(remove_ocr_btn, 4, 0, 1, 2)

        ocr_config.setLayout(ocr_layout)
        layout.addWidget(ocr_config)

        # Image display
        self.scroll = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_image_press
        self.image_label.mouseMoveEvent = self.on_image_move
        self.image_label.mouseReleaseEvent = self.on_image_release
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        # Status
        self.status_label = QLabel("Load an image to start")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image = QPixmap(file_path)
            self.update_display()
            self.status_label.setText("Image loaded. Add OCR fields and select regions.")

    def add_ocr_field(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Load an image first!")
            return

        name = self.ocr_field_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a field name!")
            return

        pattern = self.ocr_pattern.text().strip()
        if not pattern:
            QMessageBox.warning(self, "Warning", "Enter a regex pattern!")
            return

        ocr_field = {
            "name": name,
            "pattern": pattern,
            "region": None
        }

        self.ocr_fields.append(ocr_field)
        self.ocr_fields_list.addItem(f"{name} - {pattern}")
        self.status_label.setText(f"OCR field '{name}' added. Select region on image.")

        self.ocr_field_name.clear()
        self.ocr_pattern.clear()
        self.ocr_region.clear()

    def on_ocr_field_selected(self, row):
        if 0 <= row < len(self.ocr_fields):
            self.current_ocr_field = self.ocr_fields[row]
            self.ocr_field_name.setText(self.current_ocr_field["name"])
            self.ocr_pattern.setText(self.current_ocr_field["pattern"])
            
            region = self.current_ocr_field.get("region")
            if region:
                self.ocr_region.setText(f"{region['x']}, {region['y']}, {region['width']}, {region['height']}")
            else:
                self.ocr_region.setText("")
            
            self.update_display()

    def remove_ocr_field(self):
        row = self.ocr_fields_list.currentRow()
        if row >= 0:
            del self.ocr_fields[row]
            self.ocr_fields_list.takeItem(row)
            self.current_ocr_field = None
            self.ocr_field_name.clear()
            self.ocr_pattern.clear()
            self.ocr_region.clear()
            self.status_label.setText("OCR field removed.")
            self.update_display()

    def start_region_selection(self):
        if self.current_ocr_field is None:
            QMessageBox.warning(self, "Warning", "Select an OCR field from the list first!")
            return
        
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Load an image first!")
            return
        
        self.status_label.setText("Click and drag on image to select region...")

    def on_image_press(self, event):
        if self.current_ocr_field is None:
            return
        
        self.drawing = True
        self.start_point = event.pos()
        self.current_rect = None

    def on_image_move(self, event):
        if not self.drawing or self.start_point is None:
            return
        
        current_pos = event.pos()
        self.current_rect = QRect(self.start_point, current_pos)
        self.update_display()

    def on_image_release(self, event):
        if not self.drawing or self.start_point is None:
            return
        
        self.drawing = False
        end_pos = event.pos()
        
        rect = QRect(self.start_point, end_pos).normalized()
        
        if rect.width() > 10 and rect.height() > 10:  # Minimum size check
            if self.current_ocr_field is not None:
                self.current_ocr_field["region"] = {
                    "x": rect.x(),
                    "y": rect.y(),
                    "width": rect.width(),
                    "height": rect.height()
                }
                self.ocr_region.setText(f"{rect.x()}, {rect.y()}, {rect.width()}, {rect.height()}")
                self.status_label.setText(f"Region selected: {rect.width()}x{rect.height()} at ({rect.x()}, {rect.y()})")
        else:
            self.status_label.setText("Region too small. Please select a larger area.")
        
        self.start_point = None
        self.update_display()

    def clear_current_rect(self):
        if self.current_ocr_field:
            self.current_ocr_field["region"] = None
            self.ocr_region.clear()
            self.status_label.setText("Region cleared for current OCR field.")
            self.update_display()

    def update_display(self):
        if self.image is None:
            return

        pixmap = self.image.copy()
        painter = QPainter(pixmap)

        # Draw all OCR field regions
        for i, ocr_field in enumerate(self.ocr_fields):
            region = ocr_field.get("region")
            if region:
                x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                
                # Use different color for current field
                if ocr_field == self.current_ocr_field:
                    color = QColor(255, 0, 0)  # Red for current
                else:
                    color = QColor(0, 255, 0)  # Green for others
                
                painter.setPen(QPen(color, 2))
                painter.drawRect(x, y, w, h)
                
                # Draw label
                painter.setPen(QPen(color, 2))
                painter.drawText(x + 5, y + 20, ocr_field["name"])

        # Draw current selection rectangle
        if self.current_rect is not None:
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.drawRect(self.current_rect)

        painter.end()
        self.image_label.setPixmap(pixmap)

    def save_ocr_fields(self):
        if not self.ocr_fields:
            QMessageBox.warning(self, "Warning", "No OCR fields to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save OCR Fields", "", "JSON Files (*.json)"
        )
        if file_path:
            ocr_data = {"ocr_fields": {}}
            for field in self.ocr_fields:
                ocr_data["ocr_fields"][field["name"]] = {
                    "pattern": field["pattern"],
                    "region": field["region"]
                }
            
            with open(file_path, "w") as f:
                json.dump(ocr_data, f, indent=2)
            QMessageBox.information(self, "Success", "OCR fields saved!")

    def load_ocr_fields(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load OCR Fields", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                self.ocr_fields = []
                self.ocr_fields_list.clear()
                
                for name, field_data in data.get("ocr_fields", {}).items():
                    ocr_field = {
                        "name": name,
                        "pattern": field_data["pattern"],
                        "region": field_data["region"]
                    }
                    self.ocr_fields.append(ocr_field)
                    self.ocr_fields_list.addItem(f"{name} - {field_data['pattern']}")
                
                self.status_label.setText("OCR fields loaded.")
                self.update_display()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load OCR fields: {str(e)}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================


class OMRApplication(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("OMR Sheet Processing System")
        self.setGeometry(100, 100, 1200, 800)

        # Create tab widget
        tabs = QTabWidget()

        # Add tabs
        self.template_creator = TemplateCreator()
        self.sheet_processor = SheetProcessor()
        self.ocr_config = OCRConfigWidget()

        tabs.addTab(self.template_creator, "Template Creator")
        tabs.addTab(self.sheet_processor, "Sheet Processor")
        tabs.addTab(self.ocr_config, "OCR Configuration")

        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    window = OMRApplication()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
