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
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

    def __init__(self, template_data, debug_mode=False):
        self.template = template_data
        self.debug_mode = debug_mode
        self.debug_images = []

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

        # Extract code using OCR (letter + 6 digits)
        ocr_code = self.extract_code_with_ocr(image)

        # Extract all fields
        results = {"barcode_number": barcode_number, "ocr_code": ocr_code}
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
        self.field_type.addItems(["horizontal", "grid"])
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

        start_field_btn = QPushButton("Click First Bubble Position")
        start_field_btn.clicked.connect(self.start_field)
        field_layout.addWidget(start_field_btn, 7, 0, 1, 2)

        clear_field_btn = QPushButton("Clear Current Field")
        clear_field_btn.clicked.connect(self.clear_field)
        field_layout.addWidget(clear_field_btn, 8, 0, 1, 2)

        field_config.setLayout(field_layout)
        layout.addWidget(field_config)

        # Image display
        self.scroll = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_image_click
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        # Status
        self.status_label = QLabel("Load an image to start")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def on_type_changed(self, type_name):
        if type_name == "horizontal":
            self.rows_spin.setValue(1)
            self.rows_spin.setEnabled(False)
            self.cols_spin.setEnabled(True)
        else:
            self.rows_spin.setEnabled(True)

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

        self.current_field = {
            "name": name,
            "type": self.field_type.currentText(),
            "cols": self.cols_spin.value(),
            "rows": self.rows_spin.value(),
            "radius": self.radius_spin.value(),
            "row_gap": self.row_gap_spin.value(),
            "col_gap": self.col_gap_spin.value(),
            "bubbles": [],
        }
        self.bubbles = []
        self.status_label.setText(
            f"Click on the FIRST bubble position (top-left) for '{name}'"
        )

    def on_image_click(self, event):
        if self.current_field is None:
            return

        # If no bubbles added yet, this is the first bubble - generate all others
        if len(self.bubbles) == 0:
            pos = event.pos()
            start_x, start_y = pos.x(), pos.y()

            # Generate all bubbles based on first position and gaps
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

                    # Assign value based on field type
                    if field_type == "horizontal":
                        value = col + 1  # Start from 1 for horizontal
                    else:
                        value = row  # 0-9 values for grid

                    bubble = {
                        "x": x,
                        "y": y,
                        "radius": radius,
                        "col": col,
                        "row": row,
                        "value": value,
                    }
                    self.bubbles.append(bubble)

            # Automatically save the field
            self.current_field["bubbles"] = self.bubbles
            field_name = self.current_field["name"]

            self.template_data["fields"][field_name] = {
                "type": self.current_field["type"],
                "cols": self.current_field["cols"],
                "rows": self.current_field["rows"],
                "bubbles": self.current_field["bubbles"],
            }

            self.update_display()
            self.status_label.setText(
                f"Field '{field_name}' created with {len(self.bubbles)} bubbles! Add another field or save template."
            )

            self.current_field = None
            self.bubbles = []
        else:
            # This shouldn't happen with auto-generation, but keep for safety
            self.status_label.setText(
                "Field already generated. Click 'Clear Current Field' to start over."
            )

    def clear_field(self):
        """Clear the current field being created"""
        if self.current_field is not None or self.bubbles:
            self.current_field = None
            self.bubbles = []
            self.update_display()
            self.status_label.setText(
                "Current field cleared. Configure and start a new field."
            )
        else:
            self.status_label.setText("No active field to clear.")

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

    def load_template(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    template_data = json.load(f)
                self.processor = OMRProcessor(template_data, debug_mode=self.debug_mode)
                self.template_path = file_path
                self.status_label.setText(f"Template loaded: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load template: {str(e)}"
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
                self.table.setItem(i, j, QTableWidgetItem(str(value)))

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

        tabs.addTab(self.template_creator, "Template Creator")
        tabs.addTab(self.sheet_processor, "Sheet Processor")

        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    window = OMRApplication()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
