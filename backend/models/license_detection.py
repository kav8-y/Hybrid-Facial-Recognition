import cv2
import numpy as np
import easyocr
import re
import ssl
import os
import pytesseract
from pathlib import Path
import traceback

class LicenseDetectionModel:
    def __init__(self):
        self.reader = None
        print("🔄 License detection model initialized (EasyOCR will load on first use)")
    
    def _load_ocr(self):
        """Lazy load EasyOCR"""
        if self.reader is None:
            print("🔄 Loading EasyOCR (first time, may take 30 seconds)...")
            try:
                # Bypass SSL verification for model downloads (common issue on Windows)
                ssl._create_default_https_context = ssl._create_unverified_context
                self.reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load EasyOCR: {e}")
                raise
    
    def _auto_rotate(self, image):
        """
        Auto-rotate image using Tesseract OSD (Orientation and Script Detection)
        """
        try:
            # Tesseract requires RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use Output.BYTES to avoid UnicodeDecodeError on noisy images that output non-utf8 garbage
            osd_bytes = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.BYTES)
            osd_str = osd_bytes.decode('utf-8', errors='ignore')
            
            angle = int(re.search(r'(?<=Rotate: )\d+', osd_str).group(0))
            
            if angle == 0:
                return image
            elif angle == 90:
                print("🔄 Auto-rotating image -90 degrees")
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                print("🔄 Auto-rotating image 180 degrees")
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                print("🔄 Auto-rotating image +90 degrees")
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return image
        except Exception as e:
            print(f"⚠️ Auto-rotation failed (Tesseract OSD error): {e}")
            return image

    def preprocess_image(self, image_path):
        """
        Advanced preprocessing pipeline for OCR:
        1. Auto-rotate
        2. Resize to standard width
        3. Grayscale
        4. CLAHE Contrast Enhancement
        5. Denoise
        6. Adaptive Thresholding
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # 1. Auto Rotate
            img = self._auto_rotate(img)
            
            # 2. Resize to standard 1280px width to ensure readable text resolution
            target_width = 1280
            height, width = img.shape[:2]
            scale = target_width / width
            new_height = int(height * scale)
            img = cv2.resize(img, (target_width, new_height))
            print(f"🔄 Resized image to {target_width}x{new_height}")
            
            # 3. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(gray)
            
            # 5. Denoise
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, 10, 7, 21)
            
            # 6. Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Save preprocessed image for debugging
            preprocessed_path = str(image_path).replace('.jpg', '_preprocessed.jpg')
            cv2.imwrite(preprocessed_path, binary)
            print(f"💾 Saved preprocessed image: {preprocessed_path}")
            
            # Also save the rotated original for original OCR pass
            rotated_original_path = str(image_path).replace('.jpg', '_rotated.jpg')
            cv2.imwrite(rotated_original_path, img)

            return rotated_original_path, preprocessed_path
            
        except Exception as e:
            print(f"⚠️ Preprocessing warning: {e}")
            return str(image_path), None
    
    def extract_license_info(self, image_path):
        """
        Extract license information using OCR
        Supports Indian, US (real and temporary), and international licenses
        """
        try:
            self._load_ocr()
            
            print(f"🔄 Extracting license info from: {image_path}")
            
            # Preprocess image
            original_path, preprocessed_path = self.preprocess_image(image_path)
            
            # Run OCR on both original and preprocessed
            print("🔄 Running OCR on original image...")
            result_original = self.reader.readtext(original_path)
            
            # Display detected text
            for detection in result_original:
                bbox, text, confidence = detection
                if confidence > 0.3:
                    print(f"   📝 Detected: '{text}' (confidence: {confidence:.2f})")
            
            if preprocessed_path:
                print("🔄 Running OCR on preprocessed image...")
                result_preprocessed = self.reader.readtext(preprocessed_path)
                
                for detection in result_preprocessed:
                    bbox, text, confidence = detection
                    if confidence > 0.3:
                        print(f"   📝 Detected (preprocessed): '{text}' (confidence: {confidence:.2f})")
            else:
                result_preprocessed = []
            
            # Combine results
            all_results = result_original + result_preprocessed
            
            # Extract text
            full_text = " ".join([text.upper() for _, text, conf in all_results if conf > 0.3])
            print(f"📄 Full OCR text: {full_text}")
            
            # Parse the text
            extracted_data = self._parse_license_text(full_text, all_results)
            
            return {
                "success": True,
                "extracted_data": extracted_data,
                "raw_text": full_text
            }
            
        except Exception as e:
            print(f"❌ OCR Error: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "extracted_data": {}
            }
    
    def _parse_license_text(self, full_text, ocr_results):
        """
        Parse OCR text to extract structured data using strict formats.
        Prioritizes DL Number format over fuzzy lookups.
        """
        extracted_data = {
            "license_number": None,
            "name": None,
            "dob": None,
            "address": None,
            "tokens": [] # Store structured tokens for fallback fuzzy search
        }
        
        # Keep raw tokens for advanced scoring
        for detection in ocr_results:
            bbox, text, conf = detection
            if conf > 0.4:
                 extracted_data["tokens"].append({
                     "text": str(text).upper().replace(',','').strip(),
                     "box": bbox,
                     "conf": conf
                 })
        
        # ========== EXTRACT LICENSE NUMBER (STRICT INVARIANT) ==========
        # Indian DL format is strictly APXXYYYYNNNNNNN or AP-XX-YYYY-NNN
        # We look for Two letters followed by at least 11 digits (can have spaces/dashes)
        license_patterns = [
            # Strict Indian DL: Two letters, maybe space/dash, maybe 2 digits, maybe space/dash, 11 digits
            r'\b([A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7})\b',
            r'\b([A-Z]{2}[-\s]?\d{10,13})\b',
            
            # Generic catch-all for 10-18 alphanumeric heavy string
            r'([A-Z0-9\-]{10,20})' 
        ]
        
        for pattern in license_patterns:
            match = re.search(pattern, full_text)
            if match:
                license_num = match.group(1)
                # Normalize: Remove spaces and dashes
                license_num = re.sub(r'[-\s]', '', license_num)
                
                # Validation rule: Must have at least 2 letters and 7 digits
                if len(re.findall(r'[A-Z]', license_num)) >= 2 and len(re.findall(r'\d', license_num)) >= 7:
                    extracted_data["license_number"] = license_num
                    print(f"✅ Extracted DL Number: {extracted_data['license_number']}")
                    break
                    
        if not extracted_data["license_number"]:
             print("⚠️ Structured DL extraction failed.")
             # Fallback: Find the longest contiguous alphanumeric string that matches loose criteria
             candidates = [t["text"] for t in extracted_data["tokens"] if len(t["text"]) >= 11 and sum(c.isdigit() for c in t["text"]) > 5]
             if candidates:
                 extracted_data["license_number"] = re.sub(r'[-\s]', '', candidates[0])
                 print(f"✅ Fallback Extracted DL Number: {extracted_data['license_number']}")


        # ========== EXTRACT DATE OF BIRTH (STRICT REGEX) ==========
        dob_patterns = [
            r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b',
            r'DOB.*?\b(\d{2}[/-]\d{2}[/-]\d{4})\b'
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                dob = match.group(1).replace('/', '-')
                # Basic validation 
                y_match = re.search(r'\d{4}', dob)
                if y_match and 1920 <= int(y_match.group()) <= 2024:
                    extracted_data["dob"] = dob
                    print(f"✅ Extracted DOB: {extracted_data['dob']}")
                    break
        
        # ========== EXTRACT NAME (HEURISTIC) ==========
        # Names are hard on unlabelled Indian DLs. We'll wait for the token-matching engine downstream.
        # But we'll try a basic heuristic: Get the highest confidence large-text token that isn't a known label
        excluded_words = ['UNION', 'INDIA', 'STATE', 'DRIVING', 'LICENCE', 'LICENSE', 'TRANSPORT', 'AUTHORITY', 'DATE', 'ISSUE']
        
        best_name_candidates = []
        for token in extracted_data["tokens"]:
             # Is it mostly alphabetical?
             clean_text = re.sub(r'[^A-Z\s]', '', token["text"])
             if len(clean_text) > 4 and not any(w in clean_text for w in excluded_words):
                 best_name_candidates.append(clean_text)
                 
        if len(best_name_candidates) > 0:
             # Just store the top 3 alphanumeric blocks to help downstream logging
             extracted_data["name"] = " ".join(best_name_candidates[:3])
             
        # Address skipping - not highly reliable for verification
        
        print("✅ OCR extraction complete")
        return extracted_data
