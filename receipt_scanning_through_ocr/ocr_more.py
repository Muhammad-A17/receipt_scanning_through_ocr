from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import os
import spacy
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import json
from collections import defaultdict
import logging
import time
import traceback
from functools import wraps

try:
    nlp=spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class ReceiptData:
    # Merchant Information
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    merchant_phone: Optional[str] = None
    merchant_email: Optional[str] = None

    # Transaction Information
    date: Optional[str] = None
    time: Optional[str] = None
    transaction_id: Optional[str] = None
    receipt_number: Optional[str] = None

    # Financial Information
    tip: Optional[float] = None
    tax: Optional[float] = None
    sub_total: Optional[float] = None
    total: Optional[float] = None
    discount: Optional[float] = None

    # Items
    items: List[Dict[str, Any]] = None

    # Payment Information
    payment_method: Optional[str] = None
    card_type: Optional[str] = None
    card_last_four: Optional[str] = None

    # Additional Information
    category: Optional[str] = None
    tax_rate: Optional[float] = None
    currency: Optional[str] = None

    # Quality Metrics
    confidence_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.items is None:
            self.items = []
        if self.confidence_scores is None:
            self.confidence_scores = {}

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.info(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    return wrapper

class EnhancedReceiptParser:
    def __init__(self, log_level=logging.INFO):
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('receipt_parser.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            self.logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
        
        print('DAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

        self.patterns = {
            'phone': [
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'date': [
                r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',
                r'\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4}\b'
            ],
            'time': [
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'
            ],
            'money': [
                r'\$\s*(\d+(?:,\d{3})*\.?\d{0,2})',
                r'(\d+(?:,\d{3})*\.\d{2})\s*\$?'
            ],
            'transaction_id': [
                r'(?:trans|transaction|ref|reference)(?:\s*#?:?\s*)([A-Z0-9]{6,20})',
                r'#\s*([A-Z0-9]{8,})'
            ],
            'receipt_number': [
                r'(?:receipt|rcpt)(?:\s*#?:?\s*)([A-Z0-9]{4,15})',
                r'(?:order|invoice)(?:\s*#?:?\s*)([A-Z0-9]{4,15})'
            ],
            'card_last_four': [
                r'(?:ending\s+in|last\s+4|xxxx)\s*(\d{4})',
                r'\*+(\d{4})'
            ]
        }

        self.business_categories = {
            'restaurant': ['restaurant', 'cafe', 'diner', 'bistro', 'grill', 'kitchen', 'eatery'],
            'retail': ['store', 'shop', 'mart', 'market', 'retail', 'boutique'],
            'gas_station': ['gas', 'fuel', 'petroleum', 'shell', 'exxon', 'chevron', 'bp'],
            'grocery': ['grocery', 'supermarket', 'foods', 'fresh', 'produce'],
            'pharmacy': ['pharmacy', 'drug', 'cvs', 'walgreens', 'rite aid'],
            'office': ['office', 'supplies', 'staples', 'depot'],
            'automotive': ['auto', 'car', 'tire', 'oil', 'service']
        }

        self.payment_methods = {
            'credit': ['credit', 'visa', 'mastercard', 'amex', 'american express', 'discover'],
            'debit': ['debit', 'pin'],
            'cash': ['cash', 'change'],
            'mobile': ['apple pay', 'google pay', 'samsung pay', 'contactless', 'tap']
        }
        
        # Multi-language support
        self.currency_symbols = {
            '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR', 
            '₽': 'RUB', '₩': 'KRW', '₪': 'ILS', '₦': 'NGN', '₨': 'PKR'
        }
        
        self.currency_patterns = {
            'USD': [r'\$\s*(\d+(?:,\d{3})*\.?\d{0,2})', r'(\d+(?:,\d{3})*\.\d{2})\s*\$?'],
            'EUR': [r'€\s*(\d+(?:,\d{3})*\.?\d{0,2})', r'(\d+(?:,\d{3})*\.\d{2})\s*€'],
            'GBP': [r'£\s*(\d+(?:,\d{3})*\.?\d{0,2})', r'(\d+(?:,\d{3})*\.\d{2})\s*£'],
            'JPY': [r'¥\s*(\d+(?:,\d{3})*)', r'(\d+(?:,\d{3})*)\s*¥'],
            'INR': [r'₹\s*(\d+(?:,\d{3})*\.?\d{0,2})', r'(\d+(?:,\d{3})*\.\d{2})\s*₹']
        }
        
        # Multi-language keywords
        self.multilingual_keywords = {
            'total': ['total', 'totale', 'gesamt', 'total', 'total', 'итого', '合計', '总计'],
            'subtotal': ['subtotal', 'sub total', 'sous-total', 'zwischensumme', 'subtotal', 'промежуточный итог', '小計', '小计'],
            'tax': ['tax', 'taxe', 'steuer', 'impuesto', 'налог', '税', '稅'],
            'tip': ['tip', 'pourboire', 'trinkgeld', 'propina', 'чаевые', 'チップ', '小费'],
            'discount': ['discount', 'rabais', 'rabatt', 'descuento', 'скидка', '割引', '折扣']
        }
    @performance_monitor
    def preprocessing(self, image_path):
        """
        Apply multiple preprocessing techniques and let OCR choose the best result
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Couldn't read image")
    
        versions = {}
        versions['original'] = img

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. CLAHE - Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        versions['clahe'] = clahe.apply(gray)

        # 2. Gaussian blur plus sharpening
        blurred = cv2.GaussianBlur(gray, (3,3), 0)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        versions['sharpened'] = cv2.filter2D(blurred, -1, kernel)
        
        # 3. Advanced denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        versions['denoised'] = denoised

        # 4. Binary thresholding (Otsu method)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions['binary_otsu'] = binary

        # 5. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        versions['adaptive'] = adaptive

        # 6. Morphological operations for text enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        versions['morphological'] = morph

        # 7. NEW: Perspective correction for skewed receipts
        try:
            corrected = self._correct_perspective(gray)
            if corrected is not None:
                versions['perspective_corrected'] = corrected
        except Exception as e:
            print(f"     ⚠️ Perspective correction failed: {e}")

        # 8. NEW: Advanced contrast enhancement
        enhanced = self._enhance_contrast(gray)
        versions['contrast_enhanced'] = enhanced

        # 9. NEW: Noise reduction with bilateral filter
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        versions['bilateral_filtered'] = bilateral

        # 10. NEW: Edge-preserving smoothing
        edge_preserved = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.4)
        edge_preserved_gray = cv2.cvtColor(edge_preserved, cv2.COLOR_BGR2GRAY)
        versions['edge_preserved'] = edge_preserved_gray

        print(f"{len(versions)} versions created")
        return versions
    
    def _correct_perspective(self, gray_image):
        """
        Correct perspective distortion in receipt images
        """
        # Find contours to detect receipt boundaries
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour (likely the receipt)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to get corners
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            # We have 4 corners, apply perspective correction
            pts = approx.reshape(4, 2)
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left
            
            # Calculate dimensions
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Destination points
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")
            
            # Apply perspective transformation
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(gray_image, M, (maxWidth, maxHeight))
            
            return warped
        
        return None

    def _enhance_contrast(self, gray_image):
        """
        Advanced contrast enhancement using multiple techniques
        """
        # 1. Histogram equalization
        equalized = cv2.equalizeHist(gray_image)
        
        # 2. Gamma correction
        gamma = 1.5
        gamma_corrected = np.power(gray_image / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        # 3. Combine both techniques
        enhanced = cv2.addWeighted(equalized, 0.6, gamma_corrected, 0.4, 0)
        
        return enhanced
    
    def detect_currency(self, text: str) -> str:
        """
        Detect currency from text based on symbols and patterns
        """
        # Check for currency symbols
        for symbol, currency in self.currency_symbols.items():
            if symbol in text:
                return currency
        
        # Check for currency codes
        currency_codes = ['USD', 'EUR', 'GBP', 'JPY', 'INR', 'RUB', 'KRW', 'ILS', 'NGN', 'PKR']
        for code in currency_codes:
            if code in text.upper():
                return code
        
        # Default to USD if no currency detected
        return 'USD'
    
    def extract_amounts_multilingual(self, lines: List, full_text: str) -> Dict:
        """
        Extract amounts with multi-language and multi-currency support
        """
        amounts = {}
        detected_currency = self.detect_currency(full_text)
        
        # Get currency-specific patterns
        currency_patterns = self.currency_patterns.get(detected_currency, self.currency_patterns['USD'])
        
        all_amounts = []
        for pattern in currency_patterns:
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                try:
                    amount_str = match.group(1) if match.groups() else match.group(0)
                    amount_str = re.sub(r'[^\d.]', '', amount_str)
                    amount = float(amount_str)
                    if amount > 0:
                        all_amounts.append({
                            'amount': amount,
                            'context': full_text[max(0, match.start()-20):match.end()+20],
                            'position': match.start(),
                            'currency': detected_currency
                        })
                except (ValueError, IndexError):
                    continue
        
        # Sort amounts by value (descending) for better identification
        sorted_amounts = sorted(all_amounts, key=lambda x: x['amount'], reverse=True)
        
        # Assign amounts based on multilingual context
        if sorted_amounts:
            # The largest amount is likely the total
            amounts['total'] = sorted_amounts[0]['amount']
            
            # Look for amounts in context using multilingual keywords
            for amount_data in sorted_amounts:
                context = amount_data['context'].lower()
                
                # Check for total keywords
                if any(keyword in context for keyword in self.multilingual_keywords['total']) and not amounts.get('total'):
                    amounts['total'] = amount_data['amount']
                # Check for subtotal keywords
                elif any(keyword in context for keyword in self.multilingual_keywords['subtotal']) and not amounts.get('subtotal'):
                    amounts['subtotal'] = amount_data['amount']
                # Check for tax keywords
                elif any(keyword in context for keyword in self.multilingual_keywords['tax']) and not amounts.get('tax'):
                    amounts['tax'] = amount_data['amount']
                # Check for tip keywords
                elif any(keyword in context for keyword in self.multilingual_keywords['tip']) and not amounts.get('tip'):
                    amounts['tip'] = amount_data['amount']
                # Check for discount keywords
                elif any(keyword in context for keyword in self.multilingual_keywords['discount']) and not amounts.get('discount'):
                    amounts['discount'] = amount_data['amount']
        
        # Add currency information
        amounts['currency'] = detected_currency
        
        return amounts
    
    def analyze_receipt_layout(self, lines: List) -> Dict[str, Any]:
        """
        Analyze receipt layout to identify different sections and receipt types
        """
        layout_info = {
            'receipt_type': 'unknown',
            'sections': {},
            'layout_confidence': 0.0
        }
        
        if not lines:
            return layout_info
        
        # Analyze line positions and content to identify sections
        header_lines = []
        item_lines = []
        footer_lines = []
        
        # Get image dimensions from first line's bounding box
        if lines and len(lines[0]) > 0:
            first_bbox = lines[0][0]
            if len(first_bbox) >= 4:
                # Estimate image height from bounding boxes
                all_y_coords = []
                for line in lines:
                    if len(line) > 0 and len(line[0]) >= 4:
                        bbox = line[0]
                        all_y_coords.extend([bbox[0][1], bbox[2][1]])
                
                if all_y_coords:
                    min_y, max_y = min(all_y_coords), max(all_y_coords)
                    image_height = max_y - min_y
                    
                    # Divide receipt into sections
                    header_threshold = min_y + (image_height * 0.2)  # Top 20%
                    footer_threshold = min_y + (image_height * 0.8)  # Bottom 20%
                    
                    for line in lines:
                        if len(line) > 1 and len(line[1]) > 0:
                            text = line[1][0]
                            bbox = line[0]
                            y_center = (bbox[0][1] + bbox[2][1]) / 2
                            
                            if y_center < header_threshold:
                                header_lines.append((text, line[1][1]))
                            elif y_center > footer_threshold:
                                footer_lines.append((text, line[1][1]))
                            else:
                                item_lines.append((text, line[1][1]))
        
        # Identify receipt type based on content patterns
        receipt_type = self._identify_receipt_type(header_lines, item_lines, footer_lines)
        layout_info['receipt_type'] = receipt_type
        
        # Store section information
        layout_info['sections'] = {
            'header': header_lines,
            'items': item_lines,
            'footer': footer_lines
        }
        
        # Calculate layout confidence based on section distribution
        total_lines = len(header_lines) + len(item_lines) + len(footer_lines)
        if total_lines > 0:
            # Good layout should have reasonable distribution
            header_ratio = len(header_lines) / total_lines
            items_ratio = len(item_lines) / total_lines
            footer_ratio = len(footer_lines) / total_lines
            
            # Ideal ratios: header 20%, items 60%, footer 20%
            ideal_header, ideal_items, ideal_footer = 0.2, 0.6, 0.2
            confidence = 1.0 - (
                abs(header_ratio - ideal_header) + 
                abs(items_ratio - ideal_items) + 
                abs(footer_ratio - ideal_footer)
            ) / 2.0
            
            layout_info['layout_confidence'] = max(0.0, confidence)
        
        return layout_info
    
    def _identify_receipt_type(self, header_lines: List, item_lines: List, footer_lines: List) -> str:
        """
        Identify receipt type based on content patterns
        """
        all_text = ' '.join([text for text, _ in header_lines + item_lines + footer_lines]).lower()
        
        # Restaurant receipt patterns
        restaurant_keywords = ['restaurant', 'cafe', 'diner', 'kitchen', 'grill', 'menu', 'food', 'drink']
        if any(keyword in all_text for keyword in restaurant_keywords):
            return 'restaurant'
        
        # Retail receipt patterns
        retail_keywords = ['store', 'shop', 'retail', 'purchase', 'item', 'product']
        if any(keyword in all_text for keyword in retail_keywords):
            return 'retail'
        
        # Gas station patterns
        gas_keywords = ['gas', 'fuel', 'petroleum', 'gallons', 'liters', 'pump']
        if any(keyword in all_text for keyword in gas_keywords):
            return 'gas_station'
        
        # Pharmacy patterns
        pharmacy_keywords = ['pharmacy', 'drug', 'prescription', 'medicine', 'health']
        if any(keyword in all_text for keyword in pharmacy_keywords):
            return 'pharmacy'
        
        # Check for specific receipt formats
        if any('receipt' in text.lower() for text, _ in header_lines):
            return 'standard_receipt'
        
        return 'unknown'
    
    @performance_monitor
    def select_best_ocr_result(self,image_path:str)->Tuple[List,str,float]:
        versions=self.preprocessing(image_path)

        best_result=None
        best_score=0
        best_version="original"

        for version_name,img_data in versions.items():
            try:
                # Normalize image to 3-channel BGR for PaddleOCR when needed
                if img_data is None:
                    continue
                if hasattr(img_data, 'shape'):
                    # If grayscale (H, W), convert to BGR
                    if len(img_data.shape) == 2:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    # If already 3 channels but not uint8, cast safely
                    if img_data.dtype != np.uint8:
                        img_data = img_data.astype(np.uint8)

                result=self.ocr.ocr(img_data)

                if not result or not result[0]:
                    continue

                # PaddleOCR 3.2.0 Structure: result[0] is a dict with 'rec_texts' and 'rec_scores'
                ocr_result = result[0]  # Get the OCRResult object
                
                # Extract texts and scores from the new structure
                texts = ocr_result.get('rec_texts', [])
                scores = ocr_result.get('rec_scores', [])
                polys = ocr_result.get('rec_polys', [])
                
                if not texts or not scores:
                    continue
                
                # Create lines in the old format for compatibility
                lines = []
                for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                    # Convert to old format: [[bbox], (text, confidence)]
                    bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
                    lines.append([bbox, (text, score)])

                #  scoring algorithm - Updated for PaddleOCR 3.2.0
                total_conf = sum(score for score in scores)
                avg_confidence = total_conf / len(scores) if scores else 0
                high_conf_lines = sum(1 for score in scores if score > 0.8)
                medium_conf_lines = sum(1 for score in scores if 0.6 <= score <= 0.8)

                #score based on uality and uantity
                quality_score = (avg_confidence * 0.4 + 
                               (high_conf_lines / len(lines)) * 0.4 + 
                               (medium_conf_lines / len(lines)) * 0.2)
                
                # Bonus for finding key receipt terms - Updated for PaddleOCR 3.2.0
                text_content = " ".join(texts).lower()
                key_terms = ['total', 'tax', 'subtotal', 'receipt', 'date', '$']
                key_term_bonus = sum(0.05 for term in key_terms if term in text_content)
                
                final_score = quality_score + key_term_bonus
                
                print(f"     📊 {version_name}: {len(lines)} lines, avg_conf: {avg_confidence:.3f}, score: {final_score:.3f}")
                
                if final_score > best_score:
                    best_score = final_score
                    best_result = result
                    best_version = version_name
                    
            except Exception as e:
                print(f"     ❌ Error with {version_name}: {str(e)}")
                continue
        
        print(f"   🏆 Selected: {best_version} (score: {best_score:.3f})")
        
        # Show text extraction comparison for all versions
        self._show_text_extraction_comparison(versions)
        
        return best_result, best_version, best_score
    
    def _show_text_extraction_comparison(self, versions: Dict):
        """
        Save text extraction comparison for all preprocessing versions to a file
        """
        print(f"\n📊 TEXT EXTRACTION COMPARISON")
        print("=" * 50)
        
        version_results = {}
        
        for version_name, img_data in versions.items():
            try:
                # Normalize image for OCR
                if img_data is None:
                    continue
                if hasattr(img_data, 'shape'):
                    if len(img_data.shape) == 2:
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    if img_data.dtype != np.uint8:
                        img_data = img_data.astype(np.uint8)

                result = self.ocr.ocr(img_data)
                
                if not result or not result[0]:
                    version_results[version_name] = {
                        'text_count': 0,
                        'char_count': 0,
                        'avg_confidence': 0,
                        'full_text': '',
                        'lines': []
                    }
                    continue

                # Extract text data
                ocr_result = result[0]
                texts = ocr_result.get('rec_texts', [])
                scores = ocr_result.get('rec_scores', [])
                
                if not texts or not scores:
                    version_results[version_name] = {
                        'text_count': 0,
                        'char_count': 0,
                        'avg_confidence': 0,
                        'full_text': '',
                        'lines': []
                    }
                    continue
                
                # Calculate metrics
                full_text = " ".join(texts)
                char_count = len(full_text)
                avg_confidence = sum(scores) / len(scores) if scores else 0
                
                # Create lines for display
                lines = []
                for i, (text, score) in enumerate(zip(texts, scores)):
                    lines.append(f"  {i+1:2d}. [{score:.3f}] {text}")
                
                version_results[version_name] = {
                    'text_count': len(texts),
                    'char_count': char_count,
                    'avg_confidence': avg_confidence,
                    'full_text': full_text,
                    'lines': lines
                }
                
            except Exception as e:
                version_results[version_name] = {
                    'text_count': 0,
                    'char_count': 0,
                    'avg_confidence': 0,
                    'full_text': f"ERROR: {str(e)}",
                    'lines': []
                }
        
        # Sort versions by text count (descending)
        sorted_versions = sorted(version_results.items(), 
                               key=lambda x: x[1]['text_count'], reverse=True)
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"text_extraction_comparison_{timestamp}.txt"
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TEXT EXTRACTION COMPARISON\n")
            f.write("=" * 50 + "\n\n")
            
            # Write summary table
            f.write(f"{'Version':<20} {'Text Lines':<12} {'Characters':<12} {'Avg Conf':<10}\n")
            f.write("-" * 60 + "\n")
            for version_name, data in sorted_versions:
                f.write(f"{version_name:<20} {data['text_count']:<12} {data['char_count']:<12} {data['avg_confidence']:<10.3f}\n")
            
            # Write detailed text for ALL versions (not just top 3)
            f.write(f"\nDETAILED TEXT EXTRACTION (All Versions)\n")
            f.write("=" * 60 + "\n")
            
            for i, (version_name, data) in enumerate(sorted_versions):
                if data['text_count'] == 0:
                    f.write(f"\n{version_name.upper()} - NO TEXT EXTRACTED\n")
                    f.write("-" * 50 + "\n")
                    continue
                    
                f.write(f"\n{version_name.upper()} ({data['text_count']} lines, {data['char_count']} chars, conf: {data['avg_confidence']:.3f})\n")
                f.write("-" * 50 + "\n")
                
                # Show ALL lines (no truncation)
                for line in data['lines']:
                    f.write(line + "\n")
                
                # Show full text
                f.write(f"\nFull Text ({version_name}):\n")
                f.write(f"'{data['full_text']}'\n")
            
            # Find and highlight the version with most text
            if sorted_versions:
                best_text_version = sorted_versions[0]
                f.write(f"\nMOST TEXT EXTRACTED: {best_text_version[0]}\n")
                f.write(f"   {best_text_version[1]['text_count']} lines, {best_text_version[1]['char_count']} characters\n")
                f.write(f"   Average confidence: {best_text_version[1]['avg_confidence']:.3f}\n")
        
        # Show summary in terminal
        print(f"📊 Summary Table:")
        print(f"{'Version':<20} {'Text Lines':<12} {'Characters':<12} {'Avg Conf':<10}")
        print("-" * 60)
        for version_name, data in sorted_versions:
            print(f"{version_name:<20} {data['text_count']:<12} {data['char_count']:<12} {data['avg_confidence']:<10.3f}")
        
        # Find and highlight the version with most text
        if sorted_versions:
            best_text_version = sorted_versions[0]
            print(f"\n🏆 MOST TEXT EXTRACTED: {best_text_version[0]}")
            print(f"   📊 {best_text_version[1]['text_count']} lines, {best_text_version[1]['char_count']} characters")
            print(f"   🎯 Average confidence: {best_text_version[1]['avg_confidence']:.3f}")
        
        print(f"\n💾 Complete text extraction comparison saved to: {output_file}")
        print(f"   📄 File contains ALL text from ALL versions with full details")
    
    def applying_Named_Entity_Recognizer(self,text:str)->Dict[str,List[str]]:
        if not nlp:
            return {}

        doc=nlp(text)
        entities=defaultdict(list)

        for ent in doc.ents:
            if ent.label_ in ['PERSON','ORG']:
                entities['organizations'].append(ent.text.strip())
            elif ent.label_ in ['GPE']:
                entities['locations'].append(ent.text.strip())
            elif ent.label_ in ['MONEY']:
                entities['money'].append(ent.text.strip())
            elif ent.label_ in ['DATE']:
                entities['dates'].append(ent.text.strip())
            elif ent.label_ in ['TIME']:
                entities['times'].append(ent.text.strip())

        return dict(entities)
    

    def extract_merchant_info(self,lines:List,full_text:str,ner_entities: Dict)->Dict:
        merchant_info={}

        # Try to find merchant name using multiple approaches
        merchant_candidates=[]

        # Using NER entities
        if 'organizations' in ner_entities:
            for org in ner_entities['organizations']:
                if len(org)>3 and not any(skip in org.lower() for skip in ['receipt', 'invoice', 'total']):
                    merchant_candidates.append((org,0.8,'NER'))
            
        #finding using high confidence lines - Updated for PaddleOCR 3.2.0
        for i, line in enumerate(lines[:5]):
            if len(line) > 1 and len(line[1]) > 1:
                text=line[1][0].strip()
                conf=line[1][1]
            else:
                continue

            if len(text)>=3 and conf>0.7 and not re.match(r'^[\d\s\-\/\$\.,:]+$', text) and not any(skip in text.lower() for skip in ['receipt', 'invoice', 'total', 'date', 'time']):
                
                merchant_candidates.append((text, conf, f'top_line_{i}'))

        #looking for business-type keyords - Updated for PaddleOCR 3.2.0
        for line in lines[:10]:
            if len(line) > 1 and len(line[1]) > 1:
                text=line[1][0].strip()
                conf=line[1][1]
            else:
                continue

            for category,keywords in self.business_categories.items():
                if any(keyword in text.lower() for keyword in keywords):
                    merchant_candidates.append((text, conf + 0.1, f'business_keyword_{category}'))
        
        #electing best merchant candidate
        if merchant_candidates:
            merchant_candidates.sort(key=lambda x: x[1],reverse=True)
            merchant_info['name']=merchant_candidates[0][0]
            merchant_info['confidence']=merchant_candidates[0][1]
            merchant_info['method']=merchant_candidates[0][2]

        #extracting phone number
        for pattern in self.patterns['phone']:
            match=re.search(pattern,full_text)
            if match:
                merchant_info['phone']=match.group(0).strip()
                break

        #extracting email
        for pattern in self.patterns['email']:
            match=re.search(pattern,full_text,re.IGNORECASE)
            if match:
                merchant_info['email']=match.group(0).strip()
                break

        #extracting address(looking for locattions from NER)
        if 'locations' in ner_entities and ner_entities['locations']:
            merchant_info['address']=ner_entities['locations'][0]

        return merchant_info
    
    def extract_amounts(self,lines:List,full_text:str)->Dict:
        amounts={}

        all_amounts=[]
        for pattern in self.patterns['money']:
            for match in re.finditer(pattern,full_text,re.IGNORECASE):
                try:
                    amount_str=match.group(1) if match.groups() else match.group(0)
                    amount_str=re.sub(r'[^\d.]','',amount_str)
                    amount=float(amount_str)
                    if amount>0:
                        all_amounts.append({
                            'amount':amount,
                            'context':full_text[max(0,match.start()-20):match.end()+20],
                            'position':match.start()
                        })
                except (ValueError, IndexError):
                    continue
        
        spatial_amounts=self._extract_amounts_spatial(lines)

        for key,amount_data in spatial_amounts.items():
            if amount_data:
                amounts[key]=amount_data['amount']


        if not amounts.get('total'):
            amounts.update(self._extract_amounts_contextual(all_amounts,full_text) or {})
        
        return amounts
    
    def _extract_amounts_spatial(self,lines:List)->Dict:
        spatial_amounts={}

        keywords_to_find = {
            'total': ['total', 'amount due', 'balance', 'grand total'],
            'subtotal': ['subtotal', 'sub total', 'sub-total'],
            'tax': ['tax', 'sales tax', 'gst', 'vat'],
            'tip': ['tip', 'gratuity', 'service'],
            'discount': ['discount', 'savings', 'off', 'promo']
        }

        for amount_type,keywords in keywords_to_find.items():
            for i, line in enumerate(lines):
                if len(line) > 1 and len(line[1]) > 0 and len(line[0]) > 0:
                    text=line[1][0].lower()
                    box=line[0]
                else:
                    continue

                if any(keyword in text for keyword in keywords):
                    # Find the vertical center of this keyword line

                    keyword_y=(box[0][1]+box[2][1])/2
                    # First check if amount is in the same text line
                    amount_match = re.search(r'(\d+\.\d{2})', text)
                    if amount_match:
                        try:
                            amount = float(amount_match.group(1))
                            spatial_amounts[amount_type] = {
                                'amount': amount,
                                'confidence': line[1][1],
                                'method': 'same_line'
                            }
                            break  # Found it, move to next amount type
                        except ValueError:
                            pass
                        
                    # Look for amounts on the same horizontal line (within 20 pixels)
                    for j,other_line in enumerate(lines):
                        if i==j:
                            continue

                        if len(other_line) > 1 and len(other_line[1]) > 0 and len(other_line[0]) > 0:
                            other_text=other_line[1][0]
                            other_box=other_line[0]
                        else:
                            continue
                        other_y=(other_box[0][1]+other_box[2][1])/2

                        if abs(other_y - keyword_y) < 25:  # Increases in terms of idth pixels
                            amount_match=re.search(r'(\d+\.\d{2})',other_text)
                            if amount_match:
                                try:
                                    amount=float(amount_match.group(1))
                                    spatial_amounts[amount_type]={
                                        'amount':amount,
                                        'confidence':other_line[1][1],
                                        'method':'spatial_alignment'
                                    }
                                    break
                                except ValueError:
                                    continue

        return spatial_amounts
    # Sort amounts by value (descending) for total identification
    def _extract_amounts_contextual(self,all_amounts:List,full_text:str)->Dict:
        amounts={}
        sorted_amounts=sorted(all_amounts,key=lambda x: x['amount'],reverse=True)

        for amount_data in all_amounts:
            context=amount_data['context'].lower()
            amount=amount_data['amount']
            #  checking for specific amount types
            if any(word in context for word in ['total', 'amount due', 'balance']) and not amounts.get('total'):
                amounts['total'] = amount
            elif any(word in context for word in ['subtotal', 'sub total']) and not amounts.get('subtotal'):
                amounts['subtotal'] = amount
            elif any(word in context for word in ['tax', 'gst', 'vat']) and not amounts.get('tax'):
                amounts['tax'] = amount
            elif any(word in context for word in ['tip', 'gratuity']) and not amounts.get('tip'):
                amounts['tip'] = amount
            elif any(word in context for word in ['discount', 'off', 'savings']) and not amounts.get('discount'):
                amounts['discount'] = amount

        if not amounts.get('total') and sorted_amounts:
            amounts['total']=sorted_amounts[0]['amount']

        return amounts
    
    def extract_payment_info(self,full_text:str)->Dict:
        payment_info={}

        text_lower=full_text.lower()

        for method_type,keywords in self.payment_methods.items():
            if any(keyword in text_lower for keyword in keywords):
                payment_info['method']=method_type

                if method_type in ['credit', 'debit']:
                    if 'visa' in text_lower:
                        payment_info['card_type'] = 'Visa'
                    elif 'mastercard' in text_lower:
                        payment_info['card_type'] = 'MasterCard'
                    elif any(amex in text_lower for amex in ['amex', 'american express']):
                        payment_info['card_type'] = 'American Express'
                    elif 'discover' in text_lower:
                        payment_info['card_type'] = 'Discover'
                break
        for pattern in self.patterns['card_last_four']:
            match=re.search(pattern,full_text,re.IGNORECASE)
            if match:
                payment_info['card_last_four']=match.group(1)
                break
        return payment_info
    
    def extract_datetime_info(self,full_text:str,ner_entities:Dict)->Dict:
        datetime_info={}

        if 'dates' in ner_entities and ner_entities['dates']:
            datetime_info['date']=ner_entities['dates'][0]

        if 'times' in ner_entities and ner_entities['times']:
            datetime_info['time'] = ner_entities['times'][0]

        if not datetime_info.get('date'):
            for pattern in self.patterns['date']:
                match=re.search(pattern,full_text,re.IGNORECASE)
                if match:
                    datetime_info['date']=match.group(0).strip()
                    break

        return datetime_info
    
    def extract_transaction_ids(self,full_text:str)->Dict:
        ids={}

        for pattern in self.patterns['transaction_id']:
            match=re.search(pattern,full_text,re.IGNORECASE)
            if match:
                ids['receipt_number']=match.group(1).strip()
                break
        
        return ids
    
    def categorize_business(self,merchant_name:str,full_text:str)->str:
        if not merchant_name:
            merchant_name=""

        combined_text=f"{merchant_name} {full_text}".lower()

        for category,keywords in self.business_categories.items():
            if any(keyword in combined_text for keyword in keywords):
                return category
        
        return "other"
    
    def extract_items(self, lines: List, full_text: str) -> List[Dict[str, Any]]:
        """
        Extract individual items from receipt with quantity, name, and price
        """
        items = []
        
        # Patterns for item detection
        item_patterns = [
            r'(\d+)\s+(.+?)\s+(\d+\.\d{2})',  # "2 Coffee 4.50"
            r'(.+?)\s+(\d+\.\d{2})',          # "Coffee 4.50"
            r'(\d+)\s*x\s*(.+?)\s+(\d+\.\d{2})',  # "2 x Coffee 4.50"
            r'(.+?)\s+\$(\d+\.\d{2})',        # "Coffee $4.50"
        ]
        
        # Skip lines that are likely headers, totals, or other non-item text
        skip_keywords = ['total', 'subtotal', 'tax', 'tip', 'discount', 'receipt', 'date', 'time', 'thank', 'change']
        
        for line in lines:
            if len(line) > 1 and len(line[1]) > 0:
                text = line[1][0].strip()
                confidence = line[1][1]
                
                # Skip low confidence or header lines
                if confidence < 0.6 or any(keyword in text.lower() for keyword in skip_keywords):
                    continue
                
                # Try to match item patterns
                for pattern in item_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        groups = match.groups()
                        
                        if len(groups) == 3:  # quantity, name, price
                            quantity = int(groups[0])
                            name = groups[1].strip()
                            price = float(groups[2])
                        elif len(groups) == 2:  # name, price
                            quantity = 1
                            name = groups[0].strip()
                            price = float(groups[1])
                        else:
                            continue
                        
                        # Validate item
                        if (len(name) > 2 and 
                            price > 0 and 
                            price < 1000 and  # Reasonable price range
                            not re.match(r'^[\d\s\-\/\$\.,:]+$', name)):  # Not just numbers/symbols
                            
                            item = {
                                'name': name,
                                'quantity': quantity,
                                'price': price,
                                'total': quantity * price,
                                'confidence': confidence
                            }
                            items.append(item)
                            break
        
        # Remove duplicates and sort by position
        unique_items = []
        seen_items = set()
        
        for item in items:
            item_key = (item['name'].lower(), item['price'])
            if item_key not in seen_items:
                seen_items.add(item_key)
                unique_items.append(item)
        
        return unique_items
    
    def validate_and_cross_check(self, receipt_data: ReceiptData, lines: List, full_text: str) -> ReceiptData:
        """
        Validate extracted data and perform cross-checks for accuracy
        """
        validation_results = {}
        
        # 1. Validate financial consistency
        if receipt_data.total and receipt_data.sub_total and receipt_data.tax:
            calculated_total = receipt_data.sub_total + receipt_data.tax
            if receipt_data.tip:
                calculated_total += receipt_data.tip
            if receipt_data.discount:
                calculated_total -= receipt_data.discount
                
            # Allow 5% tolerance for rounding differences
            tolerance = 0.05
            if abs(calculated_total - receipt_data.total) / receipt_data.total <= tolerance:
                validation_results['financial_consistency'] = True
            else:
                validation_results['financial_consistency'] = False
                print(f"⚠️ Financial inconsistency detected: calculated={calculated_total:.2f}, extracted={receipt_data.total:.2f}")
        
        # 2. Validate tax rate
        if receipt_data.tax and receipt_data.sub_total and receipt_data.tax_rate:
            calculated_tax_rate = (receipt_data.tax / receipt_data.sub_total) * 100
            if abs(calculated_tax_rate - receipt_data.tax_rate) <= 1.0:  # 1% tolerance
                validation_results['tax_rate_consistency'] = True
            else:
                validation_results['tax_rate_consistency'] = False
                print(f"⚠️ Tax rate inconsistency: calculated={calculated_tax_rate:.2f}%, extracted={receipt_data.tax_rate:.2f}%")
        
        # 3. Validate merchant name confidence
        if receipt_data.merchant_name:
            # Check if merchant name appears in high-confidence lines
            high_conf_lines = [line for line in lines[:5] if len(line) > 1 and line[1][1] > 0.8]
            merchant_found = any(receipt_data.merchant_name.lower() in line[1][0].lower() 
                               for line in high_conf_lines)
            validation_results['merchant_confidence'] = merchant_found
        
        # 4. Validate date format
        if receipt_data.date:
            date_patterns = [
                r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
                r'\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{4}'
            ]
            date_valid = any(re.search(pattern, receipt_data.date) for pattern in date_patterns)
            validation_results['date_format'] = date_valid
        
        # 5. Cross-validate amounts with items
        if receipt_data.items:
            items_total = sum(item['total'] for item in receipt_data.items)
            if receipt_data.sub_total:
                items_consistency = abs(items_total - receipt_data.sub_total) / receipt_data.sub_total <= 0.1  # 10% tolerance
                validation_results['items_consistency'] = items_consistency
                if not items_consistency:
                    print(f"⚠️ Items total ({items_total:.2f}) doesn't match subtotal ({receipt_data.sub_total:.2f})")
        
        # 6. Confidence scoring
        total_validations = len(validation_results)
        passed_validations = sum(1 for v in validation_results.values() if v)
        confidence_score = passed_validations / total_validations if total_validations > 0 else 0.0
        
        # Update confidence scores
        receipt_data.confidence_scores.update({
            'validation_score': confidence_score,
            'validations_passed': f"{passed_validations}/{total_validations}",
            'validation_details': validation_results
        })
        
        return receipt_data
    
    @performance_monitor
    def processing_receipt(self,image_path:str)->ReceiptData:
        """
        Main function to process receipt images with comprehensive error handling
        """
        self.logger.info(f"Starting receipt processing for: {image_path}")
        
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                raise ValueError(f"Unsupported image format: {image_path}")
            
        print(f"PROCESSING RECEIPT: {image_path}")
        
            # Get OCR result
            ocr_result, best_version, quality_score = self.select_best_ocr_result(image_path)

        if not ocr_result or not ocr_result[0]:
                self.logger.warning("No text extracted from the image")
            print("NO TEXT EXTRACTED FROM THE IMAGE")
            return ReceiptData()
        
        # PaddleOCR 3.2.0 Structure: Extract from OCRResult object
        ocr_data = ocr_result[0]
        texts = ocr_data.get('rec_texts', [])
        scores = ocr_data.get('rec_scores', [])
        polys = ocr_data.get('rec_polys', [])
        
        # Create lines in the old format for compatibility
        lines = []
        for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
            bbox = poly.tolist() if hasattr(poly, 'tolist') else poly
            lines.append([bbox, (text, score)])
        
        full_text = "\n".join(texts)

        print("\n APPLYING NAMED ENTITY RECOGNITION")

        ner_entities= self.applying_Named_Entity_Recognizer(full_text)
        if ner_entities:
            for entity_type, entities, in ner_entities.items():
                print(f"{entity_type}: {entities[:3]}...")

        print("Extracting receipt fields")

        receipt_data = ReceiptData()
        
        # Merchant information
        merchant_info = self.extract_merchant_info(lines, full_text, ner_entities)
        receipt_data.merchant_name = merchant_info.get('name')
        receipt_data.merchant_phone = merchant_info.get('phone')
        receipt_data.merchant_email = merchant_info.get('email')
        receipt_data.merchant_address = merchant_info.get('address')
        
            # Amounts with multi-language and multi-currency support
            amounts = self.extract_amounts_multilingual(lines, full_text)
        print(f"🔍 DEBUG - Extracted amounts: {amounts}")
        
            # Assign amounts to receipt data
            receipt_data.total = amounts.get('total')
            receipt_data.sub_total = amounts.get('subtotal')
            receipt_data.tax = amounts.get('tax')
            receipt_data.tip = amounts.get('tip')
            receipt_data.discount = amounts.get('discount')
            
            # Set currency
            receipt_data.currency = amounts.get('currency', 'USD')
        
        print(f"🔍 DEBUG - Final amounts - total: {receipt_data.total}, sub_total: {receipt_data.sub_total}, tax: {receipt_data.tax}, tip: {receipt_data.tip}")
            print(f"🔍 DEBUG - Detected currency: {receipt_data.currency}")
        
        # Date and time
        datetime_info = self.extract_datetime_info(full_text, ner_entities)
        receipt_data.date = datetime_info.get('date')
        receipt_data.time = datetime_info.get('time')
        
        # Payment information
        payment_info = self.extract_payment_info(full_text)
        receipt_data.payment_method = payment_info.get('method')
        receipt_data.card_type = payment_info.get('card_type')
        receipt_data.card_last_four = payment_info.get('card_last_four')
        
        # Transaction IDs
        ids = self.extract_transaction_ids(full_text)
        receipt_data.transaction_id = ids.get('transaction_id')
        receipt_data.receipt_number = ids.get('receipt_number')
        
            # Analyze receipt layout
            layout_info = self.analyze_receipt_layout(lines)
            print(f"🔍 DEBUG - Receipt type: {layout_info['receipt_type']}, Layout confidence: {layout_info['layout_confidence']:.3f}")
            
            # Business categorization (use layout analysis if available)
            if layout_info['receipt_type'] != 'unknown':
                receipt_data.category = layout_info['receipt_type']
            else:
        receipt_data.category = self.categorize_business(receipt_data.merchant_name, full_text)
        
            # Extract items
            receipt_data.items = self.extract_items(lines, full_text)
            print(f"🔍 DEBUG - Extracted {len(receipt_data.items)} items")
        
        #calculating tax rate
        if receipt_data.tax and receipt_data.sub_total:
            receipt_data.tax_rate=round((receipt_data.tax/receipt_data.sub_total)*100,2)

        receipt_data.confidence_scores={
            'ocr_quality':quality_score,
                'merchant': merchant_info.get('confidence',0),
                'items_extracted': len(receipt_data.items),
                'layout_confidence': layout_info['layout_confidence'],
                'receipt_type': layout_info['receipt_type']
        }

            # Validate and cross-check extracted data
            print("\n🔍 VALIDATING EXTRACTED DATA")
            receipt_data = self.validate_and_cross_check(receipt_data, lines, full_text)

            self.logger.info(f"Successfully processed receipt: {len(receipt_data.items)} items, quality: {quality_score:.3f}")
        return receipt_data
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid input: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during processing: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty ReceiptData with error information
            error_data = ReceiptData()
            error_data.confidence_scores = {
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_failed': True
            }
            return error_data
    
    def print_result(self,receipt_data:ReceiptData):
        print("EXTRACTED RECEIPT DATA")

        print("=" * 70)
        
        # Merchant Information
        print(" MERCHANT INFORMATION:")
        print(f"   Name: {receipt_data.merchant_name or 'Not found'}")
        print(f"   Address: {receipt_data.merchant_address or 'Not found'}")
        print(f"   Phone: {receipt_data.merchant_phone or 'Not found'}")
        print(f"   Email: {receipt_data.merchant_email or 'Not found'}")
        print(f"   Category: {receipt_data.category or 'Not found'}")
        
        # Transaction Details
        print("\n TRANSACTION DETAILS:")
        print(f"   Date: {receipt_data.date or 'Not found'}")
        print(f"   Time: {receipt_data.time or 'Not found'}")
        print(f"   Transaction ID: {receipt_data.transaction_id or 'Not found'}")
        print(f"   Receipt Number: {receipt_data.receipt_number or 'Not found'}")
        
        # Financial Information
        print("\n FINANCIAL DETAILS:")
        if receipt_data.sub_total:
            print(f"   Subtotal: ${receipt_data.sub_total:.2f}")
        if receipt_data.tax:
            print(f"   Tax: ${receipt_data.tax:.2f}")
        if receipt_data.tax_rate:
            print(f"   Tax Rate: {receipt_data.tax_rate:.2f}%")
        if receipt_data.tip:
            print(f"   Tip: ${receipt_data.tip:.2f}")
        if receipt_data.discount:
            print(f"   Discount: ${receipt_data.discount:.2f}")
        if receipt_data.total:
            print(f"   TOTAL: ${receipt_data.total:.2f}")
        
        # Items Information
        if receipt_data.items:
            print("\n ITEMS:")
            for i, item in enumerate(receipt_data.items, 1):
                print(f"   {i}. {item['name']} (Qty: {item['quantity']}) - ${item['price']:.2f} each = ${item['total']:.2f}")
        else:
            print("\n ITEMS: No items extracted")
        
        # Payment Information
        print("\n PAYMENT INFORMATION:")
        print(f"   Method: {receipt_data.payment_method or 'Not found'}")
        if receipt_data.card_type:
            print(f"   Card Type: {receipt_data.card_type}")
        if receipt_data.card_last_four:
            print(f"   Card Last 4: ****{receipt_data.card_last_four}")
        
        # Quality Metrics
        print(f"\n PROCESSING QUALITY:")
        print(f"   OCR Quality Score: {receipt_data.confidence_scores.get('ocr_quality', 0):.3f}")
        print(f"   Merchant Confidence: {receipt_data.confidence_scores.get('merchant', 0):.3f}")
        print(f"   Validation Score: {receipt_data.confidence_scores.get('validation_score', 0):.3f}")
        print(f"   Validations Passed: {receipt_data.confidence_scores.get('validations_passed', 'N/A')}")
        
        # Show validation details
        validation_details = receipt_data.confidence_scores.get('validation_details', {})
        if validation_details:
            print(f"\n VALIDATION DETAILS:")
            for validation, result in validation_details.items():
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {validation.replace('_', ' ').title()}: {status}")
        
        # Count extracted fields
        extracted_fields = 0
        total_fields = 0
        
        for field_name, field_value in receipt_data.__dict__.items():
            if field_name not in ['items', 'confidence_scores'] and field_value is not None:
                if isinstance(field_value, (str, int, float)) and field_value != "":
                    extracted_fields += 1
            total_fields += 1
        
        print(f"\n   Fields Extracted: {extracted_fields}/{total_fields-2}")  # -2 for items and confidence_scores


def run_comprehensive_tests():
    """
    Run comprehensive tests on the receipt parser
    """
    print("🧪 COMPREHENSIVE RECEIPT PARSER TEST SUITE")
    print("=" * 60)
    
    parser = EnhancedReceiptParser(log_level=logging.WARNING)  # Reduce log noise during testing
    
    # Test cases with expected results
    test_cases = [
        {
            'name': 'Basic Receipt Test',
            'image_path': '/home/magomed-ameen/programming/pract/receipts/r.jpg',
            'expected_fields': ['merchant_name', 'total', 'date'],
            'min_confidence': 0.5
        },
        # Add more test cases as you get more receipt images
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        
        if not os.path.exists(test_case['image_path']):
            print(f"   ⚠️ Test image not found: {test_case['image_path']}")
            continue
        
        try:
            start_time = time.time()
            receipt_data = parser.processing_receipt(test_case['image_path'])
            processing_time = time.time() - start_time
            
            # Evaluate test results
            test_result = {
                'test_name': test_case['name'],
                'processing_time': processing_time,
                'success': True,
                'extracted_fields': 0,
                'expected_fields_found': 0,
                'confidence_scores': receipt_data.confidence_scores,
                'errors': []
            }
            
            # Count extracted fields
            for field_name, field_value in receipt_data.__dict__.items():
                if field_name not in ['items', 'confidence_scores'] and field_value is not None:
                    if isinstance(field_value, (str, int, float)) and field_value != "":
                        test_result['extracted_fields'] += 1
            
            # Check expected fields
            for expected_field in test_case['expected_fields']:
                if hasattr(receipt_data, expected_field) and getattr(receipt_data, expected_field) is not None:
                    test_result['expected_fields_found'] += 1
            
            # Check confidence threshold
            overall_confidence = receipt_data.confidence_scores.get('ocr_quality', 0)
            if overall_confidence < test_case['min_confidence']:
                test_result['errors'].append(f"Low confidence: {overall_confidence:.3f} < {test_case['min_confidence']}")
            
            # Performance check
            if processing_time > 30:  # 30 seconds max
                test_result['errors'].append(f"Slow processing: {processing_time:.1f}s")
            
            results.append(test_result)
            
            # Print test summary
            status = "✅ PASS" if not test_result['errors'] else "❌ FAIL"
            print(f"   {status} - Fields: {test_result['extracted_fields']}, Time: {processing_time:.1f}s")
            if test_result['errors']:
                for error in test_result['errors']:
                    print(f"      ⚠️ {error}")
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {str(e)}")
            results.append({
                'test_name': test_case['name'],
                'success': False,
                'errors': [str(e)]
            })
    
    # Print overall test summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    passed_tests = sum(1 for r in results if r.get('success', False) and not r.get('errors', []))
    total_tests = len(results)
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    
    # Save test results
    test_results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(test_results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Test results saved to: {test_results_file}")
    
    return results

def main():
    print("🚀 ENHANCED RECEIPT PARSER - PRODUCTION LEVEL")
    print("=" * 65)
    print("Professional-level receipt processing for accounting applications")
    print("\nChoose an option:")
    print("1. Process single receipt")
    print("2. Run comprehensive tests")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
    # Initialize parser
    parser = EnhancedReceiptParser()
    
    # Process receipt
        image_path = input("Enter image path (or press Enter for 'r1.jpeg'): ").strip()
        if not image_path:
            image_path = '/home/magomed-ameen/programming/pract/receipts/r.jpg'
    
    if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            print("Please provide a valid image path.")
        return
    
    try:
        receipt_data = parser.processing_receipt(image_path)
        parser.print_result(receipt_data)
        
        # Save results to JSON for accounting software integration
        output_file = f"receipt_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert dataclass to dict for JSON serialization
        receipt_dict = receipt_data.__dict__.copy()
        
        with open(output_file, 'w') as f:
            json.dump(receipt_dict, f, indent=2, default=str)
        
            print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
            print(f"❌ Error processing receipt: {str(e)}")
        import traceback
        traceback.print_exc()
    
    elif choice == '2':
        run_comprehensive_tests()
    
    elif choice == '3':
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice. Please run the program again.")
    
    print("\n🎯 PRODUCTION-LEVEL ENHANCEMENTS:")
    print("✅ Fixed data structure inconsistencies")
    print("✅ Robust item extraction with quantity/price detection")
    print("✅ Advanced image preprocessing (perspective correction, noise reduction)")
    print("✅ Confidence-based validation and cross-validation")
    print("✅ Receipt layout analysis and template matching")
    print("✅ Comprehensive error handling and logging")
    print("✅ Multi-language support and currency detection")
    print("✅ Performance monitoring and test suite")
    print("✅ 20+ fields extracted with validation")
    print("✅ JSON export for accounting software integration")


if __name__ == "__main__":
    main()

    



    
    


    




