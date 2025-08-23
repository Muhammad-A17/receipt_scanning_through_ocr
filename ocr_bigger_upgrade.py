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

try:
    nlp=spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class ReceiptData:
    merchant_name:Optional[str]=None
    merchant_address:Optional[str]=None
    merchant_phone:Optional[str]=None
    merchant_email:Optional[str]=None

    date:Optional[str]=None
    time:Optional[str]=None

    transaction_id:Optional[str]=None
    receipt_number:Optional[str]=None

    tip:Optional[str]=None

    tax:Optional[str]=None
    sub_total:Optional[str]=None

    total:Optional[str]=None
    discount:Optional[str]=None

    items:List[Dict[str, Any]]=None

    payment_method:Optional[str]=None
    card_type:Optional[str]=None
    card_last_four:Optional[str]=None

    category:Optional[str]=None
    tax_rate:Optional[str]=None
    currency:Optional[str]=None

    confidence_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.items is None:
            self.items=[]
        if self.confidence_scores is None:
            self.confidence_scores={}

class EnhancedReceiptParser:
    def __init__(self):
        self.ocr=PaddleOCR(use_angle_cls=True,lang='en',show_log=True)

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
    def preprocessing(self,image_path):
        """
        Apply multiple preprocessing techniques and let OCR choose the best result
        """
        img=cv2.imread(image_path)
        if img is None:
            raise ValueError("Could'nt read image")
    
        versions={}
        versions['original']=img


        #converted to grayscale
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Clahe - contrast limited adaptive histogram eualiztaion
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        versions['clahe']=clahe.apply(gray)

        # gaussian blur plus sharpening
        blurred=cv2.GaussianBlur(gray,(3,3),0)
        kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        versions['sharpened']=cv2.filter2D(blurred,-1,kernel)
        
        #denoising and enhancing
        denoised=cv2.fastNlMeansDenoising(gray)
        versions['denoised']=denoised

        # binary thresholding (otsu method)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions['binary_otsu'] = binary

        #adaptive thresholding
        adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
        versions['adaptive']=adaptive

        #morphological operations for text enhancements
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
        morph=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
        versions['morphological']=morph

        print(f"{len(versions)} versions created")
        return versions
    
    def select_best_ocr_result(self,image_path:str)->Tuple[List,str,float]:
        versions=self.preprocessing(image_path)

        best_result=None
        best_score=0
        best_version="original"

        for version_name,img_data in versions.items():
            try:
                result=self.ocr.ocr(img_data,cls=True)

                if not result or not result[0]:
                    continue

                lines=result[0]
                if not lines:
                    continue

                #  scoring algorithm
                total_conf=sum(line[1][1] for line in lines)
                avg_confidence=total_conf/len(lines)
                high_conf_lines=sum(1 for line in lines if line[1][1]>0.8)
                medium_conf_lines=sum(1 for line in lines if 0.6<=line[1][1]<=0.8)

                #score based on uality and uantity
                quality_score = (avg_confidence * 0.4 + 
                               (high_conf_lines / len(lines)) * 0.4 + 
                               (medium_conf_lines / len(lines)) * 0.2)
                
                # Bonus for finding key receipt terms
                text_content = " ".join([line[1][0] for line in lines]).lower()
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
        return best_result, best_version, best_score
    
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
            
        #finding using high confidence lines
        for i, line in enumerate(lines[:5]):
            text=line[1][0].strip()
            conf=line[1][1]

            if len(text)>=3 and conf>0.7 and not re.match(r'^[\d\s\-\/\$\.,:]+$', text) and not any(skip in text.lower() for skip in ['receipt', 'invoice', 'total', 'date', 'time']):
                
                merchant_candidates.append((text, conf, f'top_line_{i}'))

        #looking for business-type keyords
        for line in lines[:10]:
            text=line[1][0].strip()
            conf=line[1][1]

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
                text=line[1][0].lower()
                box=line[0]

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

                        other_text=other_line[1][0]
                        other_box=other_line[0]
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
    def processing_receipt(self,image_path:str)->ReceiptData:
        print(f"PROCESSING RECEIPT: {image_path}")
        
        ocr_result,best_version,quality_score = self.select_best_ocr_result(image_path)

        if not ocr_result or not ocr_result[0]:
            print("NO TEXT EXTRACTED FROM THE IMAGE")
            return ReceiptData()
        
        lines=ocr_result[0]
        full_text="\n".join([line[1][0] for line in lines])

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
        
        # Amounts
        amounts = self.extract_amounts(lines, full_text)
        receipt_data.total_amount = amounts.get('total')
        receipt_data.subtotal = amounts.get('subtotal')
        receipt_data.tax_amount = amounts.get('tax')
        receipt_data.tip_amount = amounts.get('tip')
        receipt_data.discount_amount = amounts.get('discount')
        
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
        
        # Business categorization
        receipt_data.category = self.categorize_business(receipt_data.merchant_name, full_text)
        
        #calculating tax rate
        if receipt_data.tax_amount and receipt_data.sub_total:
            receipt_data.tax_rate=round((receipt_data.tax_amount/receipt_data.sub_total)*100,2)

        receipt_data.currency="USD"

        receipt_data.confidence_scores={
            'ocr_quality':quality_score,
            'merchant': merchant_info.get('confidence',0)
        }

        return receipt_data
    
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
        if receipt_data.subtotal:
            print(f"   Subtotal: ${receipt_data.subtotal:.2f}")
        if receipt_data.tax_amount:
            print(f"   Tax: ${receipt_data.tax_amount:.2f}")
        if receipt_data.tax_rate:
            print(f"   Tax Rate: {receipt_data.tax_rate}%")
        if receipt_data.tip_amount:
            print(f"   Tip: ${receipt_data.tip_amount:.2f}")
        if receipt_data.discount_amount:
            print(f"   Discount: ${receipt_data.discount_amount:.2f}")
        if receipt_data.total_amount:
            print(f"   TOTAL: ${receipt_data.total_amount:.2f}")
        
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
        
        # Count extracted fields
        extracted_fields = 0
        total_fields = 0
        
        for field_name, field_value in receipt_data.__dict__.items():
            if field_name not in ['items', 'confidence_scores'] and field_value is not None:
                if isinstance(field_value, (str, int, float)) and field_value != "":
                    extracted_fields += 1
            total_fields += 1
        
        print(f"   Fields Extracted: {extracted_fields}/{total_fields-2}")  # -2 for items and confidence_scores


def main():
    print(" Enhanced Receipt Parser with Named Entity Recognition")
    print("=" * 65)
    print("Professional-level receipt processing for accounting applications")
    
    # Initialize parser
    parser = EnhancedReceiptParser()
    
    # Process receipt
    image_path = 'r1.jpeg'  # Update this path
    
    if not os.path.exists(image_path):
        print(f" File not found: {image_path}")
        print("Please update the image_path variable with your receipt image.")
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
        
        print(f"\n Results saved to: {output_file}")
        
        print("\n ENHANCEMENTS IN THIS VERSION:")
        print(" Named Entity Recognition (NER) for better text understanding")
        print(" 15+ fields extracted (vs 4 in original)")
        print(" Business categorization for accounting")
        print(" Enhanced spatial analysis using bounding boxes")
        print(" Multiple extraction strategies with fallbacks")
        print(" Professional data structure for accounting integration")
        print(" JSON export for accounting software")
        print(" Confidence scoring and quality metrics")
        print("Payment method and card information extraction")
        print(" Transaction ID and receipt number detection")
        
    except Exception as e:
        print(f" Error processing receipt: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

    



    
    


    




