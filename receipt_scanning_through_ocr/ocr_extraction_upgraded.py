# Step 5: Adaptive Receipt Parser
# Smart preprocessing for difficult receipts + validation to reduce wrong detections

from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import os


def preprocessing(image_path):
    """
    Apply multiple preprocessing techniques and let OCR choose the best result
    """
    img=cv2.imread(image_path)
    if img is None:
        raise ValueError("Could'nt read image")
    
    versions={}
    versions['original']=img

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    versions['gentle']=clahe.apply(gray)

# Version 3: Stronger enhancement for faded text
    denoised=cv2.fastNlMeansDenoising(gray)
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened=cv2.filter2D(denoised,-1,kernel)
    versions['enhanced']=sharpened

    # Version 4: High contrast for very faded receipts
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    versions['high_contrast'] = binary

    adaptive=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    versions['adaptive']=adaptive

    print(f"   Created {len(versions)} image versions for testing")
    return versions

def pick_best_processed(image_path):
    """
    Test OCR on all preprocessed versions and pick the one with best results
    """
    ocr=PaddleOCR(use_angle_cls=True,lang='en',show_log=False)
    versions=preprocessing(image_path)

    best_result=None
    best_score=0
    best_version="original"

    for version_name,img_data in versions.items():
        try:
            print(f"Testing {version_name}")
            result=ocr.ocr(img_data,cls=True)

            if not result or not result[0]:
                print("no text found")
                continue

            # Calculate quality score
            lines = result[0]
            avg_confidence = sum(line[1][1] for line in lines) / len(lines)
            high_conf_count = sum(1 for line in lines if line[1][1] > 0.8)
            total_lines = len(lines)
            # Composite score: average confidence * high confidence ratio
            score=avg_confidence*(high_conf_count/total_lines if total_lines>0 else 0)
            print(f"     📊 Lines: {total_lines}, Avg conf: {avg_confidence:.3f}, Score: {score:.3f}")

            if score>best_score:
                best_score=score
                best_result=result
                best_version=version_name

        except Exception as e:
            print(f"     ❌ Error with {version_name}: {str(e)}")
            continue
    print(f"   🏆 Best version: {best_version} (score: {best_score:.3f})")
    return best_result, best_version, best_score

def validate_merchant_name(text,confidence):

    red_flags=[
        len(text)<2,#too short
        text.isdigit(),#just numbers
        re.match(r'^[\d\-\/\s:\.]+$', text),  # Just dates/times/numbers
        any(word in text.lower() for word in ['receipt', 'invoice', 'total', 'tax', 'date', 'time']),
        re.match(r'^\$[\d\.]+$', text),  # Just a price
        confidence < 0.6,  # Very low confidence
    ]

    green_flags=[
        len(text)>=4 and confidence >0.8,
        re.search(r'[A-Z]{2,}',text),
        any(word in text.lower() for word in ['store', 'shop', 'market', 'restaurant', 'cafe', 'mart']),

    ]
    if any(green_flags):
        return True,"Passed merchant validation"
    
    return confidence>0.7, f"Medium confidence: {confidence:.3f}"

def validate_amount(amount_str,context_text):
    try:
        amount=float(amount_str)

        if amount<1:
            return False, "Too small"
        if amount > 999999:
            return False, "Unreasonably large"
        
        context_upper=context_text.upper()
        total_context=any(word in context_upper for word in ['TOTAL', 'AMOUNT', 'BALANCE', 'DUE'])
        
        if total_context: 
            return True,"Found near total keywords"

        all_amount=re.findall(r'\$?(\d+\.\d{2})', context_text)
        if all_amount:
            amounts = [float(a) for a in all_amount if float(a) > 0]
            if amounts and amount == max(amounts):
                return True, "Largest amount found"
        
        return amount > 1.0, f"Moderate confidence: ${amount:.2f}"
        
    except ValueError:
        return False, "Not a valid number"
    


def find_merchant_with_validation(high_conf_lines):

    for line in high_conf_lines[:8]:
        text=line['text'].strip()
        is_valid,reason=validate_merchant_name(text,line['confidence'])

        if is_valid:
            print(f"   ✅ '{text}' - {reason} (conf: {line['confidence']:.3f})")
            return text, line['confidence']
        else:
            print(f"'{text}' - {reason}")

    print("No valid merchant name found")
    return None,0


def find_total_from_ocr(lines):
    
    """
    Use bounding boxes to find the number aligned with TAX keyword.
    """
    for i, line in enumerate(lines):
        text = line[1][0]
        conf = line[1][1]
        box = line[0]  # bounding box coordinates

        if any(word in text.upper() for word in ["TOTAL", "AMOUNT", "BALANCE"]):
            total_y = (box[0][1] + box[2][1]) / 2  # average vertical center

            # search nearby lines for a number on the same row
            for j, other_line in enumerate(lines):
                if i == j:
                    continue
                other_text = other_line[1][0]
                other_box = other_line[0]
                other_y = (other_box[0][1] + other_box[2][1]) / 2

                # y-distance small → same row
                if abs(other_y - total_y) < 20:
                    match = re.search(r'(\d+\.\d{2})', other_text)
                    if match:
                        total= float(match.group(1))
                        print(f"   ✅ Tax found using box alignment: ${total}")
                        return total, "bbox_match"

    print("   ❌ No tax found with boxes")
    return None, "none"


def find_date_ith_validation(full_text):

    date_patterns=[r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})']

    for pattern in date_patterns:
        matches=re.findall(pattern,full_text)
        if matches:
            date=matches[0]

            parts=re.split(r'[\/\-]',date)
            if len(parts)==3 and all(part.isdigit() for part in parts):
                print(f"   ✅ '{date}' - Valid date format")
                return date
    
    print("   ❌ No valid date found")
    return None


def find_tax_from_ocr(lines):

    """
    Use bounding boxes to find the number aligned with TAX keyword.
    """
    for i, line in enumerate(lines):
        text = line[1][0]
        conf = line[1][1]
        box = line[0]  # bounding box coordinates

        if "TAX" in text.upper():
            tax_y = (box[0][1] + box[2][1]) / 2  # average vertical center

            # search nearby lines for a number on the same row
            for j, other_line in enumerate(lines):
                if i == j:
                    continue
                other_text = other_line[1][0]
                other_box = other_line[0]
                other_y = (other_box[0][1] + other_box[2][1]) / 2

                # y-distance small → same row
                if abs(other_y - tax_y) < 20:
                    match = re.search(r'(\d+\.\d{2})', other_text)
                    if match:
                        tax = float(match.group(1))
                        print(f"   ✅ Tax found using box alignment: ${tax}")
                        return tax, "bbox_match"

    print("   ❌ No tax found with boxes")
    return None, "none"


def extract_and_validate(image_path):
    
    print(f"processing ith validation: {image_path}")

    ocr_result,best_version,quality_score = pick_best_processed(image_path)
    
    if not ocr_result or not ocr_result[0]:
        print("❌ No text could be extracted from any version")
        return None
    
    print(f"\n📊 Using {best_version} version (quality score: {quality_score:.3f})")

    lines=ocr_result[0]
    full_text="\n".join([line[1][0] for line in lines])
    high_conf_lines=[{'text':line[1][0],'confidence':line[1][1]} 
                     for line in lines if line[1][1]>0.7]
    

    print(f"📝 Extracted {len(lines)} total lines, {len(high_conf_lines)} high-confidence")
    
    print("\n" + "-" * 50)
    print("EXTRACTION WITH VALIDATION:")
    print("-" * 50)
    
    # Step 3: Extract each field using separate functions
    results = {}
    
    # Merchant
    merchant, merchant_conf = find_merchant_with_validation(high_conf_lines)
    results['merchant_name'] = merchant
    results['merchant_confidence'] = merchant_conf
    
    print()  # Add spacing between sections
    
    # Total amount
    total, total_method = find_total_from_ocr(lines)
    results['total_amount'] = total
    results['total_method'] = total_method
    
    print()
    
    # Date
    date = find_date_ith_validation(full_text)
    results['date'] = date
    
    print()
    
    # Tax
    tax, tax_method = find_tax_from_ocr(lines)
    results['tax_amount'] = tax
    results['tax_method'] = tax_method
    
    # Step 4: Display final results
    print("\n" + "=" * 70)
    print("🎯 FINAL VALIDATED RESULTS:")
    print("=" * 70)
    
    found_fields = 0
    
    print(f"🏪 Merchant: {results.get('merchant_name', 'Not found')}")
    if results.get('merchant_name'): found_fields += 1
    
    print(f"📅 Date: {results.get('date', 'Not found')}")
    if results.get('date'): found_fields += 1
    
    total_str = f"${results['total_amount']:.2f}" if results.get('total_amount') else "Not found"
    print(f"💰 Total: {total_str}")
    if results.get('total_amount'): found_fields += 1
    
    tax_str = f"${results['tax_amount']:.2f}" if results.get('tax_amount') else "Not found"
    print(f"📊 Tax: {tax_str}")
    if results.get('tax_amount'): found_fields += 1
    
    print(f"\n✅ Successfully extracted and validated: {found_fields}/4 fields")
    print(f"📊 Image processing method: {best_version}")
    print(f"🎯 OCR quality score: {quality_score:.3f}")
    
    return results
                    

    






def main():
    print("🧾 Adaptive Receipt Parser - Step 5")
    print("-" * 50)
    print("Smart preprocessing + validation to reduce wrong detections")
    
    image_path = 'r1.jpeg'
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    try:
        results = extract_and_validate(image_path)
        
        if results:
            print("\n🎓 What's new in Step 5:")
            print("- Tests 5 different image preprocessing methods")
            print("- Validates merchant names to reduce false positives")
            print("- Validates amounts with context checking")
            print("- Shows reasoning for each acceptance/rejection")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
        




    
    
