from paddleocr import PaddleOCR
import re
import os
from datetime import datetime


def extract_text_ith_confidence(image_path,min_conf=0.5):
    """
    Extract text but filter by confidence to reduce noise
    """

    ocr=PaddleOCR(use_angle_cls=True,lang='en',show_log=False)
    result=ocr.ocr(image_path,cls=True)

    if not result or not result[0]:
        return [],"", ""
    
    high_conf_lines=[]
    medium_conf_lines=[]
    full_text=""

    for line in result[0]:
        text=line[1][0].strip()
        confidence=line[1][1]

        line_data={
            'text':text,
            'confidence':confidence
        }

        if confidence>= 0.8:
            high_conf_lines.append(line_data)
        elif confidence >= min_conf:
            medium_conf_lines.append(line_data)

        full_text+=text+"\n"

    print(f"   ✅ High confidence lines: {len(high_conf_lines)}")
    print(f"   🟡 Medium confidence lines: {len(medium_conf_lines)}")
    print(f"   🔴 Low confidence lines: {len(result[0]) - len(high_conf_lines) - len(medium_conf_lines)}")

    return high_conf_lines,medium_conf_lines,full_text

def find_merchant_name(high_conf_lines,medium_conf_lines):
    """
    Improved merchant name detection with multiple strategies
    """
    # High confidence lines first
    for i, line in enumerate(high_conf_lines[:5]):
        text=line['text']
        confidence=line['confidence']

        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^[\d\-\/\s:]+$',  # Just dates/times
            r'^(receipt|invoice|bill|tel|phone|address)$',  # Common words
            r'^\$[\d\.]+$',  # Just prices
        ]
        if any(re.match(pattern,text,re.IGNORECASE) for pattern in skip_patterns):
            continue

        if len(text)>=3:
            print(f"   Found (Strategy 1): '{text}' (confidence: {line['confidence']:.3f})")
            return text, line['confidence']
        
    business_patterns=[
        r'[A-Z]{2,}[\s&]*[A-Z]*',  # Uppercase words (like "WALMART", "TARGET")
        r'\w+\s+(STORE|SHOP|MARKET|RESTAURANT)', # Contains business words
        ]

    all_lines=high_conf_lines+medium_conf_lines
    for line in all_lines[:8]:
        text=line['text']
        for pattern in business_patterns:
            if re.search(pattern,text,re.IGNORECASE) and len(text)>=4:
                print(f"   Found (Strategy 2): '{text}' (confidence: {line['confidence']:.3f})")
                return text, line['confidence']
            
    
    
    print("   ❌ Could not identify merchant name")
    return None,0

def find_date(full_text,high_conf_lines):
    """
    Look for date patterns in the text
    """

    # Common date patterns
    for line in high_conf_lines:
        date_patterns = [
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',  # MM/DD/YYYY or MM-DD-YY
        r'(\d{1,2}\s+\w{3}\s+\d{2,4})',          # DD MMM YYYY
        r'(\w{3}\s+\d{1,2},?\s+\d{2,4})',        # MMM DD, YYYY
        ]
    

        for pattern in date_patterns:
            matches=re.findall(pattern,line['text'],re.IGNORECASE)
            if matches:
                date_found=matches[0]
                print(f" Found: '{date_found}'")
                return date_found,line['confidence']
    # Fallback to full text search
    date_patterns = [
        r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        r'(\d{2,4}[\/\-]\d{1,2}[\/\-]\d{1,2})',
    ]
    for pattern in date_patterns:
        matches=re.findall(pattern,full_text)
        if matches:
            date_found=matches[0]
            print(f"   Found (fallback): '{date_found}'")
            return date_found, 0.5


        
    print("    Could not find date")
    return None,0

def find_total_amount(full_text,high_conf_lines):
    """
    Look for the total amount
    """
    print("💰 Looking for total amount...")
    
    # Convert to uppercase for easier matching
    amounts_found={}
    
    
    # Pattern 1: Look for explicit total keywords
    total_patterns = [
        (r'TOTAL[\s:$]*(\d+\.?\d{2})', 'explicit_total'),
        (r'AMOUNT[\s:$]*(\d+\.?\d{2})', 'amount'),
        (r'BALANCE[\s:$]*(\d+\.?\d{2})', 'balance'),
        (r'GRAND\s+TOTAL[\s:$]*(\d+\.?\d{2})', 'grand_total'),
        (r'SUB\s*TOTAL[\s:$]*(\d+\.?\d{2})', 'subtotal'),
    ]

    text_upper=full_text.upper()


    
    for pattern,label in total_patterns:
        matches = re.findall(pattern, text_upper)
        if matches:
            try:
                amount = float(matches[-1])  # Take the last match (often the final total)
                amounts_found[label]=amount
                print(f"   Found {label}: ${amount:.2f}")
            except ValueError:
                continue
    
    # Fallback: look for dollar amounts at end of lines

    dollar_amount=[]
    for line in high_conf_lines:
        dollar_matches=re.findall(r'\$(\d+\.\d{2})',line['text'])
        for match in dollar_matches:
            try:
                amount=float(match)
                dollar_amount.append((amount,line['confidence']))
            except ValueError:
                continue

    if dollar_amount:
        # Sort by amount (largest first) and take highest confidence among large amounts
        dollar_amount.sort(key=lambda x: x[0],reverse=True)
        print(f"   Found dollar amounts: {[f'${amt:.2f}' for amt, _ in dollar_amount[:3]]}")
        amounts_found['dollar_amount'] = dollar_amount[0][0]



    if 'grand_total' in amounts_found:
        return amounts_found['grand_total'], 'grand_total'
    elif 'explicit_total' in amounts_found:
        return amounts_found['explicit_total'], 'explicit_total'
    elif 'amount' in amounts_found:
        return amounts_found['amount'], 'amount'
    elif 'dollar_amount' in amounts_found:
        return amounts_found['dollar_amount'], 'dollar_amount'
    
    print("   ❌ Could not find total amount")
    return None,'none'

def find_tax_amount(full_text):
    """
    Look for tax amount
    """
    print("📊 Looking for tax amount...")
    
    text_upper = full_text.upper()
    
    tax_patterns = [
        (r'TAX[\s:$]*(\d+\.?\d{2})', 'tax'),
        (r'SALES\s+TAX[\s:$]*(\d+\.?\d{2})', 'sales_tax'),
        (r'HST[\s:$]*(\d+\.?\d{2})', 'hst'),
        (r'GST[\s:$]*(\d+\.?\d{2})', 'gst'),
        (r'VAT[\s:$]*(\d+\.?\d{2})', 'vat'),
        (r'(\d+\.?\d{2})\s*%?\s*TAX', 'tax_after'),
    ]
    
    for pattern,label in tax_patterns:
        matches = re.findall(pattern, text_upper)
        if matches:
            try:
                tax = float(matches[-1])
                print(f"   Found {label}: ${tax:.2f}")
                return tax,label
            except ValueError:
                continue
    
    print("   ❌ Could not find tax amount")
    return None,'none'

def parse_receipt(image_path):
    """
    Main function to parse receipt and extract key information
    """
    print(f"🧾 Parsing receipt: {image_path}")
    print("=" * 50)
    
    # Step 1: Extract text with OCR
    high_conf,medium_conf, full_text = extract_text_ith_confidence(image_path)
    
    if not high_conf and not medium_conf:
        print("❌ No text found in image")
        return None
    
    print(f"\n📊 Text Quality Assessment:")
    total_lines = len(high_conf) + len(medium_conf)
    if total_lines>0:
        high_percentage = (len(high_conf) / total_lines) * 100
        print(f"   High confidence text: {high_percentage:.1f}%")
        
        if high_percentage > 60:
            print("   🟢 Good quality image - should parse well")
        elif high_percentage > 30:
            print("   🟡 Medium quality image - some fields may be missed")
        else:
            print("   🔴 Poor quality image - results may be unreliable")
    
    print("\n" + "-" * 40)
    print("PARSING RESULTS:")
    print("-" * 40)

    receipt_data = {}
    confidence_scores={}

    merchant,merchant_conf=find_merchant_name(high_conf,medium_conf)
    
    receipt_data['merchant_name'] = merchant
    confidence_scores['merchant_name']=merchant_conf


    date_result = find_date(full_text, high_conf)
    if date_result and date_result[0]:  # Check if we got a valid result
        date, date_conf = date_result
        receipt_data['date'] = date
        confidence_scores['date'] = date_conf
    else:
        receipt_data['date'] = None
        confidence_scores['date'] = 0

    total, total_method = find_total_amount(full_text, high_conf)
    receipt_data['total_amount'] = total
    receipt_data['total_method'] = total_method
    
    # Tax
    tax, tax_method = find_tax_amount(full_text)
    receipt_data['tax_amount'] = tax
    receipt_data['tax_method'] = tax_method
    
    # Step 3: Display results
    print("\n" + "=" * 50)
    print("📋 EXTRACTED RECEIPT DATA:")
    print("=" * 50)

    def format_field(name, value, confidence=None):
        if value:
            conf_indicator = ""
            if confidence:
                if confidence > 0.8:
                    conf_indicator = "🟢"
                elif confidence > 0.6:
                    conf_indicator = "🟡"
                else:
                    conf_indicator = "🔴"
            return f"{name}: {value} {conf_indicator}"
        else:
            return f"{name}: Not found ❌"
    
    total_str = f"${receipt_data['total_amount']:.2f}" if receipt_data['total_amount'] else None
    tax_str   = f"${receipt_data['tax_amount']:.2f}" if receipt_data['tax_amount'] else None

    print(f"🏪 {format_field('Merchant', receipt_data['merchant_name'], confidence_scores['merchant_name'])}")
    print(f"📅 {format_field('Date', receipt_data['date'], confidence_scores['date'])}")
    print(f"💰 {format_field('Total', total_str)} ({receipt_data['total_method']})")
    print(f"📊 {format_field('Tax', tax_str)} ({receipt_data['tax_method']})")


    
    # Success metrics
    found_count = sum(1 for key, value in receipt_data.items() 
                     if key not in ['total_method', 'tax_method'] and value is not None)
    print(f"\n🎯 Successfully extracted: {found_count}/4 fields")
    
    return receipt_data

def main():
    print("🧾 Receipt Data Parser - Step 3")
    print("-" * 40)
    print("This extracts key information from your receipt")
    
    image_path = 'r1.jpeg'
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    receipt_data = parse_receipt(image_path)
    
    if receipt_data:
        print("\n🎓 Improvements in this version:")
        print("- Confidence-based text filtering")
        print("- Multiple parsing strategies per field")
        print("- Image quality assessment")
        print("- Better validation and fallback methods")

if __name__ == "__main__":
    main()
        




    
    
