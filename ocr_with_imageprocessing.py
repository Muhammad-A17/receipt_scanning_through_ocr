from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import os

def preprocess_image(image_path,show_steps=False):
    """
    Clean up the image before OCR to improve accuracy
    """
    print("Preprocessing image")
    img=cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read Image")
    
    # Step 1: Convert to grayscale
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if show_steps:
        print("Converted to grayscale")
    
    # Step 2: Remove noise
    denoised=cv2.fastNlMeansDenoising(gray)


    # Step 3: Improve contrast
    clahe=cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    contrast_enhanced= clahe.apply(denoised)
    if show_steps:
        print("   ✓ Enhanced contrast")



    # Step 4: Apply threshold to make text clearer
    _, thresh=cv2.threshold(contrast_enhanced,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show_steps:
        print("   ✓ Applied threshold")

    # Save processed image for comparison (optional)
    processed_path="processed_"+os.path.basename(image_path)
    cv2.imwrite(processed_path,thresh)
    if show_steps:
        print(f"   ✓ Saved processed image as: {processed_path}")
    
    return thresh, processed_path

def run_ocr_comparison(image_path):
    """
    Compare OCR results: original vs preprocessed image
    """
    ocr=PaddleOCR(use_angle_cls=True,lang='en',show_log=False)

    print(f"\n Testing image: {image_path}")
    
    # Test 1: Original image
    print("\n" + "="*50)
    print("📄 ORIGINAL IMAGE RESULTS:")
    print("="*50)

    try:
        result_original = ocr.ocr(image_path,cls=True)
        original_scores=analyze_results(result_original,"Original")
    except Exception as e:
        print(f"❌ Error with original: {str(e)}")
        return
    

    # Test 2: Preprocessed image
    print("\n" + "="*50)
    print(" PREPROCESSED IMAGE RESULTS:")
    print("="*50)

    try:
        processed_img,processed_path=preprocess_image(image_path,show_steps=True)
        result_processed=ocr.ocr(processed_img,cls=True)
        processed_scores=analyze_results(result_processed,"Processed")
        
        # Comparison
        print("\n" + "="*50)
        print("📊 COMPARISON SUMMARY:")
        print("="*50)
        print(f"Original    - Lines: {original_scores['total_lines']}, Avg Confidence: {original_scores['avg_confidence']:.3f}")
        print(f"Processed   - Lines: {processed_scores['total_lines']}, Avg Confidence: {processed_scores['avg_confidence']:.3f}")

        if processed_scores['avg_confidence'] > original_scores['avg_confidence']:
            print("🎉 Preprocessing IMPROVED the results!")
        else:
            print("🤔 Original was better (this happens sometimes)")
            
        print(f"\n💡 You can see the processed image at: {processed_path}")
        
    except Exception as e:
        print(f"❌ Error with preprocessing: {str(e)}")

def analyze_results(result,label):
    """
    Analyze OCR results and return statistics
    """
    if not result or not result[0]:
        print(f"❌ No text found in {label} image")
        return {'total_lines': 0, 'avg_confidence': 0.0, 'high_conf_lines': 0}
    lines=result[0]
    confidences=[line[1][1] for line in lines]

    total_lines=len(lines)
    avg_confidence=sum(confidences)/len(confidences) if confidences else 0
    high_conf_lines = sum(1 for conf in confidences if conf > 0.8)
    
    print(f"✅ Found {total_lines} text lines")
    print(f"📈 Average confidence: {avg_confidence:.3f}")
    print(f"🎯 High confidence lines (>0.8): {high_conf_lines}")
    
    print(f"\n📝 {label} Text:")


    for i, line in enumerate(lines[:10],1):# Show first 10 liness

        text=line[1][0]
        confidence=line[1][1]

        if confidence > 0.9:
            print("HIGH CONFIDENCE")
        elif confidence > 0.7 and confidence<=0.9:
            print("MEDIUM CONFIDENCE")
        else:
            print("LO CONFIDENCE")
    if len(lines)>10:
        print(f"... and {len(lines) - 10} more lines")
    
    return {
        'total_lines': total_lines,
        'avg_confidence': avg_confidence,
        'high_conf_lines': high_conf_lines
    }

def main():
    print(" OCR with Preprocessing Test")
    print("-" * 40)
    print("This will compare original vs processed image results")
    
    # Get image path (simplified to avoid the loop issue)
    image_path = 'r1.jpeg'
    
    if not os.path.exists(image_path):
        print(f" File not found: {image_path}")
        return
    
    run_ocr_comparison(image_path)
    
    print("\n What you learned:")
    print("- Image preprocessing can improve OCR accuracy")
    print("- Different techniques work better for different images")
    print("- Confidence scores help you judge quality")

if __name__ == "__main__":
    main()

    

