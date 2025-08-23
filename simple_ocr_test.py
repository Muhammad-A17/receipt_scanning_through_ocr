from paddleocr import PaddleOCR
import os

def simple_ocr_test(image_path):
    """
    Simple function to test OCR on an image
    """
    # Initialize PaddleOCR
    ocr=PaddleOCR(use_angle_cls=True,lang='en',show_log=False)

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        # Run OCR on the image
        result=ocr.ocr(image_path,cls=True)

        print("\n" + "="*50)
        print(" OCR RESULTS:")
        print("="*50)

        if result and result[0]:
            print(f" Found {len(result[0])} text lines\n")

            for i, line in enumerate(result[0],1):
                text=line[1][0]
                confidence=line[1][1]

                if confidence>0.9:
                    print("High Confidence, greater than 0.9.")
                elif confidence>0.7 and confidence<=0.9:
                    print("Medium Confidence, greater than 0.7, less than or eual to 0.9.")
                else:
                    print("Bad Confidence")

            print("\n" + "-"*30)
            print(" FULL TEXT:")
            print("-"*30)

            full_text="\n".join([line[1][0] for line in result[0]])
            print(full_text)
        else:
            print("No text found in image")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def main():
    print(" Simple Receipt OCR Test")
    print("-" * 30)

    while True:
        image_path='r1.jpeg'
        
        if image_path:
            simple_ocr_test(image_path)
        else:
            print("  Please enter a valid image path")

if __name__ == "__main__":
    main()

        





