import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from ocr_bigger_upgrade import EnhancedReceiptParser
import tempfile





def interf():
    st.write("Receipt Scanner ") 
    img_file = st.file_uploader("Digga, Upload an image of a Receipt", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded", use_column_width=True)


        suffix = "." + img_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        with st.spinner("Processing..."):
            parser = EnhancedReceiptParser()
            receipt_data = parser.processing_receipt(tmp_path)
        st.success("Done")

        data = {k: v for k, v in receipt_data.__dict__.items()
                if k not in ["items", "confidence_scores"] and v not in [None, "", []]}

        # Pretty labels (optional minimal mapping)
        labels = {
            "merchant_name": "Merchant",
            "merchant_address": "Address",
            "merchant_phone": "Phone",
            "merchant_email": "Email",
            "category": "Category",
            "date": "Date",
            "time": "Time",
            "transaction_id": "Transaction ID",
            "receipt_number": "Receipt Number",
            "subtotal": "Subtotal",
            "tax_amount": "Tax",
            "tax_rate": "Tax Rate (%)",
            "tip_amount": "Tip",
            "discount_amount": "Discount",
            "total_amount": "Total",
            "payment_method": "Payment Method",
            "card_type": "Card Type",
            "card_last_four": "Card Last 4",
            "currency": "Currency",
        }

        st.subheader("Extracted Fields")
        for key, value in data.items():
            label = labels.get(key, key.replace("_", " ").title())
            st.write(f"**{label}:** {value}")

        # If you want, you can also show confidence scores
        if receipt_data.confidence_scores:
            st.subheader("Confidence")
            for k, v in receipt_data.confidence_scores.items():
                st.write(f"**{k.replace('_',' ').title()}:** {v:.3f}")

    else:
        st.write("No image uploaded")

    
    



if __name__=="__main__":
    interf()
