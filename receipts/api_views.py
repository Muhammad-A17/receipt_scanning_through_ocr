from rest_framework import generics,status#generic is an API to view classes
from rest_framework.response import Response#sends json responses to requests
from rest_framework.parsers import MultiPartParser,FormParser#handles file uploads
from rest_framework.utils import serializer_helpers
from .models import Receipt,UserProfileReg# Receipt is the db model, UserProfileReg is the user registration model
from .serializers import ReceiptSerializer,UserProfileRegSerializer#serializer converts python objects to json
from decimal import Decimal,InvalidOperation
import os
import sys
from datetime import datetime
# Import OCR parser from package
from receipt_scanning_through_ocr.ocr_more_lat import EnhancedReceiptParser

def to_decimal_or_none(v):
    if v in (None, '', 'NaN', 'Infinity', '-Infinity'):
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError, TypeError):
        return None

def date_handler(date_str):
    if not date_str or date_str in ('None','','NaN'):
        return None
    try:
        if len(date_str) == 10 and date_str.count('-') == 2:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        else:
            return None
    except (ValueError, AttributeError):
        return None




class ReceiptListAPIView(generics.ListCreateAPIView):#ListCreateAPIView handles both GET(list) and POST(create) requests
    queryset = Receipt.objects.all()#the db records that will be retrieved
    serializer_class = ReceiptSerializer#the serializer class that will be used to convert the db records to json

class ReceiptDetailAPIView(generics.RetrieveAPIView):#RetrieveAPIView handles GET requests for a single record, gets one specific receipt by id
    queryset = Receipt.objects.all()
    serializer_class=ReceiptSerializer

class ReceiptUploadAPIView(generics.CreateAPIView):
    """Upload image only - no OCR processing yet"""
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        # Only accept image upload
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Create receipt with just the image
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        receipt = serializer.save()
        
        # Return receipt with just image info (not processed yet)
        return Response({
            'id': receipt.id,
            'image': receipt.image.url,
            'processed': False,
            'message': 'Image uploaded successfully. Click "Process Receipt" to extract data.'
        }, status=status.HTTP_201_CREATED)

class ReceiptProcessAPIView(generics.CreateAPIView):
    """Process receipt with OCR - called when user clicks "Process" button"""
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer

    def create(self, request, *args, **kwargs):
        receipt = self.get_object()
        
        if receipt.processed:
            return Response({'message': 'Receipt already processed'}, status=status.HTTP_200_OK)
        
        try:
            # Process with OCR
            print(f"Image path: {receipt.image.path}")
            print(f"Image exists: {os.path.exists(receipt.image.path)}")
            image_path = receipt.image.path
            parser = EnhancedReceiptParser()
            print("Parser created successfully")
            receipt_data = parser.processing_receipt(image_path)
            print(f"OCR result: {receipt_data}")
            
            # Update receipt with extracted data
            receipt.merchant_name = receipt_data.merchant_name
            receipt.merchant_address = receipt_data.merchant_address
            receipt.merchant_phone = receipt_data.merchant_phone
            receipt.merchant_email = receipt_data.merchant_email
            receipt.date = date_handler(receipt_data.date)
            receipt.time = receipt_data.time
            receipt.transaction_id = receipt_data.transaction_id
            receipt.receipt_number = receipt_data.receipt_number
            receipt.tip = to_decimal_or_none(receipt_data.tip)
            receipt.tax = to_decimal_or_none(receipt_data.tax)
            receipt.sub_total = to_decimal_or_none(receipt_data.sub_total)
            receipt.total = to_decimal_or_none(receipt_data.total)
            receipt.discount = to_decimal_or_none(receipt_data.discount)
            receipt.items = receipt_data.items
            receipt.payment_method = receipt_data.payment_method
            receipt.card_type = receipt_data.card_type
            receipt.card_last_four = receipt_data.card_last_four
            receipt.category = receipt_data.category
            receipt.tax_rate = to_decimal_or_none(receipt_data.tax_rate)
            receipt.currency = receipt_data.currency
            receipt.confidence_scores = receipt_data.confidence_scores
            receipt.processed = True
            receipt.save()

            return Response(ReceiptSerializer(receipt).data, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error: {e}")
            return Response({'error': f'OCR processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ReceiptEditAPIView(generics.UpdateAPIView):
    queryset = Receipt.objects.all()
    serializer_class=ReceiptSerializer

    def update(self, request, *args, **kwargs):
        receipt=self.get_object()

        if not receipt.processed:
            return Response({'error': 'Receipt must be processed first, and it hasnt been'},status=status.HTTP_400_BAD_REQUEST)
        receipt.merchant_name = request.data.get('merchant_name', receipt.merchant_name)
        receipt.merchant_address = request.data.get('merchant_address', receipt.merchant_address)
        receipt.merchant_phone = request.data.get('merchant_phone', receipt.merchant_phone)
        receipt.merchant_email = request.data.get('merchant_email', receipt.merchant_email)
        receipt.date = request.data.get('date', receipt.date)
        receipt.time = request.data.get('time', receipt.time)
        receipt.transaction_id = request.data.get('transaction_id', receipt.transaction_id)
        receipt.receipt_number = request.data.get('receipt_number', receipt.receipt_number)
        receipt.tip = to_decimal_or_none(request.data.get('tip', receipt.tip))
        receipt.tax = to_decimal_or_none(request.data.get('tax', receipt.tax))
        receipt.sub_total = to_decimal_or_none(request.data.get('sub_total', receipt.sub_total))
        receipt.total = to_decimal_or_none(request.data.get('total', receipt.total))
        receipt.discount = to_decimal_or_none(request.data.get('discount', receipt.discount))
        receipt.items = request.data.get('items', receipt.items)
        receipt.payment_method = request.data.get('payment_method', receipt.payment_method)
        receipt.card_type = request.data.get('card_type', receipt.card_type)
        receipt.card_last_four = request.data.get('card_last_four', receipt.card_last_four)
        receipt.category = request.data.get('category', receipt.category)
        receipt.tax_rate = to_decimal_or_none(request.data.get('tax_rate', receipt.tax_rate))
        receipt.currency = request.data.get('currency', receipt.currency)

        receipt.save()
        return Response(ReceiptSerializer(receipt).data, status=status.HTTP_200_OK)

    
class RegisterAPIview(generics.CreateAPIView):
    queryset=UserProfileReg.objects.all()
    serializer_class=UserProfileRegSerializer
    

    


        


