"""
API Views for Receipt Scanner Application
Centralized API endpoints using service layer
"""
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
import logging
import threading
import time

from .models import Receipt, UserProfileReg
from .serializers import ReceiptSerializer, UserProfileRegSerializer
from .services.ocr_service import get_ocr_service
from .services.receipt_service import ReceiptService

logger = logging.getLogger(__name__)




class ReceiptListAPIView(generics.ListCreateAPIView):#ListCreateAPIView handles both GET(list) and POST(create) requests
    queryset = Receipt.objects.all()#the db records that will be retrieved
    serializer_class = ReceiptSerializer#the serializer class that will be used to convert the db records to json

class ReceiptDetailAPIView(generics.RetrieveUpdateDestroyAPIView):#RetrieveUpdateDestroyAPIView handles GET, PUT, PATCH, and DELETE requests
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer
    
    def destroy(self, request, *args, **kwargs):
        """Delete a receipt"""
        try:
            receipt = self.get_object()
            receipt_id = receipt.id
            receipt.delete()
            logger.info(f"Receipt {receipt_id} deleted successfully")
            return Response(
                {'message': f'Receipt {receipt_id} deleted successfully'}, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error deleting receipt: {str(e)}")
            return Response(
                {'error': f'Delete failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ReceiptUploadAPIView(generics.CreateAPIView):
    """Upload image only - no OCR processing yet"""
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer
    parser_classes = (MultiPartParser, FormParser)

    def create(self, request, *args, **kwargs):
        """Handle receipt image upload"""
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Use service to create receipt
            receipt = ReceiptService.create_receipt_from_upload(request.FILES['image'])
            
            return Response({
                'id': receipt.id,
                'image': receipt.image.url,
                'processed': False,
                'message': 'Image uploaded successfully. Click "Process Receipt" to extract data.'
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error uploading receipt: {str(e)}")
            return Response(
                {'error': f'Upload failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ReceiptProcessAPIView(generics.CreateAPIView):
    """Process receipt with OCR - called when user clicks "Process" button"""
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer

    def create(self, request, *args, **kwargs):
        """Process receipt image with OCR"""
        receipt = self.get_object()
        
        # Validate receipt can be processed
        is_valid, error_message = ReceiptService.validate_receipt_for_processing(receipt)
        if not is_valid:
            return Response(
                {'message': error_message}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Get OCR service and process image
            ocr_service = get_ocr_service()
            image_path = receipt.image.path

            logger.info(f"Processing receipt {receipt.id} with image: {image_path}")

            # Process with OCR with timeout (4 minutes)
            processing_result = {}
            processing_error = None

            def process_with_timeout():
                nonlocal processing_result, processing_error
                try:
                    processing_result = ocr_service.process_receipt_image(image_path)
                except Exception as e:
                    processing_error = e

            processing_thread = threading.Thread(target=process_with_timeout)
            processing_thread.daemon = True
            processing_thread.start()

            # Wait for processing to complete with timeout
            processing_thread.join(timeout=240)  # 4 minutes timeout

            if processing_thread.is_alive():
                logger.error(f"OCR processing timed out for receipt {receipt.id}")
                return Response(
                    {'error': 'Processing timed out. The receipt may be too complex. Please try again.'},
                    status=status.HTTP_408_REQUEST_TIMEOUT
                )

            if processing_error:
                raise processing_error

            receipt_data = processing_result

            # Update receipt with extracted data
            receipt = ocr_service.update_receipt_with_ocr_data(receipt, receipt_data)

            logger.info(f"Receipt {receipt.id} processed successfully")
            return Response(
                ReceiptSerializer(receipt).data,
                status=status.HTTP_200_OK
            )

        except FileNotFoundError as e:
            logger.error(f"Image file not found: {str(e)}")
            return Response(
                {'error': f'Image file not found: {str(e)}'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"OCR processing failed for receipt {receipt.id}: {str(e)}")
            return Response(
                {'error': f'OCR processing failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ReceiptEditAPIView(generics.UpdateAPIView):
    """Edit processed receipt data"""
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer

    def update(self, request, *args, **kwargs):
        """Update receipt fields"""
        receipt = self.get_object()

        # Validate receipt can be edited
        is_valid, error_message = ReceiptService.validate_receipt_for_editing(receipt)
        if not is_valid:
            return Response(
                {'error': error_message},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Use service to update receipt
            receipt = ReceiptService.update_receipt_fields(receipt, request.data)
            return Response(
                ReceiptSerializer(receipt).data, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error updating receipt {receipt.id}: {str(e)}")
            return Response(
                {'error': f'Update failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



class ReceiptBulkDeleteAPIView(generics.GenericAPIView):
    """Bulk delete multiple receipts"""
    queryset = Receipt.objects.all()
    serializer_class = ReceiptSerializer
    
    def post(self, request, *args, **kwargs):
        """Delete multiple receipts by IDs"""
        receipt_ids = request.data.get('ids', [])
        
        if not receipt_ids:
            return Response(
                {'error': 'No receipt IDs provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not isinstance(receipt_ids, list):
            return Response(
                {'error': 'IDs must be a list'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            deleted_count = 0
            failed_ids = []
            
            for receipt_id in receipt_ids:
                try:
                    receipt = Receipt.objects.get(id=receipt_id)
                    receipt.delete()
                    deleted_count += 1
                    logger.info(f"Receipt {receipt_id} deleted successfully")
                except Receipt.DoesNotExist:
                    failed_ids.append(receipt_id)
                    logger.warning(f"Receipt {receipt_id} not found")
                except Exception as e:
                    failed_ids.append(receipt_id)
                    logger.error(f"Error deleting receipt {receipt_id}: {str(e)}")
            
            response_data = {
                'message': f'Successfully deleted {deleted_count} receipt(s)',
                'deleted_count': deleted_count,
                'total_requested': len(receipt_ids)
            }
            
            if failed_ids:
                response_data['failed_ids'] = failed_ids
                response_data['warning'] = f'Failed to delete {len(failed_ids)} receipt(s)'
            
            status_code = status.HTTP_200_OK if deleted_count > 0 else status.HTTP_400_BAD_REQUEST
            return Response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Error in bulk delete: {str(e)}")
            return Response(
                {'error': f'Bulk delete failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RegisterAPIview(generics.CreateAPIView):
    """User registration endpoint"""
    queryset = UserProfileReg.objects.all()
    serializer_class = UserProfileRegSerializer

