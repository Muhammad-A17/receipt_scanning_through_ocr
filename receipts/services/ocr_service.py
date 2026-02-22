"""
Centralized OCR processing service
"""
import os
import logging
import threading
from typing import Optional
from receipt_scanning_through_ocr.ocr_more_lat import EnhancedReceiptParser
from receipts.utils.helpers import map_receipt_data_to_model

logger = logging.getLogger(__name__)


class OCRService:
    """
    Centralized service for OCR processing operations.
    This service handles all OCR-related business logic.
    """
    
    def __init__(self):
        """Initialize the OCR service with parser instance."""
        self.parser = None
        self._initialize_parser()
    
    def _initialize_parser(self):
        """Initialize the OCR parser (lazy loading)."""
        try:
            self.parser = EnhancedReceiptParser()
            logger.info("OCR parser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR parser: {str(e)}")
            raise
    
    def process_and_update_receipt(self, receipt, timeout=240):
        """
        Process a receipt with OCR and update its data in one step.
        Handles threading and timeout internally.
        """
        image_path = receipt.image.path
        processing_result = {}
        processing_error = None

        def process_with_timeout():
            nonlocal processing_result, processing_error
            try:
                processing_result = self.process_receipt_image(image_path)
            except Exception as e:
                processing_error = e

        thread = threading.Thread(target=process_with_timeout)
        thread.daemon = True
        thread.start()

        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError("Processing timed out. The receipt may be too complex. Please try again.")

        if processing_error:
            raise processing_error

        # Update and save receipt
        return self.update_receipt_with_ocr_data(receipt, processing_result)
    
    def process_receipt_image(self, image_path: str):
        """
        Process a receipt image using OCR and extract data.
        
        Args:
            image_path: Full path to the receipt image file
        
        Returns:
            ReceiptData object with extracted information
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image path is invalid
            Exception: If OCR processing fails
        """
        if not image_path:
            raise ValueError("Image path is required")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Processing receipt image: {image_path}")
        
        try:
            # Process the receipt with OCR
            receipt_data = self.parser.processing_receipt(image_path)
            logger.info(f"OCR processing completed successfully for: {image_path}")
            return receipt_data
        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}") from e
    
    def update_receipt_with_ocr_data(self, receipt, receipt_data):
        """
        Update a Receipt model instance with OCR extracted data.
        
        Args:
            receipt: Receipt model instance
            receipt_data: ReceiptData object from OCR parser
        
        Returns:
            Updated Receipt model instance
        """
        receipt = map_receipt_data_to_model(receipt, receipt_data)
        receipt.save()
        logger.info(f"Receipt {receipt.id} updated with OCR data")
        return receipt


# Singleton instance
_ocr_service_instance = None


def get_ocr_service() -> OCRService:
    """
    Get the singleton OCR service instance.
    
    Returns:
        OCRService instance
    """
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = OCRService()
    return _ocr_service_instance



