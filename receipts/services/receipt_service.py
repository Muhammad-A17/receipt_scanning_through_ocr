"""
Centralized receipt business logic service
"""
import logging
from typing import Optional, Dict, Any, Tuple
from decimal import Decimal
from receipts.models import Receipt
from receipts.utils.helpers import to_decimal_or_none

logger = logging.getLogger(__name__)


class ReceiptService:
    """
    Centralized service for receipt-related business logic.
    This service handles all receipt operations.
    """
    
    @staticmethod
    def create_receipt_from_upload(image_file) -> Receipt:
        """
        Create a new receipt instance from uploaded image.
        
        Args:
            image_file: Uploaded image file
        
        Returns:
            Receipt instance
        """
        receipt = Receipt(image=image_file)
        receipt.save()
        logger.info(f"Receipt {receipt.id} created from image upload")
        return receipt
    
    @staticmethod
    def update_receipt_fields(receipt: Receipt, data: Dict[str, Any]) -> Receipt:
        """
        Update receipt fields from request data using automated mapping.
        """
        # Define fields and their types (defaulting to string/pass-through)
        fields_to_map = [
            'merchant_name', 'merchant_address', 'merchant_phone', 'merchant_email',
            'date', 'time', 'transaction_id', 'receipt_number',
            'items', 'payment_method', 'card_type', 'card_last_four',
            'category', 'currency'
        ]
        
        decimal_fields = ['tip', 'tax', 'sub_total', 'total', 'discount', 'tax_rate']

        # Map standard fields
        for field in fields_to_map:
            if field in data:
                setattr(receipt, field, data.get(field))
        
        # Map decimal fields
        for field in decimal_fields:
            if field in data:
                setattr(receipt, field, to_decimal_or_none(data.get(field)))
        
        receipt.save()
        logger.info(f"Receipt {receipt.id} fields updated automatically")
        return receipt
    
    @staticmethod
    def validate_receipt_for_processing(receipt: Receipt) -> Tuple[bool, Optional[str]]:
        """
        Validate if a receipt can be processed.
        
        Args:
            receipt: Receipt instance to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if receipt.processed:
            return False, "Receipt has already been processed"
        
        if not receipt.image:
            return False, "Receipt image is missing"
        
        return True, None
    
    @staticmethod
    def validate_receipt_for_editing(receipt: Receipt) -> Tuple[bool, Optional[str]]:
        """
        Validate if a receipt can be edited.
        
        Args:
            receipt: Receipt instance to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not receipt.processed:
            return False, "Receipt must be processed before editing"
        
        return True, None

