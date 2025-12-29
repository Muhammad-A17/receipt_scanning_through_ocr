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
        Update receipt fields from request data.
        
        Args:
            receipt: Receipt instance to update
            data: Dictionary of field values
        
        Returns:
            Updated Receipt instance
        """
        # Merchant information
        receipt.merchant_name = data.get('merchant_name', receipt.merchant_name)
        receipt.merchant_address = data.get('merchant_address', receipt.merchant_address)
        receipt.merchant_phone = data.get('merchant_phone', receipt.merchant_phone)
        receipt.merchant_email = data.get('merchant_email', receipt.merchant_email)
        
        # Transaction information
        receipt.date = data.get('date', receipt.date)
        receipt.time = data.get('time', receipt.time)
        receipt.transaction_id = data.get('transaction_id', receipt.transaction_id)
        receipt.receipt_number = data.get('receipt_number', receipt.receipt_number)
        
        # Financial information
        receipt.tip = to_decimal_or_none(data.get('tip', receipt.tip))
        receipt.tax = to_decimal_or_none(data.get('tax', receipt.tax))
        receipt.sub_total = to_decimal_or_none(data.get('sub_total', receipt.sub_total))
        receipt.total = to_decimal_or_none(data.get('total', receipt.total))
        receipt.discount = to_decimal_or_none(data.get('discount', receipt.discount))
        
        # Items and payment
        receipt.items = data.get('items', receipt.items)
        receipt.payment_method = data.get('payment_method', receipt.payment_method)
        receipt.card_type = data.get('card_type', receipt.card_type)
        receipt.card_last_four = data.get('card_last_four', receipt.card_last_four)
        
        # Additional fields
        receipt.category = data.get('category', receipt.category)
        receipt.tax_rate = to_decimal_or_none(data.get('tax_rate', receipt.tax_rate))
        receipt.currency = data.get('currency', receipt.currency)
        
        receipt.save()
        logger.info(f"Receipt {receipt.id} fields updated")
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

