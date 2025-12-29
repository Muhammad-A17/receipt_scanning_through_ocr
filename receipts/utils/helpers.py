"""
Helper utility functions for data conversion and validation
"""
from decimal import Decimal, InvalidOperation
from datetime import datetime


def to_decimal_or_none(value):
    """
    Convert a value to Decimal or return None if conversion fails.
    
    Args:
        value: Value to convert (can be string, int, float, Decimal, or None)
    
    Returns:
        Decimal or None
    """
    if value in (None, '', 'NaN', 'Infinity', '-Infinity'):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def date_handler(date_str):
    """
    Validate and return a date string in YYYY-MM-DD format.
    
    Args:
        date_str: Date string to validate
    
    Returns:
        Valid date string in YYYY-MM-DD format or None
    """
    if not date_str or date_str in ('None', '', 'NaN'):
        return None
    try:
        if len(date_str) == 10 and date_str.count('-') == 2:
            # Validate the date format
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        else:
            return None
    except (ValueError, AttributeError):
        return None


def map_receipt_data_to_model(receipt_model, receipt_data):
    """
    Map OCR extracted receipt data to Receipt model instance.
    
    Args:
        receipt_model: Receipt model instance to update
        receipt_data: ReceiptData object from OCR parser
    
    Returns:
        Updated Receipt model instance
    """
    
    receipt_model.merchant_name = receipt_data.merchant_name
    receipt_model.merchant_address = receipt_data.merchant_address
    receipt_model.merchant_phone = receipt_data.merchant_phone
    receipt_model.merchant_email = receipt_data.merchant_email
    receipt_model.date = date_handler(receipt_data.date)
    receipt_model.time = receipt_data.time
    receipt_model.transaction_id = receipt_data.transaction_id
    receipt_model.receipt_number = receipt_data.receipt_number
    receipt_model.tip = to_decimal_or_none(receipt_data.tip)
    receipt_model.tax = to_decimal_or_none(receipt_data.tax)
    receipt_model.sub_total = to_decimal_or_none(receipt_data.sub_total)
    receipt_model.total = to_decimal_or_none(receipt_data.total)
    receipt_model.discount = to_decimal_or_none(receipt_data.discount)
    receipt_model.items = receipt_data.items
    receipt_model.payment_method = receipt_data.payment_method
    receipt_model.card_type = receipt_data.card_type
    receipt_model.card_last_four = receipt_data.card_last_four
    receipt_model.category = receipt_data.category
    receipt_model.tax_rate = to_decimal_or_none(receipt_data.tax_rate)
    receipt_model.currency = receipt_data.currency
    receipt_model.confidence_scores = receipt_data.confidence_scores
    receipt_model.processed = True
    
    return receipt_model

