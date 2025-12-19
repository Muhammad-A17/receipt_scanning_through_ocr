from django.contrib import admin
from .models import Receipt

@admin.register(Receipt)
class ReceiptAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'merchant_name', 'total', 'sub_total', 'tax', 'tip',
        'payment_method', 'card_type', 'card_last_four', 'date', 'created_at', 'processed'
    )
    list_filter = ('processed', 'payment_method', 'card_type', 'created_at', 'date', 'category')
    search_fields = (
        'merchant_name', 'merchant_address', 'merchant_phone', 'merchant_email',
        'transaction_id', 'receipt_number', 'category', 'currency'
    )
    readonly_fields = ('created_at', 'processed', 'confidence_scores_display')

    fieldsets = (
        ('Merchant', {
            'fields': ('merchant_name', 'merchant_address', 'merchant_phone', 'merchant_email', 'category')
        }),
        ('Transaction', {
            'fields': ('date', 'time', 'transaction_id', 'receipt_number', 'currency')
        }),
        ('Financials', {
            'fields': ('sub_total', 'tax', 'tax_rate', 'tip', 'discount', 'total')
        }),
        ('Payment', {
            'fields': ('payment_method', 'card_type', 'card_last_four')
        }),
        ('Data', {
            'fields': ('items', 'confidence_scores_display', 'image', 'created_at',)
        }),
    )

    def confidence_scores_display(self, obj):
        if not obj.confidence_scores:
            return '-'
        # compact preview for admin
        from json import dumps
        return dumps(obj.confidence_scores, ensure_ascii=False)
    confidence_scores_display.short_description = 'Confidence Scores'
