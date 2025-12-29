from django.urls import path
from . import api_views

urlpatterns=[
    path('receipts/',api_views.ReceiptListAPIView.as_view(),name='api-receipt-list'),
    path('receipts/<int:pk>/',api_views.ReceiptDetailAPIView.as_view(),name='api-receipt-detail'),
    path('receipts/upload/',api_views.ReceiptUploadAPIView.as_view(),name='api-receipt-upload'),
    path('receipts/<int:pk>/process/',api_views.ReceiptProcessAPIView.as_view(),name='api-receipt-process'),
    path('receipts/<int:pk>/edit/',api_views.ReceiptEditAPIView.as_view(),name='api-receipt-edit'),
    path('receipts/bulk-delete/',api_views.ReceiptBulkDeleteAPIView.as_view(),name='api-receipt-bulk-delete'),
    path('register',api_views.RegisterAPIview.as_view(),name='api-register') #new
]
