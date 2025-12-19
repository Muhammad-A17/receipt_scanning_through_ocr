from django.shortcuts import redirect, render
from django.contrib import messages
from .models import Receipt
from .forms import ReceiptUploadForm
import os
import sys

sys.path.append('/home/magomed-ameen/programming/pract/receipt_scanning_through_ocr')
#from ocr_bigger_upgrade import EnhancedReceiptParser
# Create your views here.
def home(request):
    receipts=Receipt.objects.all().order_by('-created_at')
    return render(request,'receipts/home.html',{'receipts': receipts})
def upload_receipt(request):
    if request.method == 'POST':
        form = ReceiptUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            receipt = form.save()
            messages.success(request, 'Receipt uploaded successfully!')
            return redirect('home')
    else:
        form = ReceiptUploadForm()
    
    return render(request, 'receipts/upload.html', {'form': form})

def receipt_detail(request,receipt_id):
    receipt = Receipt.objects.get(id=receipt_id)
    return render(request, 'receipts/detail.html', {'receipt': receipt})



