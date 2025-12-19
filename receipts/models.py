from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.base_user import BaseUserManager
from django.utils import timezone

class Receipt(models.Model):
    merchant_name=models.CharField(max_length=255,blank=True,null=True)
    merchant_address=models.TextField(max_length=255,blank=True,null=True)
    merchant_phone=models.CharField(max_length=20,blank=True,null=True)
    merchant_email=models.EmailField(blank=True,null=True)
    date=models.DateField(blank=True,null=True)
    time=models.TimeField(blank=True,null=True)

    transaction_id=models.CharField(max_length=255,blank=True,null=True)
    receipt_number=models.CharField(max_length=100,blank=True,null=True)

    tip=models.DecimalField(max_digits=20,decimal_places=2,blank=True,null=True)
    
    tax=models.DecimalField(max_digits=20,decimal_places=2,blank=True,null=True)
    sub_total=models.DecimalField(max_digits=20,decimal_places=2,blank=True,null=True)

    total=models.DecimalField(max_digits=20,decimal_places=2,blank=True,null=True)
    discount=models.DecimalField(max_digits=20,decimal_places=2,blank=True,null=True)

    items=models.JSONField(blank=True, null=True)

    payment_method=models.CharField(max_length=255,blank=True,null=True)
    card_type=models.CharField(max_length=150,blank=True,null=True)
    card_last_four=models.CharField(max_length=4,blank=True,null=True)

    category=models.CharField(max_length=255,blank=True,null=True)
    tax_rate=models.DecimalField(max_digits=5,decimal_places=2,blank=True,null=True)
    currency=models.CharField(max_length=3,blank=True,null=True)

    confidence_scores=models.JSONField(blank=True,null=True)

    image=models.ImageField(upload_to='receipts/',blank=True,null=True)
    created_at=models.DateTimeField(default=timezone.now)
    processed=models.BooleanField(default=False)

class UserProfileRegManager(BaseUserManager):
    def create_user(self,email,password,**extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email=self.normalize_email(email)
        user=self.model(email=email,**extra_fields)
        user.set_password(password)
        user.save()
        return user
    
    def create_superuser(self,email,password,**extra_fields):
        extra_fields.setdefault('is_staff',True)
        extra_fields.setdefault('is_superuser',True)
        extra_fields.setdefault('is_active',True)
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        return self.create_user(email,password,**extra_fields)

class UserProfileReg(AbstractUser):
    username=None
    
    email=models.EmailField(unique=True,null=False,blank=False)
    phone_number=models.CharField(max_length=20,blank=True,null=True)

    USERNAME_FIELD='email'
    REQUIRED_FIELDS=['first_name','last_name']
    
    objects = UserProfileRegManager()


