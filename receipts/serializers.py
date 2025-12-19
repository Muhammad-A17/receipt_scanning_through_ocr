from django.contrib.auth.password_validation import password_changed
from rest_framework import serializers
from .models import Receipt, UserProfileReg

class ReceiptSerializer(serializers.ModelSerializer):#Automatically generates serializers for model
    class Meta:
        model=Receipt#the model to serialize
        fields='__all__'#include all fields from the 
        
class UserProfileRegSerializer(serializers.ModelSerializer):
    password=serializers.CharField(write_only=True)
    password2=serializers.CharField(write_only=True)
    class Meta:
        model=UserProfileReg
        fields=['first_name','last_name','email','password','password2','phone_number']
        extra_kwargs={'first_name':{'required':True},
        'last_name':{'required':True},
        'email':{'required':True},
        'phone_number':{'required':False}}
    
    def create(self,validated_data):
        user=UserProfileReg.objects.create_user(first_name=validated_data['first_name'],
        last_name=validated_data['last_name'],
        email=validated_data['email'],
        password=validated_data['password'],
        phone_number=validated_data.get('phone_number',None))
        return user