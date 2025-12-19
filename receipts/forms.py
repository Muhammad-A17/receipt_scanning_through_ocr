from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.forms import widgets
from .models import Receipt, UserProfileReg


class ReceiptUploadForm(forms.ModelForm):
    class Meta:#special section that tells django ho# to configure the form
        model=Receipt#uses receipt model
        fields=['image']#Ony include  the image field in this form, means teh form will only show the image file upload file
        widgets={'image': forms.FileInput(attrs={'class':'form-control','accept':'image/*'})}#customize ho the form fields look and behave
        #'image': forms.FileInput make the image field a file upload input
        #attrs={'class':'form-control   html attributes, this one is for form styling
        #'accept':'image/*'    only accept image files to be selected
        
class RegisterForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model=UserProfileReg
        fields=['first_name','last_name','email','phone_number']#password1 and password2 are handled by the parent class

    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)#because we are overriding the init method, some customizations are lost or added to the form, so we need to call the parent class init method to keep the customizations

        for field_name in self.fields:#css applied
            self.fields[field_name].widget.attrs['class']='form-control'
    
