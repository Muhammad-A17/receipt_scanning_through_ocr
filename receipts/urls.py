from django.urls import path
from . import views

urlpatterns= [
    path('',views.home,name='home'),#empty '' string means the home vie or the root url of the site, vies.home runs the home function hen 
    #someone visits root, home is the url name
    path('upload/',views.upload_receipt,name='upload'),
    path('receipt/<int:receipt_id>/',views.receipt_detail,name='receipt_data'),#url ith a number like sit.com/receipt/123/
    #captures the number and call it receipt id,   views.receipt_detail   runs the receipt_detail function when someone visits this url
    #path('register/',views.register,name='register'), front end is handling it so not needed in html form

]