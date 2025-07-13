from django.urls import path
from . import views

urlpatterns = [
#    path('', views.home, name='home'),
    path('', views.traducteur_darija, name='traducteur'),
    #path('login/', views.login_view, name='login'),  # Login
    #path('signup/', views.signup_view, name='signup'),  # Signup
]
