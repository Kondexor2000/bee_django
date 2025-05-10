"""
URL configuration for bee project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.contrib import admin
from django.conf.urls.static import static
from django.urls import path
from beeapp import views
from beeapp.views import CustomLoginView, CustomLogoutView, SignUpView, EditProfileView, DeleteAccountView

urlpatterns = [
    path('signup/', SignUpView.as_view(), name='signup'),
    path('edit-profile/', EditProfileView.as_view(), name='edit_profile'),
    path('delete-account/', DeleteAccountView.as_view(), name='delete_account'),
    path('admin/', admin.site.urls),
    path('', CustomLoginView.as_view(), name='login'),
    path('logout/', CustomLogoutView.as_view(), name='logout'),
    path('detect/', views.detect_bee, name='detect_bee'),
    path('result/', views.display_results, name='display_results'),
    path('image/', views.process_image, name='process_image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
