from django.contrib import admin
from .models import Bee

# Register your models here.

class BeeAdmin(admin.ModelAdmin):
    list_display = ['is_bee', 'user']


admin.site.register(Bee, BeeAdmin)