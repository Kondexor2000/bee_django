from django.contrib import admin
from .models import Bee

class BeeAdmin(admin.ModelAdmin):
    list_display = ['is_bee', 'user', 'image', 'image_download_link']

    def image_download_link(self, obj):
        if obj.image:
            return '<a href="{0}" download>Download</a>'.format(obj.image.url)
        else:
            return 'No Image'

    image_download_link.allow_tags = True
    image_download_link.short_description = 'Image Download'

admin.site.register(Bee, BeeAdmin)