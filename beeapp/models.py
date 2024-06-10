from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Bee(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    is_bee = models.BooleanField(default=False)
    image = models.ImageField(upload_to='bee_images/')
    predictions = models.TextField(null=True, blank=True)