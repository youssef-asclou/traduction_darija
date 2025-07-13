# models.py
from django.db import models

class Translation(models.Model):
    darija_text = models.TextField()
    english_translation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Translation: {self.darija_text[:50]}..."
