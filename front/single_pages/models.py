from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Ingredient(models.Model):
    author=models.ForeignKey(User,null=True,on_delete=models.SET_NULL)
    ingred = models.CharField(max_length=30)
    count = models.IntegerField() # ??
    created_at = models.DateTimeField(auto_now_add = True)
    updated_at = models.DateTimeField(auto_now = True)

    def __str__(self):
        return f'({self.pk})-{self.ingred}-{self.count}개'

    def get_ingred(self):
        return self.ingred