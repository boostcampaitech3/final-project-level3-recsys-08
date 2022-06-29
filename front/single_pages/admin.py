from django.contrib import admin
from .models import Ingredient

# class IngredientAdmin(admin.ModelAdmin):
#     prepopulated_fields = {'slug':('name',)}

admin.site.register(Ingredient)
