from django.contrib import admin

from .models import Recipe, Tag, Category1, Category2, Category3, Category4

admin.site.register(Recipe)

class CategoryAdmin(admin.ModelAdmin):
    prepopulated_fields = {'slug':('name',)}

admin.site.register(Category1, CategoryAdmin)
admin.site.register(Category2, CategoryAdmin)
admin.site.register(Category3, CategoryAdmin)
admin.site.register(Category4, CategoryAdmin)

class TagAdmin(admin.ModelAdmin):
    prepopulated_fields = {'slug':('name',)}

admin.site.register(Tag, TagAdmin)