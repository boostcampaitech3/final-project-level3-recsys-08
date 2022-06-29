from django.db import models
from django.contrib.auth.models import User
from markdownx.models import MarkdownxField
from markdownx.utils import markdown
from hitcount.models import HitCountMixin, Hit
from django.contrib.admin.models import LogEntry
import time

# Create your models here.
class Tag(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=200, unique=True, allow_unicode=True)
    
    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return f'/tag/{self.slug}/'

class Category1(models.Model):    
    name=models.CharField(max_length=50, unique=True)
    slug=models.SlugField(max_length=200, unique=True, allow_unicode=True)

    def __str__(self):
        return self.name

    class Meta:        
        verbose_name_plural='Categories1'

class Category2(models.Model):    
    name=models.CharField(max_length=50, unique=True)
    slug=models.SlugField(max_length=200, unique=True, allow_unicode=True)

    def __str__(self):
        return self.name

    class Meta:        
        verbose_name_plural='Categories2'

class Category3(models.Model):    
    name=models.CharField(max_length=50, unique=True)
    slug=models.SlugField(max_length=200, unique=True, allow_unicode=True)

    def __str__(self):
        return self.name

    class Meta:        
        verbose_name_plural='Categories3'

class Category4(models.Model):    
    name=models.CharField(max_length=50, unique=True)
    slug=models.SlugField(max_length=200, unique=True, allow_unicode=True)

    def __str__(self):
        return self.name

    class Meta:        
        verbose_name_plural='Categories4'


class Recipe(models.Model, HitCountMixin):
    recipe_id = models.IntegerField()    
    title = models.CharField(max_length=50)
    hook = models.TextField()
    content = MarkdownxField()
    img_url = models.TextField()
    count_hit = models.IntegerField()
    ingredient = models.ManyToManyField(Tag, blank=True)
    
    category1 =models.ForeignKey(Category1, null=True, blank=True, on_delete=models.SET_NULL)
    category2 =models.ForeignKey(Category2, null=True, blank=True, on_delete=models.SET_NULL)
    category3 =models.ForeignKey(Category3, null=True, blank=True, on_delete=models.SET_NULL)
    category4 =models.ForeignKey(Category4, null=True, blank=True, on_delete=models.SET_NULL)

    serving = models.TextField() # 인원수
    elapsed_time = models.TextField() # 소요시간
    grade = models.TextField() # 난이도
    
    
    def get_absolute_url(self):
        return f'/recipe/{self.pk}/'

    def get_content_markdown(self):
        return markdown(self.content,extensions=['tables'])

    def __str__(self):
        return f'({self.pk})-{self.title}'




