from django.shortcuts import render
from .models import Recipe, Tag
from django.db import models
from hitcount.views import HitCountDetailView
import pandas as pd
import random
import torch
import time
# from hitcount.models import HitCountMixin
# from hitcount.views import HitCountDetailView
# from .models import Category

class RecipeDetail(HitCountDetailView):
    model=Recipe
    count_hit = True


    def get_context_data(self, **kwargs):
        context=super(RecipeDetail, self).get_context_data(**kwargs)
        
        rid = Recipe.objects.get(pk=self.kwargs.get('pk')).recipe_id          
        cs = torch.load('recipe_Data/cos.pt')[rid].topk(12).indices.tolist()
        # cs = torch.load('/Users/bagjunhyeong/Recipe/recipe_/main_page/static/cos.pt')[rid].topk(12).indices.tolist()
        context['similar'] = Recipe.objects.filter(recipe_id__in = cs)
        return context


def popular(request):    
    recipe_all = Recipe.objects.all().order_by('-count_hit')
    
    r = list(range(100))
    popular = []
    random.shuffle(r)

    for i in [0,3,6,9,12]:
        popular.append([recipe_all[r[i]],
                        recipe_all[r[i+1]],
                        recipe_all[r[i+2]]],
                        )
    
    return render(request, 'main_page/landing.html',
                    {
                        'popular' : popular,
                    })

def search_ings(request):
    if request.method == 'POST':
        try:        
            ingredient = request.POST.get('select_ings') # keyword를 입력받음            
            tag = Tag.objects.get(name = ingredient) # 해당 키워드를 가진 tag 클래스 오픈
            recipes= Recipe.objects.filter(ingredient = tag) # 해당 태그를 가진 post 저장
            
            return render(request, 'main_page/search_ings.html',
                            {
                                'recipes' : recipes, 
                                'ingredient' : ingredient,                                
                            })
        except:
            return render(request, 'main_page/search_ings.html')

    elif request.method == 'GET':
        start = time.time()
        try:        
            ingredient = request.GET.getlist('select_ings') # keyword를 입력받음                                    
            ingredient = ingredient[:-1]
            ingredient = (i.replace(' ','') for i in ingredient)

            tags = []
            for i in ingredient:
                tags.append(Tag.objects.get(name = i))
            print(tags)

            recipes = Recipe.objects.filter(ingredient = tags[0])

            for t in tags[1:]:
                recipes = set(recipes) & set(Recipe.objects.filter(ingredient = t))
                
            recipes = list(recipes)

            return render(request, 'main_page/search_ings.html',
                        {
                            'recipes' : recipes, 
                            'n_recipes' : len(recipes),
                            'ingredient' : ingredient,
                            'time' : round(time.time()-start,4)
                        })
        except:
            
            return render(request, 'main_page/search_ings.html')

def search_recipe(request):
    if request.method == 'POST':
        try:        
            search_keyword = request.POST.getlist('select_recipe') # keyword를 입력받음                                    
            recipes = []
            for i in Recipe.objects.all():
                if len(recipes) >= 10:
                    break
                if search_keyword in i.title:
                    recipes.append(i)            
            return render(request, 'main_page/search_recipe.html',
                            {
                                'recipes' : recipes, 

                            })
        except:
            return render(request, 'main_page/search_recipe.html')

    elif request.method == 'GET':
        start = time.time()

        try:        
            search_keyword = request.GET.get('search_recipe') # keyword를 입력받음            
            recipes = []
            
            for i in Recipe.objects.all():                
                if search_keyword in i.title:
                    recipes.append(i)
            
            return render(request, 'main_page/search_recipe.html',
                            {
                                'search_keyword' : search_keyword,
                                'recipes' : recipes, 
                                'n_recipes' : len(recipes),
                                'time' : round(time.time()-start,4)


                            })
        except:
            return render(request, 'main_page/search_recipe.html')
