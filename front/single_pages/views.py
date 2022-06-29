from django.shortcuts import render
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from .models import Ingredient

class IngredientList(ListView):
    model = Ingredient
    
    # ordering='-pk'
        
    def get_context_data(self, **kwargs):
        context=super(IngredientList,self).get_context_data()
        user = self.request.user

        ingredients_old = Ingredient.objects.filter(author=user)

        n = 5

        ingredients_new = [ingredients_old[i * n:(i + 1) * n] for i in range((len(ingredients_old) + n - 1) // n )] 

        context['ingredients'] = ingredients_new

        return context
