from django.urls import path
from . import views

urlpatterns = [
    path('', views.popular), # 인덱스 함수 실행
    path('recipe/<int:pk>/', views.RecipeDetail.as_view()),
    path('search_ing', views.search_ings),
    path('search_recipe', views.search_recipe)

]