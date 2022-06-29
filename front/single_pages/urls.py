from django.urls import path
from . import views

urlpatterns = [
    path('', views.IngredientList.as_view()), # 인덱스 함수 실행
]