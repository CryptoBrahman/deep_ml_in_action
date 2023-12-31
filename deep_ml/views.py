from django.shortcuts import render


def index(request):
    title_proj = 'Hi, deep_ml_learning'
    return render(request, 'deep_ml/index.html',{'title_proj': title_proj})