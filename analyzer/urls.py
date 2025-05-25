from django.urls import path
from . import views

urlpatterns = [
   
    path('generate-pdf/', views.generate_pdf, name='generate_pdf'),
    path('', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
   
    path('dashboard/', views.enhanced_dashboard_view, name='dashboard'),
    # ou si vous voulez garder l'ancienne et ajouter une nouvelle
    path('dashboard/enhanced/', views.enhanced_dashboard_view, name='enhanced_dashboard'),
    path('predict/', views.predict, name='predict'),
    path('profile/', views.profile, name='profile'),
 
    path('admin/admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    
    # Messages
    path('admin/messages/', views.admin_messages, name='admin_messages'),
    path('admin/messages/respond/<int:message_id>/', views.admin_message_respond, name='admin_message_respond'),
    path('admin/messages/delete/<int:message_id>/', views.admin_message_delete, name='admin_message_delete'),
    path('admin/messages/delete-multiple/', views.admin_message_delete_multiple, name='admin_message_delete_multiple'),
    
    # Utilisateurs
    path('admin/users/', views.admin_users, name='admin_users'),
    path('admin/users/add/', views.admin_user_add, name='admin_user_add'),
    path('admin/users/edit/<int:user_id>/', views.admin_user_edit, name='admin_user_edit'),
    path('admin/users/delete/<int:user_id>/', views.admin_user_delete, name='admin_user_delete'),
    
    # API
    path('admin/api/message/<int:message_id>/', views.admin_api_message, name='admin_api_message'),
    path('admin/api/user/<int:user_id>/', views.admin_api_user, name='admin_api_user'),
    
    # Paramètres
    path('admin/settings/', views.admin_settings, name='admin_settings'),
    path('logout/', views.logout_view, name='logout'),
   
    path('change-password/', views.change_password, name='change_password'),
     path('messages/', views.user_messages, name='user_messages'),
    path('chatbot-response/', views.chatbot_response, name='chatbot_response'),
    path('chat/', views.chatbot_page, name='chat'),
        # URLs pour la gestion des prédictions par l'admin
    path('admin/predictions/', views.admin_predictions, name='admin_predictions'),
    path('admin/predictions/<int:prediction_id>/delete/', views.admin_prediction_delete, name='admin_prediction_delete'),
    path('admin/predictions/delete-multiple/', views.admin_prediction_delete_multiple, name='admin_prediction_delete_multiple'),
    path('admin/predictions/stats/', views.admin_predictions_stats, name='admin_predictions_stats'),
    
    # API pour récupérer les détails d'une prédiction
    path('admin/api/prediction/<int:prediction_id>/', views.admin_api_prediction, name='admin_api_prediction'),


]


  