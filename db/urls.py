'''
Django's standard place to look for URL pattern matching. See 
https://docs.djangoproject.com/en/dev/topics/http/urls/
for more complete documentation
'''
from django.urls import path

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

from tracker import ajax, views, dbq

urlpatterns = [
    path('', views.main),

    path('setup', views.setup),
    path('setup/add/new_subject', ajax.add_new_subject),
    path('setup/add/new_task', ajax.add_new_task),
    path('setup/add/new_feature', ajax.add_new_feature),
    path('setup/add/new_system', ajax.add_new_system),
    path('setup/populate_models', ajax.populate_models),
    path('setup/add/enable_features', ajax.enable_features),
    path('setup/run_upkeep', ajax.setup_run_upkeep),
    
    path(r'exp_log/', views.list_exp_history, dict(max_entries=200)),
    path(r'exp_log/all/', views.list_exp_history),
    path("exp_log/link_data_files/<int:task_entry_id>", views.link_data_files_view_generator),
    path("exp_log/link_data_files/<int:task_entry_id>/submit", views.link_data_files_response_handler),
    path("exp_log/record_annotation", ajax.record_annotation),
    # path(r'listdb/(?P<dbname>.+?)/.*?/ajax/exp_info/(?P<idx>\d+)/', ajax.exp_info),
    # path(r'listdb/(?P<dbname>.+?)/.*?/ajax/task_info/(?P<idx>\d+)/', ajax.task_info),

    # path(r'listdb/(?P<dbname>.+?)/(?P<subject>.+?)/(?P<task>.+?)', views.listdb),
    # path(r'listdb/(?P<dbname>.+?)/(?P<subject>.+?)', views.listdb),
    # path(r'listdb/(?P<dbname>.+?)/', views.listdb),
    path(r'exp_log/get_status', ajax.get_status),
    path(r'exp_log/report', ajax.get_report),
    path(r'exp_log/ajax/task_info/<int:idx>/', ajax.task_info),
    path(r'exp_log/ajax/exp_info/<int:idx>/', ajax.exp_info),
    path(r'exp_log/ajax/hide_entry/<int:idx>', ajax.hide_entry),    
    path(r'exp_log/ajax/show_entry/<int:idx>', ajax.show_entry),    
    path(r'ajax/backup_entry/<int:idx>/', ajax.backup_entry),
    path(r'ajax/unbackup_entry/<int:idx>/', ajax.unbackup_entry),
    path(r'ajax/gen_info/<int:idx>/', ajax.gen_info),
    path(r'exp_log/ajax/save_notes/<int:idx>/', ajax.save_notes),
    path(r'make_bmi/<int:idx>/?', ajax.train_decoder_ajax_handler),
    path(r'ajax/setattr/<str:attr>/<str:value>', ajax.set_task_attr),

    path(r'start', ajax.start_experiment),
    path(r'test', ajax.start_experiment, dict(save=False)),
    path(r'saverec', ajax.start_experiment, dict(save=True, execute=False)),
    path(r'exp_log/save_entry_name', ajax.save_entry_name),

    path(r'exp_log/stop/', ajax.stop_experiment),
    path(r'enable_clda/', ajax.enable_clda),
    path(r'rewarddrain/<str:onoff>/', ajax.reward_drain),
    path(r'disable_clda/', ajax.disable_clda),
    path(r'sequence_for/<int:idx>/', views.get_sequence),
    path(r'RPC2/', dbq.rpc_handler),
    # Uncomment the next line to enable the admin:
    path(r'admin/', admin.site.urls),
]

