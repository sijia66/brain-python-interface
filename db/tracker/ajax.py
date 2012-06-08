import json
import cPickle
import xmlrpclib

import numpy as np
from django.http import HttpResponse

from riglib import experiment

import namelist
from json_param import Parameters
from tasktrack import Track
from models import TaskEntry, Feature, Sequence, Task, Generator, Subject


display = Track()

class encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Parameters):
            return o.params

        return super(encoder, self).default(o)

def _respond(data):
    return HttpResponse(json.dumps(data, cls=encoder), mimetype="application/json")

def task_info(request, idx):
    task = Task.objects.get(pk=idx)
    feats = [Feature.objects.get(name=name) for name, isset in request.GET.items() if isset == "true"]
    task_info = dict(params=task.params(feats=feats))

    if issubclass(task.get(feats=feats), experiment.Sequence):
        task_info['sequence'] = task.sequences()

    return _respond(task_info)

def exp_info(request, idx):
    entry = TaskEntry.objects.get(pk=idx)
    return _respond(entry.to_json())

def gen_info(request, idx):
    gen = Generator.objects.get(pk=idx)
    return _respond(gen.to_json())

def start_experiment(request, save=True):
    #make sure we don't have an already-running experiment
    if display.status.value != '':
        return _respond(dict(status="error", msg="Alreading running task!"))

    try:
        data = json.loads(request.POST['data'])
        task =  Task.objects.get(pk=data['task'])
        Exp = task.get(feats=data['feats'].keys())
        entry = TaskEntry(subject_id=data['subject'], task=task)
        params = Parameters.from_html(data['params'])
        params.trait_norm(Exp.class_traits())
        entry.params = params.to_json()
        kwargs = dict(subj=entry.subject, task=task, feats=Feature.getall(data['feats'].keys()),
                      params=params.params)

        if issubclass(Exp, experiment.Sequence):
            seq = Sequence.from_json(data['sequence'])
            seq.task = task
            if save:
                seq.save()
            entry.sequence = seq
            kwargs['seq'] = seq
        else:
            entry.sequence_id = -1
        
        response = dict(status="testing", subj=entry.subject.name, task=entry.task.name)
        if save:
            entry.save()
            for feat in data['feats'].keys():
                f = Feature.objects.get(pk=feat)
                entry.feats.add(f.pk)
            response['date'] = entry.date.strftime("%h %d, %Y %I:%M %p")
            response['status'] = "running"
            response['idx'] = entry.id
            kwargs['saveid'] = entry.id
        
        display.runtask(**kwargs)
        return _respond(response)

    except Exception as e:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))


def stop_experiment(request):
    #make sure that there exists an experiment to stop
    if display.status.value not in ["running", "testing"]:
        return _respond(dict(status="error", msg="No task to end!"))
    try:
        status = display.status.value
        display.stoptask()
        return _respond(dict(status="pending", msg=status))
    except:
        import cStringIO
        import traceback
        err = cStringIO.StringIO()
        traceback.print_exc(None, err)
        err.seek(0)
        return _respond(dict(status="error", msg=err.read()))

def save_notes(request, idx):
    te = TaskEntry.objects.get(pk=idx)
    te.notes = request.POST['notes']
    te.save()
    return _respond(dict(status="success"))
