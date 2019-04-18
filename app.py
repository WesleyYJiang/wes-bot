from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from flask import Flask, render_template, make_response
import torch
import re
import json
from flask_socketio import SocketIO
from Model.Voc import Voc
import Model.util as util

device = torch.device("cpu")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

voc = Voc("data")
with open('voc6.json') as f:
    v = json.load(f)
d = v["index2word"]
d = {int(k):v for k,v in d.items()}
v["index2word"] = d
voc.__dict__ = v

a = torch.jit.load("chatbot_v6.pth", map_location=torch.device('cpu'))

@app.route('/')
def sessions():
    r = make_response(render_template('session.html'))
    r.headers.set('Access-Control-Allow-Origin', '*')
    r.headers.set("Access-Control-Allow-Origin", "*")
    r.headers.set("Access-Control-Allow-Headers", "X-Requested-With")
    r.headers.set("Access-Control-Allow-Headers", "Content-Type")
    r.headers.set("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    r.headers.set("Connection", "Upgrade")
    r.headers.set("Upgrade", "websocket")
    return r


def messageReceived():
    print('message was received!!!')


@socketio.on('my event')
def handle_my_custom_event(json):
    print('received my event: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)
    if "message" in json:
        response = util.evaluateInput(a.encoder, a.decoder, a, voc, json["message"])
        socketio.emit('my response', response)


if __name__ == '__main__':
    socketio.run(app)