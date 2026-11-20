"""Phone-controlled pi-top robot.  Run:  ROBOT_PIN=1234 python3 app.py
Then open  http://<pi-top-ip>:5000  on your phone (same Wi-Fi)."""
import atexit
import hmac
import os

from flask import Flask, Response, jsonify, request

from follower import Follower
from robot import Robot

PIN = os.environ.get("ROBOT_PIN", "1234")

app = Flask(__name__)
robot = Robot()
follower = Follower(robot)


@atexit.register
def _shutdown():
    follower.close()
    robot.close()


@app.before_request
def _auth():
    if request.path == "/":
        return None
    given = request.headers.get("X-Pin") or request.args.get("pin", "")
    if not hmac.compare_digest(given, PIN):
        return jsonify(error="bad pin"), 401
    return None


def _body():
    return request.get_json(silent=True) or {}


@app.post("/api/drive")
def api_drive():
    if follower.mode == "manual":
        d = _body()
        fwd, turn = float(d.get("forward", 0)), float(d.get("turn", 0))
        robot.set_drive(0 if abs(fwd) < 0.1 else fwd, 0 if abs(turn) < 0.1 else turn)
    return jsonify(ok=True)


@app.post("/api/look")
def api_look():
    if follower.mode == "manual":
        d = _body()
        robot.look(float(d.get("pan", 0)), float(d.get("tilt", 0)))
    return jsonify(ok=True)


@app.post("/api/mode")
def api_mode():
    return jsonify(ok=follower.set_mode(_body().get("mode", "")))


@app.post("/api/stop")
def api_stop():
    follower.set_mode("manual")
    robot.stop()
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    return jsonify(follower.status())


@app.get("/video")
def video():
    return Response(follower.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")


HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>pi-top robot</title>
<style>
  body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;touch-action:none;
       -webkit-user-select:none;user-select:none}
  #cam{width:100%;display:block;background:#000;aspect-ratio:4/3}
  .row{display:flex;gap:8px;padding:8px}
  button{flex:1;padding:14px 4px;border:0;border-radius:10px;background:#2a2a2a;color:#eee;font-size:16px}
  button.on{background:#2b7de9}
  #stop{background:#d9342b;font-weight:700}
  #pad{width:200px;height:200px;border-radius:50%;background:#222;border:2px solid #444;
       margin:8px auto;position:relative;touch-action:none}
  #knob{width:80px;height:80px;border-radius:50%;background:#2b7de9;position:absolute;
        left:60px;top:60px;pointer-events:none}
  label{display:block;padding:2px 12px;font-size:13px;color:#aaa}
  input[type=range]{width:92%;margin:0 4%}
  #status{padding:6px 12px;font-size:13px;color:#9c9}
</style></head><body>
<img id="cam" alt="camera">
<div id="status">connecting...</div>
<div class="row">
  <button data-mode="manual">Manual</button>
  <button data-mode="look">Look at me</button>
  <button data-mode="follow">Follow me</button>
</div>
<div class="row"><button id="stop">STOP</button></div>
<div id="pad"><div id="knob"></div></div>
<label>Camera pan</label><input id="pan" type="range" min="-80" max="80" value="0">
<label>Camera tilt</label><input id="tilt" type="range" min="-50" max="50" value="0">
<div class="row"><button id="center">Center camera</button></div>
<script>
const $=id=>document.getElementById(id);
let pin=localStorage.getItem('pin')||prompt('Robot PIN');
localStorage.setItem('pin',pin||'');
let mode='manual';
async function api(path,body){
  try{
    const r=await fetch(path,{method:body?'POST':'GET',
      headers:{'Content-Type':'application/json','X-Pin':pin},
      body:body?JSON.stringify(body):undefined});
    if(r.status===401){localStorage.removeItem('pin');location.reload();return}
    return await r.json();
  }catch(e){}
}
$('cam').src='/video?pin='+encodeURIComponent(pin);

// joystick
const pad=$('pad'),knob=$('knob');let jx=0,jy=0;
function mv(e){
  const r=pad.getBoundingClientRect(),R=r.width/2;
  let dx=(e.clientX-r.left-R)/R,dy=(e.clientY-r.top-R)/R;
  const m=Math.hypot(dx,dy);if(m>1){dx/=m;dy/=m}
  jx=dx;jy=-dy;
  knob.style.transform='translate('+(dx*R*0.6)+'px,'+(dy*R*0.6)+'px)';
}
pad.onpointerdown=e=>{pad.setPointerCapture(e.pointerId);mv(e)};
pad.onpointermove=e=>{if(pad.hasPointerCapture(e.pointerId))mv(e)};
pad.onpointerup=pad.onpointercancel=()=>{jx=jy=0;knob.style.transform=''};
setInterval(()=>{if(mode==='manual')api('/api/drive',{forward:jy,turn:jx})},100);

// camera sliders
function sendLook(){api('/api/look',{pan:+$('pan').value,tilt:+$('tilt').value})}
$('pan').oninput=sendLook;$('tilt').oninput=sendLook;
$('center').onclick=()=>{$('pan').value=0;$('tilt').value=0;sendLook()};

// modes
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>api('/api/mode',{mode:b.dataset.mode}));
$('stop').onclick=()=>api('/api/stop',{});

// status
setInterval(async()=>{
  const s=await api('/api/status');if(!s)return;
  mode=s.mode;
  document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('on',b.dataset.mode===mode));
  if(mode!=='manual'){$('pan').value=s.pan;$('tilt').value=s.tilt}
  const f=v=>v==null?'--':v+' m';
  $('status').textContent='mode: '+mode+'  |  front '+f(s.front)+'  back '+f(s.back)+
    '  |  face: '+(s.face?'yes ('+s.face_size+')':'no');
},500);
</script></body></html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
