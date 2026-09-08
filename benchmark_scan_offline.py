"""Offline saved-video experiment. No API imports, credentials, or network calls."""
import ast,json,time,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import math, argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("video",type=Path)
parser.add_argument("model",type=Path,help="Existing u2net ONNX weights; never downloaded")
parser.add_argument("--source",type=Path,default=Path("video_scan.py"))
parser.add_argument("--output",type=Path,default=Path("benchmark-output"))
args=parser.parse_args()
video_hash=hashlib.sha256(args.video.read_bytes()).hexdigest()
model_hash=hashlib.sha256(args.model.read_bytes()).hexdigest()
source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
cv2.setNumThreads(2)
root=args.output;out=root/(video_hash[:12]+'-'+model_hash[:12]+'-'+source_hash[:12]);out.mkdir(parents=True,exist_ok=True)
names={'_frame_cutout','_local_features','_geometric_duplicate','_foreground_palette','_palette_matches'}
env={'cv2':cv2,'Image':Image,'math':math}
exec(compile(ast.Module(body=[n for n in ast.parse(args.source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'actual-code','exec'),env)
net=None
cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);interval=max(1,int(fps*.5))
if not cap.isOpened() or not 0<fps<=240 or cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps>20:
 raise ValueError("Requires readable video up to 20 seconds with valid FPS")
rows=[];idx=0;start=time.monotonic()
while True:
 ok,f=cap.read()
 if not ok:break
 current=idx;idx+=1
 if idx>int(fps*20)+1:raise ValueError("Video exceeds 20 seconds")
 if current%interval:continue
 sharp=float(cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
 if sharp<22:continue
 t=current/fps;path=out/f'{current:04}.png';beg=time.monotonic()
 if not path.exists():
  if net is None:net=cv2.dnn.readNetFromONNX(str(args.model))
  original=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
  a=np.asarray(original.resize((320,320),Image.Resampling.LANCZOS));a=a/max(np.max(a),1e-6)
  a=(a-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
  net.setInput(a.transpose(2,0,1)[None].astype('float32'));pred=net.forward(net.getUnconnectedOutLayersNames())[0][0,0]
  pred=(pred-pred.min())/max(float(pred.max()-pred.min()),1e-6)
  mask=Image.fromarray((pred.clip(0,1)*255).astype('uint8')).resize(original.size,Image.Resampling.LANCZOS)
  cut=Image.composite(original.convert('RGBA'),Image.new('RGBA',original.size,0),mask)
  env['_frame_cutout'](cut).save(path)
 rembg_seconds=time.monotonic()-beg
 cut=Image.open(path);features=env['_local_features'](cut);palette=env['_foreground_palette'](cut)
 rows.append(dict(time=t,sharpness=sharp,path=str(path),rembg_seconds=rembg_seconds,features=features,palette=palette))
 print('cutout',t,round(rembg_seconds,2),flush=True)
cap.release();groups=[];pairs=[]
for row in rows:
 matched=None
 for g in reversed(groups):
  if row['time']-g['last']>3:continue
  a=g['anchor'];color=env['_palette_matches'](row['palette'],a['palette']);geo=color and env['_geometric_duplicate'](row['features'],a['features'])
  pairs.append({'a':a['time'],'b':row['time'],'palette':color,'geometry':bool(geo)})
  if geo:matched=g;break
 if matched:
  matched['times'].append(row['time']);matched['last']=row['time']
  if row['sharpness']>matched['best']['sharpness']:matched['best']=row
 else:groups.append(dict(anchor=row,last=row['time'],times=[row['time']],best=row))
report={'video_sha256':video_hash,'model_sha256':model_hash,'source_sha256':source_hash,'fps':fps,'accepted':len(rows),'groups':[{'times':g['times'],'best':g['best']['path']} for g in groups],'cutout_seconds':sum(r['rembg_seconds'] for r in rows),'elapsed':time.monotonic()-start,'pairs':pairs,'api_calls':0}
(root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='pairs'}),flush=True)
