"""prep._app_client — the zero-build masking-app client (inline style/body/script constants).

The full four-zone UI-SPEC contract (09.3-UI-SPEC.md, approved 6/6): a top bar, a left tool rail, a
fluid center canvas, and a bottom frame strip, plus the propagation-review / QA-overlay mode. Kept in
its own module so :mod:`prep.app` (the server + export contract) stays small and its export helpers
unit-test without dragging the whole client string.

``render_page`` (in ``prep.app``) injects the spec brush hex into every ``__BRUSH_HEX__`` placeholder
so the locked teal is NEVER a client literal (D-01/D-03 — the client also fetches ``/api/spec`` at
runtime; the injected value is the startup default). The design system is the house dark theme
(``_serve_clip_picker.py`` shell + ``mask_qa.html`` review grid).

DELIBERATE gridwatch departure (WARNING 3): the in-app review/QA grid is HAND-ROLLED (mirrors
``_inpaint_prep/mask_qa.html``), NOT gridwatch. Approved in 09.3-UI-SPEC.md — this is a mask-QA review
surface (per-clip green-overlay coverage judgement inside the drawing tool), not a training-sample
comparison grid (the house gridwatch rule governs training-sample grids, which Plan 09 uses). Do NOT
"correct" this to gridwatch.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------------------
# Style — house dark theme. Spacing on the {4,8,16,24,32} scale; 44px touch targets; two weights
# (400/600); accent #458070 reserved for brush paint / active tool / swatch / primary CTA only.
# --------------------------------------------------------------------------------------------------
CLIENT_STYLE = """<style>
 :root{
  --bg:#121212;--panel:#1d1d1d;--well:#0d0d0d;--accent:__BRUSH_HEX__;--destructive:#c0453a;
  --text:#eee;--muted:#9ab;--in-band:#9fd49f;--below:#d0a24a;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
 }
 *{box-sizing:border-box}
 html,body{height:100%;margin:0}
 body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;
      display:flex;flex-direction:column;overflow:hidden}
 .topbar{display:flex;align-items:center;gap:16px;background:var(--panel);padding:8px 16px;height:52px;flex:0 0 auto}
 .topbar h1{font-size:20px;font-weight:600;margin:0;color:var(--text)}
 .clipname{font-family:var(--mono);font-size:12px;color:var(--muted)}
 .coverage{font-family:var(--mono);font-size:12px;margin-left:auto;color:var(--in-band)}
 .coverage.below{color:var(--below)}
 button{font-family:inherit;font-size:14px;font-weight:400;color:var(--text);background:#2a2a2a;
        border:1px solid #333;border-radius:6px;padding:0 12px;min-height:44px;cursor:pointer}
 button:hover{background:#333}
 button.cta{background:var(--accent);color:#06110d;border-color:var(--accent);font-weight:600}
 button.destructive{background:var(--destructive);border-color:var(--destructive);color:#fff}
 button.active{outline:2px solid var(--accent);background:#20302b}
 .main{display:flex;flex:1 1 auto;min-height:0}
 .rail{display:flex;flex-direction:column;gap:8px;background:var(--panel);padding:8px;width:120px;flex:0 0 auto;overflow:auto}
 .rail button{width:100%}
 .rail label{font-size:12px;color:var(--muted);margin-top:8px}
 .rail input[type=range]{width:100%;min-height:44px}
 .swatch{height:44px;border-radius:6px;border:1px solid #333;background:var(--accent)}
 .swatch-cap{font-family:var(--mono);font-size:12px;color:var(--muted);text-align:center}
 .canvaswrap{flex:1 1 auto;position:relative;background:var(--well);overflow:hidden;min-width:0}
 .stage{position:absolute;transform-origin:0 0}
 .stage img,.stage canvas{position:absolute;top:0;left:0;image-rendering:pixelated;display:block}
 .strip{display:flex;gap:8px;background:var(--panel);padding:8px 16px;height:96px;flex:0 0 auto;
        align-items:center;overflow-x:auto}
 .strip .thumbs{display:flex;gap:8px}
 .strip img{height:64px;border-radius:4px;border:2px solid transparent;cursor:pointer;background:#000}
 .strip img.current{border-color:var(--accent)}
 .strip .actions{display:flex;gap:8px;margin-left:16px;flex:0 0 auto}
 .empty{margin:auto;text-align:center;color:var(--muted);max-width:520px;padding:24px}
 .empty h2{font-size:16px;font-weight:600;color:var(--text)}
 .empty code{font-family:var(--mono);font-size:12px;background:var(--well);padding:2px 6px;border-radius:4px}
 .hidden{display:none!important}
 /* Review / QA overlay mode — hand-rolled mirror of mask_qa.html (approved departure, WARNING 3) */
 .review{position:absolute;inset:0;background:var(--bg);overflow:auto;padding:24px}
 .review h1{font-size:20px;font-weight:600} .review h2{font-size:16px;font-weight:600;color:var(--muted);margin-top:24px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:8px}
 .card{background:var(--panel);border-radius:8px;padding:8px}
 .card video{width:100%;border-radius:6px;background:#000}
 .card .meta{font-size:12px;margin-top:8px;line-height:1.5;font-family:var(--mono)}
 .tag{padding:1px 7px;border-radius:9px;font-size:12px}
 .tag.face{background:#2a4d69}.tag.face_hair{background:#4b3869}.tag.full_body{background:#69422a}
 .confirm{margin-top:8px;background:var(--well);border:1px solid var(--destructive);border-radius:6px;padding:8px;font-size:14px}
 .confirm .row{display:flex;gap:8px;margin-top:8px}
</style>"""

# --------------------------------------------------------------------------------------------------
# Body — the four zones. "frames" abstraction everywhere (never "video"): a 1-frame clip degrades the
# strip to a single cell. All copywriting is the exact 09.3-UI-SPEC Copywriting Contract.
# --------------------------------------------------------------------------------------------------
CLIENT_BODY = """
<div class="topbar">
  <h1>signet mask app</h1>
  <span class="clipname" id="clipname">no clip loaded</span>
  <span class="coverage" id="coverage">0.0%</span>
  <button id="reviewToggle">Review masks</button>
  <button class="cta" id="exportBtn">Export masks</button>
</div>

<div class="main">
  <div class="rail" id="rail">
    <button class="active" data-tool="brush" id="toolBrush">Brush</button>
    <button data-tool="eraser" id="toolEraser">Eraser</button>
    <label for="brushSize">Brush size</label>
    <input type="range" id="brushSize" min="4" max="64" value="24">
    <button data-tool="zoom" id="toolZoom">Zoom</button>
    <button id="undoBtn">Undo</button>
    <button id="redoBtn">Redo</button>
    <button id="clearFrameBtn">Clear frame</button>
    <button class="destructive" id="clearAllBtn">Clear all</button>
    <label>Load paintover</label>
    <button id="paintoverBtn">Load paintover</button>
    <input type="file" id="paintoverFile" accept="image/png" class="hidden">
    <div class="swatch" id="swatch" title="locked mask color"></div>
    <div class="swatch-cap" id="swatchHex">__BRUSH_HEX__</div>
  </div>

  <div class="canvaswrap" id="canvaswrap">
    <div class="empty" id="emptyNoClips">
      <h2>No clips loaded</h2>
      <p>Point the masking app at a clips directory to start &mdash; <code>python scripts/serve_mask_app.py &lt;clips_dir&gt;</code>.</p>
    </div>
    <div class="empty hidden" id="emptyNoPaint">
      <h2>Paint the region to regenerate</h2>
      <p>The teal brush is the inpaint mask. Everything you paint is what the model will regenerate; unpainted pixels are kept as context.</p>
    </div>
    <div class="stage hidden" id="stage">
      <img id="frameImg" alt="frame">
      <canvas id="overlay"></canvas>
    </div>
    <div class="confirm hidden" id="clearAllConfirm">
      Clear all masks for this clip? This erases every painted and propagated frame.
      <div class="row">
        <button class="destructive" id="clearAllYes">Clear all</button>
        <button id="clearAllNo">Cancel</button>
      </div>
    </div>
  </div>
</div>

<div class="strip" id="strip">
  <div class="thumbs" id="thumbs"></div>
  <div class="actions">
    <button id="propagateBtn">Propagate from frame 0</button>
    <button id="reseedBtn">Re-seed clip</button>
  </div>
</div>

<div class="review hidden" id="review">
  <h1>SAM mask propagation &mdash; QA (green = region to REGENERATE)</h1>
  <p>Check each clip for drift: the green region should track the painted target (face / face+hair / torso) through the whole clip. Approve to stage.</p>
  <div style="margin:16px 0"><button id="reviewBack">Back to painting</button> <button class="cta" id="approveBtn">Approve &amp; stage</button></div>
  <div class="grid" id="reviewGrid"></div>
</div>

<!-- SAM-unavailable fix-it-first message (D-14), surfaced when propagate reports SAM missing -->
<template id="samUnavailableTpl">
  <div class="empty">
    <h2>SAM3 not available</h2>
    <p>Work through setup before falling back: (1) grant HF access to <code>facebook/sam3</code>, (2) install SAM3 into <code>.venv-sam</code>, (3) pull weights. SAM 2.1 is a last-ditch fallback only &mdash; it may not track this well.</p>
  </div>
</template>

<!-- Dims-violation error copy (÷64 / 8n+1), surfaced when a clip fails the inpaint dims gate -->
<template id="dimsErrorTpl">
  <div class="empty">
    <p>Clip <code>{stem}</code> is <code>{W}×{H}</code> &mdash; inpaint requires ÷64 spatial and (frames−1) divisible by 8. Crop/trim before encode.</p>
  </div>
</template>
"""

# --------------------------------------------------------------------------------------------------
# Script — Pointer-Events canvas painter with per-frame overlays, stroke-level undo/redo (depth 20),
# zoom/pan, live coverage banding, server-side export, and the review grid. Vanilla JS, zero-build.
# --------------------------------------------------------------------------------------------------
CLIENT_SCRIPT = """<script>
const BRUSH_DEFAULT='__BRUSH_HEX__';
const UNDO_DEPTH=20;
const S={brush:BRUSH_DEFAULT,rgb:[69,128,112],bands:{},tool:'brush',size:24,zoom:1,panX:0,panY:0,
  clips:[],clip:null,frames:0,frame:0,overlays:{},undo:{},redo:{},dirtyFrames:new Set()};

const $=id=>document.getElementById(id);
const overlay=$('overlay'), octx=overlay.getContext('2d');
const frameImg=$('frameImg'), stage=$('stage');

// Fetch the spec so the teal brush + coverage bands are read from the server (never a client literal).
fetch('/api/spec').then(r=>r.json()).then(s=>{
  S.brush=s.hex||BRUSH_DEFAULT; S.rgb=s.rgb||S.rgb; S.bands=s.coverage_bands||{};
  document.documentElement.style.setProperty('--accent',S.brush);
  $('swatchHex').textContent=S.brush;
}).catch(()=>{});

// ---- clip index + empty states ----
fetch('/api/clips').then(r=>r.json()).then(d=>{
  S.clips=d.clips||[];
  if(!S.clips.length){ $('emptyNoClips').classList.remove('hidden'); return; }
  loadClip(S.clips[0]);
});

function loadClip(c){
  S.clip=c.name; S.frames=Math.max(1,c.frames||1); S.frame=0;
  S.overlays={}; S.undo={}; S.redo={}; S.dirtyFrames=new Set();
  $('clipname').textContent=c.name+'  ('+S.frames+'f)';
  $('emptyNoClips').classList.add('hidden');
  buildStrip();
  showFrame(0);
}

function buildStrip(){
  const t=$('thumbs'); t.innerHTML='';
  // "frames" abstraction: a 1-frame clip degrades the strip to a single cell.
  for(let i=0;i<S.frames;i++){
    const im=new Image(); im.src='/frame/'+encodeURIComponent(S.clip)+'/'+i;
    im.dataset.i=i; im.title='frame '+i;
    if(i===0) im.classList.add('current');
    im.onclick=()=>gotoFrame(i);
    t.appendChild(im);
  }
}

function overlayHasPaint(){
  const d=octx.getImageData(0,0,overlay.width,overlay.height).data;
  for(let i=3;i<d.length;i+=4){ if(d[i]>127) return true; }
  return false;
}
function saveOverlay(){
  // Snapshot ONLY frames the operator actually painted on — a merely-visited frame must never
  // acquire an overlay entry, or Export would blank its propagated mask on disk.
  if(!S.clip || !S.dirtyFrames.has(S.frame)) return;
  if(!overlayHasPaint()){ delete S.overlays[S.frame]; S.dirtyFrames.delete(S.frame); return; }
  S.overlays[S.frame]=overlay.toDataURL('image/png');
}

function showFrame(i){
  S.frame=i;
  frameImg.onload=()=>{
    overlay.width=frameImg.naturalWidth; overlay.height=frameImg.naturalHeight;
    stage.style.width=frameImg.naturalWidth+'px'; stage.style.height=frameImg.naturalHeight+'px';
    octx.clearRect(0,0,overlay.width,overlay.height);
    const saved=S.overlays[S.frame];
    if(saved){ const im=new Image(); im.onload=()=>{octx.drawImage(im,0,0); updateCoverage();}; im.src=saved; }
    else { updateCoverage(); }
    applyZoom(); refreshEmptyPaint();
  };
  frameImg.src='/frame/'+encodeURIComponent(S.clip)+'/'+i;
  stage.classList.remove('hidden');
  document.querySelectorAll('#thumbs img').forEach(x=>x.classList.toggle('current',+x.dataset.i===i));
}

function gotoFrame(i){ if(i<0||i>=S.frames||i===S.frame) return; saveOverlay(); showFrame(i); }

function refreshEmptyPaint(){
  const painted=S.dirtyFrames.size>0 || !!S.overlays[S.frame];
  $('emptyNoPaint').classList.toggle('hidden', painted);
}

// ---- painting (Pointer Events; touch-action:none set on the overlay for palm rejection, D-07) ----
overlay.style.touchAction='none';
let drawing=false,last=null,panning=false,panStart=null,spaceDown=false;

function toCanvas(ev){
  const r=overlay.getBoundingClientRect();
  return {x:(ev.clientX-r.left)/S.zoom, y:(ev.clientY-r.top)/S.zoom};
}
function pushUndo(){
  const k=S.frame; (S.undo[k]=S.undo[k]||[]);
  S.undo[k].push(octx.getImageData(0,0,overlay.width,overlay.height));
  if(S.undo[k].length>UNDO_DEPTH) S.undo[k].shift();
  S.redo[k]=[];
}
function strokeTo(p){
  octx.globalCompositeOperation = (S.tool==='eraser') ? 'destination-out' : 'source-over';
  octx.strokeStyle=S.brush; octx.fillStyle=S.brush; octx.globalAlpha=1.0;
  octx.imageSmoothingEnabled=false; octx.lineCap='round'; octx.lineJoin='round'; octx.lineWidth=S.size;
  if(last){ octx.beginPath(); octx.moveTo(last.x,last.y); octx.lineTo(p.x,p.y); octx.stroke(); }
  octx.beginPath(); octx.arc(p.x,p.y,S.size/2,0,Math.PI*2); octx.fill();
  last=p;
}
overlay.addEventListener('pointerdown',ev=>{
  if(S.tool==='zoom'||spaceDown){ panning=true; panStart={x:ev.clientX-S.panX,y:ev.clientY-S.panY}; return; }
  drawing=true; pushUndo(); last=null; strokeTo(toCanvas(ev)); overlay.setPointerCapture(ev.pointerId);
});
overlay.addEventListener('pointermove',ev=>{
  if(panning){ S.panX=ev.clientX-panStart.x; S.panY=ev.clientY-panStart.y; applyZoom(); return; }
  if(drawing) strokeTo(toCanvas(ev));
});
function endStroke(){ if(drawing){ drawing=false; last=null; S.dirtyFrames.add(S.frame); saveOverlay(); updateCoverage(); refreshEmptyPaint(); } panning=false; }
overlay.addEventListener('pointerup',endStroke);
overlay.addEventListener('pointerleave',endStroke);

// The overlay is rendered live at 50% opacity so content shows through (server binarizes on export).
overlay.style.opacity='0.5';

function updateCoverage(){
  const d=octx.getImageData(0,0,overlay.width,overlay.height).data;
  let painted=0; const total=overlay.width*overlay.height;
  for(let i=3;i<d.length;i+=4){ if(d[i]>127) painted++; }
  const pct=total? (painted/total*100):0;
  const type=clipType(S.clip); const band=S.bands[type];
  const el=$('coverage'); el.textContent=pct.toFixed(1)+'%';
  if(band){ const belowTgt=(pct/100)<(band.target||0); el.classList.toggle('below',belowTgt); }
}
function clipType(name){
  if(!name) return null;
  const m=name.replace(/\\.mp4$/,'').match(/__(face_hair|full_body|face)$/);
  return m?m[1]:null;
}

// ---- zoom / pan (1x-8x; image-rendering:pixelated keeps mask edges crisp) ----
function applyZoom(){ stage.style.transform='translate('+S.panX+'px,'+S.panY+'px) scale('+S.zoom+')'; }
$('toolZoom').onclick=()=>setTool('zoom');
$('canvaswrap').addEventListener('wheel',ev=>{
  if(S.tool!=='zoom' && !ev.ctrlKey) return; ev.preventDefault();
  S.zoom=Math.min(8,Math.max(1, S.zoom*(ev.deltaY<0?1.1:0.9))); applyZoom();
},{passive:false});

// ---- tools ----
function setTool(t){ S.tool=t; document.querySelectorAll('.rail button[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===t)); }
$('toolBrush').onclick=()=>setTool('brush');
$('toolEraser').onclick=()=>setTool('eraser');
$('brushSize').oninput=e=>{ S.size=+e.target.value; };

// ---- undo / redo (stroke-level, depth 20; Ctrl/Cmd+Z / Ctrl/Cmd+Shift+Z) ----
function undo(){ const k=S.frame,st=S.undo[k]; if(st&&st.length){ (S.redo[k]=S.redo[k]||[]).push(octx.getImageData(0,0,overlay.width,overlay.height)); octx.putImageData(st.pop(),0,0); S.dirtyFrames.add(k); saveOverlay(); updateCoverage(); } }
function redo(){ const k=S.frame,st=S.redo[k]; if(st&&st.length){ (S.undo[k]=S.undo[k]||[]).push(octx.getImageData(0,0,overlay.width,overlay.height)); octx.putImageData(st.pop(),0,0); S.dirtyFrames.add(k); saveOverlay(); updateCoverage(); } }
$('undoBtn').onclick=undo; $('redoBtn').onclick=redo;

// ---- clear frame (undoable, no confirm) / clear all (confirm inline; not undoable) ----
$('clearFrameBtn').onclick=()=>{ pushUndo(); octx.clearRect(0,0,overlay.width,overlay.height); S.dirtyFrames.delete(S.frame); delete S.overlays[S.frame]; updateCoverage(); refreshEmptyPaint(); };
$('clearAllBtn').onclick=()=>$('clearAllConfirm').classList.remove('hidden');
$('clearAllNo').onclick=()=>$('clearAllConfirm').classList.add('hidden');
$('clearAllYes').onclick=()=>{ S.overlays={}; S.undo={}; S.redo={}; S.dirtyFrames=new Set(); octx.clearRect(0,0,overlay.width,overlay.height); $('clearAllConfirm').classList.add('hidden'); updateCoverage(); refreshEmptyPaint(); };

// ---- keyboard: arrows scrub frames, Ctrl/Cmd+Z undo, +Shift redo, space = pan ----
window.addEventListener('keydown',ev=>{
  if(ev.key==='ArrowLeft'){ gotoFrame(S.frame-1); }
  else if(ev.key==='ArrowRight'){ gotoFrame(S.frame+1); }
  else if((ev.ctrlKey||ev.metaKey)&&ev.key.toLowerCase()==='z'){ ev.preventDefault(); ev.shiftKey?redo():undo(); }
  else if(ev.code==='Space'){ spaceDown=true; }
});
window.addEventListener('keyup',ev=>{ if(ev.code==='Space') spaceDown=false; });

// ---- export (server thresholds every pixel to binary {0,255}, region=WHITE) ----
function postExport(stem,kind,frame,dataURL){
  return fetch('/export',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stem:stem,kind:kind,frame:frame,png:dataURL})}).then(r=>{
      if(!r.ok) return r.text().then(t=>{throw new Error(t);}); return r.json(); });
}
$('exportBtn').onclick=async()=>{
  if(!S.clip){ return; }
  saveOverlay();
  const stem=S.clip.replace(/\\.mp4$/,'');
  try{
    // frame-0 seed + every PAINTED frame (S.dirtyFrames) in the per-frame layout — never a frame
    // the operator merely scrolled past, so existing propagated masks on disk stay untouched
    if(S.dirtyFrames.has(0) && S.overlays[0]) await postExport(stem,'frame0',0,S.overlays[0]);
    for(const f of [...S.dirtyFrames].sort((a,b)=>a-b)){ if(S.overlays[f]) await postExport(stem,'video',+f,S.overlays[f]); }
    $('exportBtn').textContent='Exported ✓'; setTimeout(()=>$('exportBtn').textContent='Export masks',1200);
  }catch(e){
    alert('Could not write masks to `'+stem+'`. Check the path exists and is writable, then retry Export.');
  }
};

// ---- load paintover (external iPad/Procreate teal PNG -> server chroma extraction, D-10) ----
$('paintoverBtn').onclick=()=>$('paintoverFile').click();
$('paintoverFile').onchange=e=>{
  const file=e.target.files[0]; if(!file||!S.clip) return;
  const rd=new FileReader();
  rd.onload=()=>{
    const stem=S.clip.replace(/\\.mp4$/,'');
    fetch('/paintover',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({stem:stem,png:rd.result})}).then(r=>r.json())
      .then(()=>{ alert('Paintover imported as the frame-0 seed for '+stem); showFrame(S.frame); })
      .catch(()=>alert('Paintover import failed — is the PNG a teal paintover?'));
  };
  rd.readAsDataURL(file);
};

// ---- seed -> propagate -> review -> re-seed loop (SAM is Plan 05/08; surface fix-it-first if absent) ----
$('propagateBtn').onclick=()=>{
  $('propagateBtn').textContent='Propagating…';
  fetch('/api/propagate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stem:S.clip? S.clip.replace(/\\.mp4$/,''):null})})
    .then(r=>{ if(r.status===501||r.status===404){ throw {sam:true}; } return r.json(); })
    .then(()=>{ $('propagateBtn').textContent='Propagate from frame 0'; openReview(); })
    .catch(err=>{
      $('propagateBtn').textContent='Propagate from frame 0';
      const tpl=$('samUnavailableTpl').content.cloneNode(true);
      const wrap=$('canvaswrap'); stage.classList.add('hidden'); wrap.appendChild(tpl);
    });
};
$('reseedBtn').onclick=()=>{ // offer the --rev backward seed for clips that drifted
  if(confirm('Re-seed clip from the current frame? (Cancel = seed backward with --rev)')){ /* forward */ }
  gotoFrame(0);
};

// ---- review / QA overlay mode (hand-rolled mirror of mask_qa.html — approved departure) ----
function openReview(){
  $('review').classList.remove('hidden');
  fetch('/api/review').then(r=>r.json()).then(d=>{
    const g=$('reviewGrid'); g.innerHTML='';
    (d.clips||[]).forEach(c=>{
      const type=clipType(c.stem)||'face';
      const card=document.createElement('div'); card.className='card';
      card.innerHTML='<div class="meta"><b>'+c.stem+'</b><br><span class="tag '+type+'">'+type+'</span> &middot; '+
        (c.coverage?c.coverage.toFixed(1):'--')+'% avg coverage &middot; '+c.frames+'f</div>';
      g.appendChild(card);
    });
  });
}
$('reviewToggle').onclick=openReview;
$('reviewBack').onclick=()=>{ $('review').classList.add('hidden'); stage.classList.remove('hidden'); };
$('approveBtn').onclick=()=>{ alert('Approve & stage — staging/encode proceeds only after this QA approval.'); };
</script>"""
