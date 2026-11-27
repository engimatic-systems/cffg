import {installHelp} from './help.js';
import {setupTours} from './tours.js';
import {mountGraph} from './graph.js';
import {renderTrace} from './trace.js';
import {setupCommands} from './commands.js';
import {openLibrary,installDockSizing} from './library-picker.js';

const $ = id => document.getElementById(id);
const state = {model:null, ref:'', graph:null, models:[], caps:{}, tab:'overview',
  undo:[], redo:[], job:null, events:null, jobEvents:null, openSequence:0};
const nonce = () => crypto.randomUUID().replaceAll('-','');
function el(tag,text,cls,help) {
  const node=document.createElement(tag);
  if(text!==undefined && text!==null) node.textContent=text;
  if(cls) node.className=cls;
  if(help) node.dataset.help=help;
  return node;
}
function button(text,fn,help) {
  const b=el('button',text,'neutral-button',help);
  b.type='button'; b.addEventListener('click',()=>Promise.resolve().then(fn).catch(fail));
  return b;
}
function report(text) { $('status').textContent=text; }
function fail(e) { report(e?.message || String(e)); }
async function api(path,options={}) {
  const opts={...options,headers:{...options.headers}};
  if(opts.body!==undefined && typeof opts.body!=='string') {
    opts.headers['Content-Type']='application/json'; opts.body=JSON.stringify(opts.body);
  }
  const res=await fetch(path,opts);
  const data=await res.json();
  if(!res.ok) throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail ?? data));
  return data;
}
const post=(path,body)=>api(path,{method:'POST',body});
const modelPath=()=>'/v1/models/'+state.model.id;
const short=id=>id.slice(0,8);
const bytes=n=>n>=1048576?(n/1048576).toFixed(2)+' MiB':(n/1024).toFixed(1)+' KiB';
const kindRef=()=>state.ref.slice(0,state.ref.indexOf(':'));
const nameRef=()=>state.ref.slice(state.ref.indexOf(':')+1);
function entries(container,items) {
  const dl=el('dl');
  for(const [name,value,help] of items) dl.append(el('dt',name,null,help),el('dd',value,null,help));
  container.append(dl);
}
const field=(help,key,label,type='text',value='',opts={})=>({help,key,label,type,value,...opts});
function form(title,specs,submit,accept,hint='',submitHelp,hintHelp='form.hint') {
  const dialog=$('form-dialog'), fields=$('fields'), frm=dialog.querySelector('form');
  $('form-title').textContent=title; fields.replaceChildren();
  if(hint) fields.append(el('p',hint,'empty',hintHelp));
  const inputs={};
  for(const f of specs) {
    const label=el('label',null,'field',f.help); label.append(el('span',f.label));
    const inp=el(f.type==='select'?'select':f.type==='area'?'textarea':'input');
    if(f.type==='select') {
      for(const option of f.choices) {
        const [value,text]=Array.isArray(option)?option:[option,option];
        const opt=el('option',text); opt.value=value; inp.append(opt);
      }
    } else if(f.type==='area') inp.rows=f.rows || 6;
    else inp.type=f.type;
    inp.dataset.help=f.help; inp.name=f.key; inp.value=f.value ?? '';
    if(f.min!==undefined) inp.min=f.min;
    if(f.max!==undefined) inp.max=f.max;
    if(f.type==='number') inp.step=f.step || 'any';
    inp.required=f.required ?? true;
    label.append(inp); fields.append(label); inputs[f.key]=inp;
  }
  const feedback=el('div',null,'form-feedback','form.feedback'); feedback.setAttribute('role','status'); fields.append(feedback);
  $('form-submit').dataset.help=submitHelp; $('form-submit').textContent=accept; $('form-submit').disabled=false;
  const close=()=>dialog.close();
  $('form-cancel').onclick=close;
  frm.onsubmit=async event=>{
    event.preventDefault();
    const values=Object.fromEntries(specs.map(f=>[f.key,f.type==='number'?Number(inputs[f.key].value):inputs[f.key].value]));
    try {
      $('form-submit').disabled=true;
      const result=await submit(values,feedback);
      if(result!==false && dialog.open) close();
    } catch(e) { feedback.textContent=e.message; fail(e); }
    finally { $('form-submit').disabled=false; }
  };
  dialog.showModal(); (Object.values(inputs)[0] || $('form-submit')).focus();
  return inputs;
}
async function refresh() {
  state.models=await api('/v1/models');
  const list=$('models'); list.replaceChildren();
  for(const m of state.models) {
    const b=button(m.name,()=>openModel(m.id),'library.model'); b.dataset.model=m.id;
    b.append(el('small',m.recipe,'meta','library.recipe')); b.setAttribute('aria-current',String(m.id===state.model?.id));
    list.append(b);
  }
  if(!state.models.length) list.append(el('p','No saved models. Create a graph to begin.','empty'));
}
async function openModel(id,preserve=false) {
  const seq=++state.openSequence;
  if(!preserve || state.model?.id!==id) state.plate='';
  const snap=await api('/v1/models/'+id+(state.plate?'?plate='+encodeURIComponent(state.plate):''));
  if(seq!==state.openSequence) return;
  if(!preserve || state.model?.id!==id) {
    state.undo=[]; state.redo=[]; state.ref=''; state.tab='overview'; state.job=null;
    state.jobEvents?.close();
  }
  const oldAnchor=state.model?.metadata.anchors?.[state.ref];
  if(preserve && oldAnchor) state.ref=Object.keys(snap.metadata.anchors).find(ref=>snap.metadata.anchors[ref]===oldAnchor) || state.ref;
  const newRevision=state.model?.revision!==snap.revision;
  state.model=snap; localStorage.setItem('studio.model',id);
  $('title').textContent=state.plate?state.plate+' · plate contents':snap.template.name;
  state.graph?.destroy();
  state.graph=mountGraph($('graph'),snap,ref=>select(ref,false));
  if(!Object.hasOwn(snap.metadata.anchors || {},state.ref)) state.ref='';
  if(state.ref) state.graph.select(state.ref);
  if(!snap.scene.items.length) {
    const empty=el('div',null,'studio-add-first');
    empty.append(el('h3','Start your graph'),button('Add variable and prior',addChannel));
    $('graph').append(empty);
  }
  outline(); inspector(); await refresh();
  if(state.tab!=='job') await activity();
  state.events?.close();
  const sub=new EventSource(modelPath()+'/events?after='+snap.revision);
  state.events=sub;
  sub.onmessage=event=>{
    const next=JSON.parse(event.data);
    if(state.model?.revision!==next.revision) openModel(id,true).catch(fail);
  };
  report(snap.template.name+' · compiled · '+short(snap.revision));
  if(newRevision) await tours.changed();
}
function outline() {
  const list=$('outline'),t=state.model.template;
  const filter=el('input',null,null,'outline.search'); filter.type='search'; filter.placeholder='Find an element'; filter.setAttribute('aria-label','Find graph element');
  filter.value=state.outlineQuery || '';
  const group=el('div',null,'outline-rows');
  function draw() {
    const q=filter.value.toLowerCase(); state.outlineQuery=q;
    group.replaceChildren();
    const sections=[['factor','Factors',t.factors],['var','Variables',t.variables],['port','Ports',t.ports],['blob','Parameters',t.blobs],['plate','Plates',t.plates || {}]];
    for(const [kind,title,collection] of sections) {
      const items=Object.keys(collection).filter(n=>!q || (n+' '+(collection[n].type || '')).toLowerCase().includes(q));
      if(!items.length) continue;
      group.append(el('h3',title+' · '+items.length,'section-label','outline.count'));
      let visible=items.slice(0,200);
      if(state.ref.startsWith(kind+':') && items.includes(nameRef()) && !visible.includes(nameRef())) visible.push(nameRef());
      for(const name of visible) {
        const ref=kind+':'+name;
        const b=button(name,()=>select(ref),'outline.'+kind); b.dataset.ref=ref;
        b.dataset.kind=kind==='var'?'variable':kind==='port'?'observed':kind;
        b.setAttribute('aria-selected',String(ref===state.ref)); group.append(b);
      }
      if(items.length>200) group.append(el('p','Filter to see more elements.','empty'));
    }
  }
  filter.addEventListener('input',draw); list.replaceChildren(filter,group); draw();
}
function select(ref,canvas=true) {
  state.ref=ref;
  if(canvas) state.graph?.select(ref);
  for(const b of $('outline').querySelectorAll('[data-ref]')) b.setAttribute('aria-selected',String(b.dataset.ref===ref));
  inspector();
}
function inspector() {
  const panel=$('inspector'); panel.replaceChildren();
  const {template:t,metadata,arrays,resources}=state.model;
  const kind=kindRef(), name=nameRef();
  const title=el('h3',state.ref?name:t.name,null,'inspector.title'); panel.append(title);
  const actions=el('div',null,'studio-actions');
  if(kind==='var' && t.variables[name]) {
    const v=t.variables[name];
    entries(panel,[['Domain',v.ef || (v.dim?'Gaussian':'Categorical'),'inspector.domain'],['Dimension',v.dim || v.cardinality || 1,'inspector.dimension']]);
    actions.append(button('Rename',rename,'command.rename'),button('Resize',resize,'command.resize'));
    for(const [factor,f] of Object.entries(t.factors)) if(Object.values({...f.edges,...f.param_edges}).includes(name))
      actions.append(button(factor,()=>select('factor:'+factor),'inspector.linkedFactor'));
  } else if(kind==='factor' && t.factors[name]) {
    const f=t.factors[name]; entries(panel,[['Family',f.type,'inspector.family']]);
    for(const [kind,bindings] of [['var',{...f.edges,...f.param_edges}],['blob',f.param_blobs]]) {
      for(const [role,value] of Object.entries(bindings)) {
        const ref=kind+':'+value;
        const link=button(role+' · '+value,()=>select(ref),kind==='var'?'inspector.boundVariable':'inspector.boundArray');
        link.dataset.ref=ref;
        actions.append(link);
      }
    }
    actions.append(button('Edit factor',()=>factorForm(false),'command.bindings'),button('Position',positionFactor,'command.position'));
    if(Object.keys(f.param_blobs).length) actions.append(button('Initialize parameters',initialize,'command.init'));
    actions.append(button('Add constraint',constraint,'command.constraint'));
  } else if(kind==='blob' && arrays[name]) {
    const a=arrays[name]; entries(panel,[['Shape',a.shape.join(' × ') || 'scalar','inspector.shape'],['Type',a.dtype,'inspector.dtype'],['Storage',bytes(a.bytes),'inspector.storage'],['Values',a.sample.map(x=>Number(x.toPrecision(4))).join(', '),'inspector.sample']]);
    actions.append(button('Set values',setArray,'inspector.setValues'));
  } else if(kind==='port' && t.ports[name]) {
    const p=t.ports[name]; entries(panel,[['Role',p.role,'inspector.portRole'],['Variable',p.variable,'inspector.portVariable']]);
    actions.append(button('Select variable',()=>select('var:'+p.variable),'inspector.selectVariable'),button('Extend boundary',addChannel,'inspector.extend'),button('Connect ports',connect,'command.connect'));
  } else if(kind==='plate' && t.plates?.[name]) {
    entries(panel,[['Dimensions',t.plates[name].dims.join(' × '),'inspector.plateDims']]);
    actions.append(button('Open plate contents',()=>enterPlate(name),'command.enter-plate'));
    actions.append(button('Resize plate',()=>editForm('Resize plate',[field('field.dims','dims','Dimensions (JSON)','text',JSON.stringify(t.plates[name].dims))],
      v=>[{action:'set_plate_dims',name,dims:JSON.parse(v.dims)}]),'inspector.resizePlate'));
  } else {
    entries(panel,[['Revision',short(state.model.revision),'inspector.revision'],['Factors',Object.keys(t.factors).length,'inspector.factors'],['Variables',Object.keys(t.variables).length,'inspector.variables'],
      ['State RAM',bytes(resources.state_bytes || 0),'profile.state'],['Parameters',bytes(resources.parameter_bytes || 0),'profile.parameters']]);
    actions.append(button('Add variable and prior',addChannel,'command.add'),button('Connect ports',connect,'command.connect'),button('Edit transaction',raw,'command.raw'));
  }
  if(state.ref) actions.append(button('Annotate',annotate,'command.annotate'),button('Critique',()=>comment('critique'),'command.critique'));
  if(state.plate) actions.append(button('Return to whole graph',()=>enterPlate(''),'command.exit-plate'));
  panel.append(actions);
  const anchor=metadata.anchors?.[state.ref];
  for(const note of metadata.notes || []) if(note.anchor===(anchor || null)) {
    panel.append(el('p',note.text,'studio-note','inspector.note'),button('Edit annotation',()=>editForm('Edit annotation',[field('field.note','text','Annotation','area',note.text)],v=>[{action:'edit_annotation',id:note.id,text:v.text}]),'inspector.editNote'),
       button('Remove annotation',()=>editForm('Remove annotation',[],()=>[{action:'remove_annotation',id:note.id}],note.text),'inspector.removeNote'));
  }
}
function changesText(changes) {
  return Object.entries(changes).map(([kind,names])=>kind+' · '+(Array.isArray(names)?names.map(n=>typeof n==='string'?n:JSON.stringify(n)).join(', '):JSON.stringify(names))).join('\n') || 'No structural changes';
}
function editForm(title,specs,build,hint='') {
  const path=modelPath(),base=state.model.revision,id=state.model.id;
  let reviewed='',key=nonce();
  return form(title,specs,async(v,feedback)=>{
    const local=build(v),commands=state.plate?[{action:'edit_plate',plate:state.plate,commands:local}]:local,signature=JSON.stringify(commands);
    const payload={base,summary:title,commands,request_id:key};
    if(reviewed!==signature) {
      key=nonce();payload.request_id=key; feedback.textContent='Checking model…';
      const preview=await post(path+'/transactions/preview',payload);
      feedback.replaceChildren(el('strong','Compiled preview'),el('pre',changesText(preview.changes)));
      $('form-submit').dataset.help='form.commit'; $('form-submit').textContent='Commit change'; reviewed=signature; return false;
    }
    const receipt=await post(path+'/transactions',payload);
    state.undo.push({base,head:receipt.revision}); state.redo=[];
    await openModel(id,true);
  },'Preview change',hint,'form.preview');
}
function enterPlate(name) {state.plate=name;state.ref='';return openModel(state.model.id,true);}
function create() {
  form('New model',[field('field.modelName','name','Model name','text','Untitled graph'),field('field.starter','kind','Starting point','select','blank',{choices:[['blank','Blank graph'],['kalman','Linear Gaussian state space'],['rslds','Recurrent switching linear dynamical system']]}),
    field('field.timeSteps','steps','Time steps','number',3,{min:1,max:256}),field('field.latent','latent','Latent dimensions','number',1,{min:1,max:64}),
    field('field.observed','observed','Observation dimensions','number',1,{min:1,max:64}),field('field.regimes','regimes','Regimes','number',2,{min:2,max:16}),field('field.modelSeed','seed','Seed','number',7)],
    async v=>{const doc=await post('/v1/models',v); await openModel(doc.id);},'Create model','','form.create');
}
function addChannel() {
  const ports=Object.keys(state.model.template.ports);
  editForm('Add variable and factor',[field('field.channelName','name','New variable'),field('field.incoming','source','Incoming boundary','select',kindRef()==='port'?nameRef():'',{required:false,choices:[['','Start with a prior'],...ports]}),
    field('field.priorDomain','family','Prior domain','select','gaussian',{choices:['gaussian','discrete']}),field('field.channelSize','size','Dimension or cardinality','number',2,{min:1,max:4096})],
    v=>[{action:'add_channel',...v}],'A prior creates an open boundary. Extend it with a transition to create a DATA port.');
}
function rename() {
  const name=nameRef();
  editForm('Rename variable',[field('field.rename','to','New name','text',name)],v=>[{action:'rename_variable',name,to:v.to}]);
}
function resize() {
  const name=nameRef(),v=state.model.template.variables[name];
  editForm('Resize variable',[field('field.resize','size','New dimension','number',v.dim || v.cardinality,{min:1,max:4096}),field('field.resizePolicy','policy','Dependent values','select','preserve',{choices:[['preserve','Preserve overlapping point values'],['reinitialize','Reset affected values and learning owners'],['map','Map existing coordinates']]}),
    field('field.coordinateMap','mapping','Coordinate map for Map mode (old index or null)','text','[]'),field('field.resetLearning','reset','Learning state for Map mode','select','keep',{choices:[['keep','Keep existing learning state'],['reset','Reset affected learning owners and recurrence sticks']]})],
    args=>[{action:'resize_variable',name,size:args.size,policy:args.policy,mapping:JSON.parse(args.mapping),reset_learning:args.reset==='reset'}],
    'The preview lists tied variables, arrays, consuming factors and learning owners.');
}
function initialize() {
  const name=nameRef(),f=state.model.template.factors[name],roles=Object.keys(f.param_blobs);
  const gaussian=f.type==='gaussian' || f.type==='gaussian_linear' || f.type==='gaussian_mixture_linear';
  function compatible(role) {
    const shape=state.model.arrays[f.param_blobs[role]].shape;
    const square=shape.length>=2 && shape.at(-1)===shape.at(-2);
    let values=['zeros','constant'];
    if(role==='cov' || role==='precision') values=['eye','random_spd'];
    if(['categorical','matrix_cat','transition_mixture'].includes(f.type)) values=['uniform','one_hot','random_stochastic',...(square?['sticky','banded','shift']:[])];
    if(role==='matrix' && gaussian && square) values.push('eye');
    return gaussian?['linear_gaussian','noisy_gaussian',...values]:values;
  }
  const presets=compatible(roles[0]);
  const inputs=editForm('Initialize factor',[field('field.paramRole','role','Parameter role','select',roles[0],{choices:roles}),field('field.preset','preset','Preset','select',presets[0],{choices:presets}),
    field('field.initSeed','seed','Random seed','number',7),field('field.noise','noise','Gaussian noise variance','number',.1,{min:0}),field('field.initOptions','params','Initializer options (JSON)','text','{}')],
    a=>[{action:'initialize_factor',factor:name,preset:a.preset,role:a.role,seed:a.seed,noise:a.noise,parameters:JSON.parse(a.params)}],
    'Gaussian recipes update the factor’s point parameters together. The preview includes shared parameters.');
  inputs.role.onchange=()=>inputs.preset.replaceChildren(...compatible(inputs.role.value).map(v=>new Option(v,v)));
}
function positionFactor(offset) {
  const name=nameRef(),current=state.model.metadata.layout?.[name] || [0,0];
  const xy=Array.isArray(offset)?offset:current;
  editForm('Position factor',[field('field.column','x','Column offset','number',xy[0]),field('field.row','y','Row offset','number',xy[1])],
    v=>[{action:'position_factor',name,offset:[v.x,v.y]}],'Offsets are relative to the automatic layout. Setting both to zero restores it.');
}
function factorForm(add=false) {
  const name=!add && kindRef()==='factor'?nameRef():'';
  const f=state.model.template.factors[name] || {type:'gaussian',edges:{out:''},param_blobs:{mean:'',cov:''},param_edges:{}};
  const inputs=editForm(name?'Edit factor bindings':'Add a factor',[
    field('field.factorName','name','Factor name','text',name),field('field.factorType','type','Factor type','select',f.type,{choices:Object.keys(state.caps.factors).sort()}),
    field('field.edges','edges','Edge roles → variable names (JSON)','area',JSON.stringify(f.edges,null,2),{rows:3}),
    field('field.params','params','Parameter roles → blob names (JSON)','area',JSON.stringify(f.param_blobs,null,2),{rows:3}),
    field('field.paramEdges','param_edges','Uncertain parameter edges (JSON)','text',JSON.stringify(f.param_edges)),
    field('field.before','before','Additional commands to apply first (JSON)','area','[]',{rows:2})],
    v=>{if(name && v.name!==name) throw Error('Use the existing factor name when editing bindings');
      return [...JSON.parse(v.before),{action:name?'replace_factor':'add_factor',spec:{...f,name:v.name,type:v.type,edges:JSON.parse(v.edges),param_blobs:JSON.parse(v.params),param_edges:JSON.parse(v.param_edges)}}];});
  inputs.type.onchange=()=>{
    const def=state.caps.factors[inputs.type.value];
    inputs.edges.value=JSON.stringify(Object.fromEntries(def.edge_roles.map(role=>[role,f.edges[role] || ''])),null,2);
    inputs.params.value=JSON.stringify(Object.fromEntries(def.param_roles.map(role=>[role,f.param_blobs[role] || ''])),null,2);
  };
}
function setArray() {
  const name=nameRef();
  editForm('Set parameter values',[field('field.values','values','Complete numeric array (JSON)','area','')],v=>[{action:'set_array',name,values:JSON.parse(v.values)}]);
}
function connect() {
  const ports=Object.keys(state.model.template.ports);
  if(ports.length<2) throw Error('Connect ports requires two open boundaries. Add another prior first.');
  const selected=kindRef()==='port'?nameRef():ports[0];
  const inputs=editForm('Connect two ports',[field('field.sourcePort','source','Keep variable at','select',selected,{choices:ports}),field('field.targetPort','target','Connect boundary','select',ports.find(p=>p!==selected),{choices:ports})],
    v=>[{action:'connect_ports',...v}],'The two boundary variables must have matching domains.');
  const t=state.model.template;
  inputs.source.onchange=()=>{
    const left=t.variables[t.ports[inputs.source.value].variable];
    const candidates=ports.filter(p=>p!==inputs.source.value && JSON.stringify(left)===JSON.stringify(t.variables[t.ports[p].variable]));
    inputs.target.replaceChildren(...candidates.map(v=>new Option(v,v)));
  };
  inputs.source.onchange();
}
function constraint() {
  const name=nameRef(),k=kindRef();
  const choices=k==='var'?['delta']:['mean_field','p_substitution','moment_matching'];
  editForm('Add constraint',[field('field.constraint','kind','Constraint','select',choices[0],{choices}),field('field.detail','detail','Details (JSON)','text','{}')],
    v=>[{action:'add_constraint',spec:{kind:v.kind,target:name,detail:JSON.parse(v.detail)}}]);
}
function raw() {
  editForm('Edit transaction',[field('field.commands','json','Ordered graph commands (JSON)','area','[\n  {"action":"add_channel","name":"signal","family":"gaussian","size":1}\n]',{rows:10})],
    v=>JSON.parse(v.json),'Group intermediate edits into one compile-checked revision. See API capabilities for the factor and action vocabulary.');
}
function annotate() {comment('annotate');}
function comment(action) {
  const ref=state.ref;
  const specs=[field(action==='annotate'?'field.note':'field.review','text',action==='annotate'?'Annotation':'Review comment','area','')];
  if(action==='propose') specs.push(field('field.proposal','commands','Proposed commands (JSON)','area','[]'));
  editForm(action==='annotate'?'Annotate graph':action==='propose'?'Propose an edit':'Critique graph',specs,
    v=>[{action,text:v.text,ref,...(v.commands?{commands:JSON.parse(v.commands)}:{})}],ref || 'Whole model');
}
function reviewAction(r,action) {
  const specs=action==='reply'?[field('field.reply','text','Reply','area','')]:[];
  editForm(action+' review',specs,v=>[{action,id:r.id,...v,reviewed_digest:state.model.metadata.model_digest}],r.text);
}
function setTab(name) {state.tab=name; return activity();}
async function activity() {
  if(!state.model) return;
  const p=$('activity'); p.replaceChildren();
  if(state.tab==='history') {
    const model=state.model.id,rows=await api(modelPath()+'/history');
    if(state.tab!=='history' || state.model.id!==model) return;
    for(const r of rows.slice(0,100)) {
      const b=button(short(r.revision)+'  '+r.summary,async()=>{
        const delta=await api(modelPath()+'/diff?base='+r.revision);
        form('Historical revision',[],async()=>{const base=state.model.revision;const rr=await post(modelPath()+'/transactions',{base,restore:r.revision,summary:'Restore '+short(r.revision)});
          state.undo.push({base,head:rr.revision}); await openModel(model,true);},'Restore this revision',changesText(delta),'form.restore','form.historyDiff');
      },'activity.history'); p.append(b);
    }
  } else if(state.tab==='critiques') {
    p.append(button('Write critique',()=>comment('critique'),'command.critique'),button('Propose edit',()=>comment('propose'),'command.propose'));
    for(const r of [...(state.model.metadata.reviews || [])].reverse()) {
      const item=el('section',null,'review-entry','activity.review');item.append(el('strong',r.author+' · '+r.type+' · '+r.status),el('p',r.text));
      for(const rep of r.replies) item.append(el('p',rep.author+' · '+rep.text));
      item.append(button('Reply',()=>reviewAction(r,'reply'),'activity.reply'));
      if(r.status==='open') {
        if(r.type==='propose') item.append(button('Preview and accept',()=>reviewAction(r,'accept'),'activity.accept'),button('Decline',()=>reviewAction(r,'decline'),'activity.decline'));
        else item.append(button('Resolve',()=>reviewAction(r,'resolve'),'activity.resolve'));
      } p.append(item);
    }
  } else if(state.tab==='runs') {
    const rows=await api('/v1/runs?model='+state.model.id);
    if(state.tab!=='runs') return;
    p.append(button('Run model',()=>runForm('run'),'command.run'),button('Step through inference',()=>runForm('trace'),'command.trace'),button('Benchmark',()=>runForm('profile'),'command.bench'));
    for(const row of rows) p.append(button(row.kind+' · '+row.status+' · '+short(row.revision)+' · '+new Date(row.created*1000).toLocaleTimeString(),()=>watchJob(row.id),'activity.job'));
  } else if(state.tab==='simulator') {
    p.append(button('Generate simulator',generateSimulator,'activity.generate'),button('Run simulator code',()=>simCode(),'activity.simulate'),button('Saved simulators and runs',()=>setTab('runs'),'command.runs'));
  } else if(state.tab==='profile') {
    const r=state.model.resources;
    const arenas=['messages','g_eta','g_lam','g_obs','g_mask','ef_eta','ef_obs','ef_mask'];
    entries(p,[['Persistent state',bytes(r.state_bytes),'profile.state'],['Parameter storage',bytes(r.parameter_bytes),'profile.parameters'],
      ...Object.entries(r.arenas).map(([k,v])=>[k,bytes(v),'profile.arena.'+(arenas.includes(k)?k:'other')])]);
    const pinned=state.model.revision;
    const rows=await api('/v1/runs?model='+state.model.id);
    const latest=rows.find(row=>row.kind==='profile' && row.revision===pinned && row.status==='finished');
    if(state.tab!=='profile' || state.model.revision!==pinned) return;
    if(latest) {
      const job=await api('/v1/runs/'+latest.id);
      if(state.tab!=='profile' || state.model.revision!==pinned) return;
      const r=job.result, samples=[...r.warm_ms].sort((a,b)=>a-b);
      const median=(samples[Math.floor((samples.length-1)/2)]+samples[Math.floor(samples.length/2)])/2;
      entries(p,[['Cold execution',r.cold_ms.toFixed(2)+' ms','profile.cold'],['Warm median',median.toFixed(2)+' ms · '+samples.length+' samples','profile.warm'],
        ['Peak process RAM',bytes(r.peak_rss_bytes),'profile.rss'],['Schedule',r.schedule,'profile.schedule'],['Device',r.device,'profile.device']]);
      p.append(button('Open measurement',()=>watchJob(job.id),'profile.open'));
    } else {
      p.append(el('p','Cold time, warm latency, and peak RAM are available after measuring this revision.','empty','profile.empty'));
    }
    p.append(button('Measure cold and warm execution',()=>runForm('profile'),'command.bench'),button('Previous runs',()=>setTab('runs'),'command.runs'));
  } else {
    p.append(el('strong','Current revision · '+short(state.model.revision),null,'activity.revision'),el('p','Compiled and saved in the model library.',null,'activity.revision'));
    p.append(button('Run history',()=>setTab('runs'),'command.runs'),button('Inspect resources',()=>setTab('profile'),'command.profile'),button('Add graph element',addChannel,'command.add'));
  }
}
async function watchJob(id) {
  report('Experiment · '+short(id)); state.tab='job';state.job=id;state.jobEvents?.close();
  async function draw() {
    const job=await api('/v1/runs/'+id);
    if(state.job!==id || state.tab!=='job') return;
    const p=$('activity');renderTrace(p,job,async action=>{await post('/v1/runs/'+id+'/control',{action});await draw();});
    if(job.result?.source) p.append(button('Inspect and run generated source',()=>simCode(job.result.source),'activity.inspectSource'));
    if(job.kind==='simulate' && job.result) p.append(button('Run this observation tape',()=>runForm('run',id),'activity.tape'));
    if(job.result?.beliefs) p.append(button('Inspect posterior beliefs',()=>form('Posterior beliefs',[],async()=>{},'Close',JSON.stringify(job.result.beliefs,null,2),'form.close','form.beliefs'),'activity.beliefs'));
    if(job.result) p.append(button('Measurement details',()=>form('Run details',[],async()=>{},'Close',JSON.stringify(job.result,null,2),'form.close','form.result'),'activity.measurement'));
    p.append(button('All runs',()=>setTab('runs'),'command.runs'));
    if(['finished','failed','cancelled','interrupted'].includes(job.status)) state.jobEvents?.close();
    return job;
  }
  const job=await draw();
  if(!job || !['queued','running','paused'].includes(job.status)) return;
  const sub=new EventSource('/v1/runs/'+id+'/events?after='+job.events.length);
  state.jobEvents=sub;
  let timer;
  sub.onmessage=()=>{clearTimeout(timer);timer=setTimeout(()=>draw().catch(fail),80);};
  sub.addEventListener('complete',()=>{sub.close();draw().catch(fail);});
}
function runForm(kind,tape='') {
  const m=state.model;
  form(kind==='profile'?'Benchmark this revision':kind==='trace'?'Trace inference':'Run inference',[
    field('field.runSeed','seed','Observation seed','number',7),field('field.iterations','iterations','Iteration budget','number',kind==='trace'?6:12,{min:1,max:kind==='trace'?128:1024}),
    field('field.tolerance','tolerance','Convergence tolerance','number',.00001),field('field.tape','tape','Prior observation tape (optional run ID)','text',tape,{required:false}),
    ...(kind==='trace'?[field('field.playback','paused','Playback','select','yes',{choices:[['yes','Start paused'],['no','Play continuously']]})]:[]),
    ...(kind==='profile'?[field('field.repetitions','repetitions','Warm samples','number',5,{min:1,max:50})]:[])],
    async v=>{const job=await post('/v1/runs',{...v,kind,tape:v.tape || null,paused:v.paused==='yes',model:m.id,revision:m.revision});await watchJob(job.id);},'Start','','form.start');
}
function generateSimulator() {
  const m=state.model;
  form('Generate observation simulator',[field('field.prompt','prompt','Stochastic dynamics','area','Simulate two switching regimes with noisy position observations.'),field('field.context','context','Share model context','select','ports',{choices:[['ports','Observation names and domains'],['structure','Include factor bindings and parameter shapes']]})],
  async v=>{const job=await post('/v1/runs',{...v,model:m.id,revision:m.revision,kind:'generate',include_structure:v.context==='structure'});await watchJob(job.id);},'Generate source',
  'Send the prompt and this revision’s DATA-port names and domains to the configured language model.','form.generate');
}
function simCode(source='') {
  const m=state.model;
  form('Run simulator code',[field('field.source','source','Python source','area',source,{rows:12}),field('field.simSeed','seed','Seed','number',7),
    field('field.simSteps','steps','Simulation steps','number',m.metadata.recipe?.steps || 3),field('field.simParams','parameters','Parameters (JSON)','text','{}')],
    async v=>{const job=await post('/v1/runs',{...v,parameters:JSON.parse(v.parameters),model:m.id,revision:m.revision,kind:'simulate'});await watchJob(job.id);},
    'Generate observations','Define simulate(seed, steps, parameters). Return observations keyed by DATA port, with optional ground_truth.','form.simulate');
}
async function undo(redo=false) {
  const stack=redo?state.redo:state.undo,item=stack.at(-1);
  if(!item || item.head!==state.model.revision) throw Error('No matching local edit. Inspect History to review other revisions.');
  const from=state.model.revision;
  const r=await post(modelPath()+'/transactions',{base:from,restore:item.base,summary:redo?'Redo edit':'Undo edit'});
  stack.pop(); if(stack.length) stack.at(-1).head=r.revision;
  (redo?state.undo:state.redo).push({base:from,head:r.revision});await openModel(state.model.id,true);
}
function fork() {
  form('Save a copy',[field('field.copyName','name','Name','text',state.model.template.name+' (copy)')],async v=>{
    const r=await post(modelPath()+'/fork',{...v,revision:state.model.revision});await openModel(r.id);},'Save as','','form.fork');
}
async function exportGraph(project=false) {
  const path=modelPath()+'/export?revision='+state.model.revision+'&project='+project;
  const response=await fetch(path);
  if(!response.ok) throw Error('Export failed ('+response.status+')');
  const url=URL.createObjectURL(await response.blob());
  const a=el('a');a.href=url;a.download=state.model.template.name+(project?'.cffgstudio':'.cffgz');
  document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
function walk(dir) {
  const t=state.model.template,related=[];
  if(kindRef()==='factor') related.push(...Object.values({...t.factors[nameRef()].edges,...t.factors[nameRef()].param_edges}).map(n=>'var:'+n));
  else if(kindRef()==='var') for(const [n,f] of Object.entries(t.factors)) if(Object.values({...f.edges,...f.param_edges}).includes(nameRef())) related.push('factor:'+n);
  if(related.length) {const ref=dir<0?related.at(-1):related[0];select(ref);}
}
function focusPanel() {
  const list=['models','outline','viewport','inspector','activity'];
  const i=list.findIndex(id=>$(id).contains(document.activeElement));
  $(list[(i+1)%list.length]).focus();
}
const tours=setupTours({
  report,
  currentModel:()=>state.model?.id,
  rootModel:()=>state.plate ? api(modelPath()) : state.model,
  async showTarget(owner,stop) {
    if (state.model?.id !== owner.id) return null;
    const plateRef=stop.plate_anchor && Object.keys(owner.metadata.anchors)
      .find(ref=>owner.metadata.anchors[ref]===stop.plate_anchor);
    if (stop.plate_anchor && !plateRef) return null;
    const scope=plateRef ? plateRef.slice(6) : '';
    if ((state.plate || '') !== scope) await enterPlate(scope);
    if (!stop.anchor) return $('graph');
    const ref=Object.keys(state.model.metadata.anchors).find(ref=>state.model.metadata.anchors[ref]===stop.anchor);
    if (!ref) return null;
    select(ref);
    return state.graph.reveal(ref) || $('inspector');
  },
});
const commands=[];
function cmd(id,label,run,keys='',enabled=()=>!!state.model) {commands.push({id,label,run,keys,enabled});}
cmd('palette','Command palette',()=>manager.showPalette(),'Mod+k',()=>true);
cmd('tours','Guided tours',()=>tours.open(),'',()=>true);
cmd('intro','Take the introductory tour',()=>tours.open('intro'),'',()=>true);
cmd('keys','Customize keyboard shortcuts',()=>manager.showKeys(),'',()=>true);
cmd('new','New model',create,'Mod+Alt+n',()=>true);
cmd('open','Open or import a model',()=>openLibrary({api,openModel,refresh,report}),'Mod+o',()=>true);
cmd('save','Save current revision',async()=>{await api(modelPath()+'/revisions/'+state.model.revision);report('Saved and compiled · '+short(state.model.revision));},'Mod+s');
cmd('fork','Save as a new model',fork,'Mod+Shift+s');
cmd('save-back','Write back to imported bundle',()=>form('Save back to source',[],async()=>{const result=await post(modelPath()+'/save-back',{base:state.model.revision,source_sha256:state.lastSourceHash?.[state.model.id]});(state.lastSourceHash ||= {})[state.model.id]=result.sha256;report('Saved '+result.path);},'Replace source bundle',state.model.metadata.source?.path || 'No source file','form.replace'),'',()=>!!state.model?.metadata.source);
cmd('export','Export portable bundle',()=>exportGraph());
cmd('export-project','Export graph with Studio annotations',()=>exportGraph(true));
cmd('undo','Undo my last edit',()=>undo(),'Mod+z');
cmd('redo','Redo my edit',()=>undo(true),'Mod+Shift+z');
cmd('add','Add variable and factor',addChannel,'a');
cmd('connect','Connect boundary ports',connect,'c');
cmd('rename','Rename selected variable',rename,'F2',()=>kindRef()==='var');
cmd('resize','Resize selected variable',resize,'Shift+r',()=>kindRef()==='var');
cmd('init','Initialize selected factor',initialize,'i',()=>kindRef()==='factor' && Object.keys(state.model.template.factors[nameRef()]?.param_blobs || {}).length>0);
cmd('constraint','Add a constraint',constraint,'',()=>['factor','var'].includes(kindRef()));
cmd('annotate','Annotate graph or selection',annotate,'n');
cmd('critique','Write a critique',()=>comment('critique'));
cmd('propose','Propose a graph change',()=>comment('propose'));
cmd('enter-plate','Open selected plate',()=>enterPlate(nameRef()),'',()=>kindRef()==='plate');
cmd('exit-plate','Return to whole graph',()=>enterPlate(''),'',()=>!!state.plate);
cmd('factor','Add a factor',()=>factorForm(true));
cmd('bindings','Edit selected factor bindings',()=>factorForm(false),'',()=>kindRef()==='factor');
cmd('position','Position selected factor',positionFactor,'',()=>kindRef()==='factor');
cmd('raw','Edit graph commands (JSON)',raw);
cmd('zoom-in','Zoom in',()=>state.graph.zoom(1.25),'=');
cmd('zoom-out','Zoom out',()=>state.graph.zoom(.8),'-');
cmd('fit','Fit graph',()=>state.graph.fit(),'0');
cmd('history','Revision history',()=>setTab('history'));
cmd('critiques','Critiques and proposals',()=>setTab('critiques'));
cmd('profile','Resource profile',()=>setTab('profile'));
cmd('bench','Benchmark cold and warm execution',()=>runForm('profile'));
cmd('run','Run model',()=>runForm('run'),'Mod+Enter');
cmd('trace','Step through inference',()=>runForm('trace'));
cmd('runs','Run history',()=>setTab('runs'));
cmd('simulator','Observation simulators',()=>setTab('simulator'));
cmd('share','Copy model or selection link',async()=>{const u=new URL(location.href);u.search='';u.searchParams.set('model',state.model.id);if(state.ref)u.searchParams.set('ref',state.ref);await navigator.clipboard.writeText(u.href);report('Copied link to '+state.model.template.name);});
cmd('focus','Focus next panel',focusPanel,'F6',()=>true);
cmd('forward','Follow wire forward',()=>walk(1),'ArrowRight');
cmd('back','Follow wire backward',()=>walk(-1),'ArrowLeft');
for (const control of document.querySelectorAll('[data-cmd]')) {
  if (!control.closest('#viewport')) control.dataset.help = 'command.' + control.dataset.cmd;
}
installHelp();
const manager=setupCommands(commands,report);
installDockSizing();
$('graph').addEventListener('graphmove',e=>{
  const {ref,dx,dy}=e.detail;select(ref);
  const pos=state.model.metadata.layout?.[nameRef()] || [0,0];
  positionFactor([pos[0]+dx,pos[1]+dy]);
});
$('graph').addEventListener('graphzoom',e=>{$('zoom').textContent=Math.round(e.detail.percent)+'%';});
$('outline').addEventListener('keydown',event=>{
  if(!['ArrowDown','ArrowUp'].includes(event.key)) return;
  event.preventDefault();const choices=[...$('outline').querySelectorAll('button')];
  if(!choices.length) return;
  const i=choices.indexOf(document.activeElement);choices[(i+(event.key==='ArrowDown'?1:-1)+choices.length)%choices.length].focus();
});
async function boot() {
  state.caps=await api('/v1/capabilities');await refresh();
  const url=new URL(location.href);
  const id=url.searchParams.get('model') || localStorage.getItem('studio.model');
  const initial=state.models.find(m=>m.id===id) || state.models.find(m=>m.recipe==='rslds') || state.models[0];
  if(initial) {await openModel(initial.id);const ref=url.searchParams.get('ref');if(ref)select(ref);const tour=url.searchParams.get('tour');if(tour)await tours.open(tour);}else report('Ready · create a blank model or open a template');
}
for(const panel of [$('outline'),$('inspector')]) {
  panel.addEventListener('pointerover',e=>state.graph?.hover(e.target.closest('[data-ref]')?.dataset.ref || null));
  panel.addEventListener('pointerleave',()=>state.graph?.hover(null));
}
$('graph').addEventListener('graphhover',e=>{
  const refs=new Set(e.detail?.refs || []);
  for(const panel of [$('outline'),$('inspector')]) {
    for(const row of panel.querySelectorAll('[data-ref]')) row.classList.toggle('graph-hover',refs.has(row.dataset.ref));
  }
});
boot().catch(fail);
