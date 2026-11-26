"""Compound document commands shared by the keyboard editor and agent clients."""
from __future__ import annotations

from dataclasses import replace
from time import time
from uuid import uuid4
import re

import numpy as np

from cffg import Engine, compile_template, inits
from cffg.core.edits import action_from_dict
from cffg.core.ir import BlobRef, EDGE_CONSTRAINTS, FactorSpec, PortRole, PortSpec, VariableSpec
from cffg.core.serialize import from_dict
from cffg.learn import Learnables
from cffg.models.rslds_fit import vague_rslds_fit_priors
from cffg.resolution.sources import _blob_shape

from .models import starter


def new_name(value, existing):
    if not isinstance(value,str) or not value.strip() or len(value)>160 or any(ord(c)<32 for c in value):
        raise ValueError("Choose a nonempty name of up to 160 characters")
    if value in existing:
        raise ValueError(f"{value!r} already exists")
    return value


def rename_refs(t, old, name):
    for f in t.factors.values():
        for roles in (f.edges,f.param_edges):
            for role,edge in roles.items():
                if edge == old:
                    roles[role] = name
    t.ports = {key:replace(p,variable=name) if p.variable==old else p for key,p in t.ports.items()}
    t.constraints = [replace(c,target=name) if c.kind in EDGE_CONSTRAINTS and c.target==old else c for c in t.constraints]
    if t.agent:
        bindings = tuple(replace(b,posterior=name) if b.posterior==old else b for b in t.agent.beliefs)
        t.agent = replace(t.agent,action=name if t.agent.action==old else t.agent.action,beliefs=bindings)


def add_channel(d, cmd):
    t = d.template
    name = new_name(cmd["name"],t.variables)
    port = t.ports.get(cmd.get("source",""))
    if cmd.get("source") and not port:
        raise ValueError("Select an existing boundary port")
    f_name = new_name(cmd.get("factor") or name+"_factor",t.factors)
    port_name = new_name(cmd.get("port") or name,t.ports)
    size = int(cmd.get("size",2))
    if not 1 <= size <= 4096:
        raise ValueError("Dimension must be between 1 and 4096")
    source = t.variables[port.variable] if port else None
    gaussian = (source.family if source else cmd.get("family","gaussian")) == "gaussian"
    if source and source.ef:
        raise ValueError("Use a typed factor command to extend an exponential-family boundary")
    t.variables[name] = VariableSpec(name,dim=size) if gaussian else VariableSpec(name,cardinality=size)
    edges = {"out":name}
    if source:
        edges["in"] = source.name
        del t.ports[port.name]
    if gaussian:
        kind = "gaussian_linear" if source else "gaussian"
        values = {"cov":np.eye(size), ("offset" if source else "mean"):np.zeros(size)}
        if source:
            values["matrix"] = np.eye(size,source.dim)
    else:
        kind = "matrix_cat" if source else "categorical"
        values = {"matrix":np.full((size,source.cardinality),1/size)} if source else {"statistic":np.full(size,1/size)}
    blobs = {}
    for role,array in values.items():
        blob = new_name(f"{f_name}.{role}",t.blobs)
        t.blobs[blob] = BlobRef(blob,f"blobs/{blob}.npy")
        d.arrays[blob] = array
        blobs[role] = blob
    t.factors[f_name] = FactorSpec(f_name,kind,edges=edges,param_blobs=blobs)
    t.ports[port_name] = PortSpec(port_name,name,PortRole.DATA)


def factor_init(d,c):
    factor = d.template.factors[c["factor"]]
    kind = c["preset"]
    seed = int(c.get("seed",0))
    noise = float(c.get("noise",.1))
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("Noise variance must be positive and finite")
    rng = np.random.default_rng(seed)
    if kind in ("linear_gaussian","noisy_gaussian"):
        if factor.type not in ("gaussian","gaussian_linear","gaussian_mixture_linear"):
            raise ValueError("Select a Gaussian prior or linear Gaussian factor")
        for role,blob in factor.param_blobs.items():
            prev = d.arrays[blob]
            if role == "matrix":
                value = np.broadcast_to(np.eye(*prev.shape[-2:]),prev.shape).copy()
                if kind == "noisy_gaussian":
                    value += rng.normal(0,np.sqrt(noise),prev.shape)
            elif role == "cov":
                value = np.broadcast_to(np.eye(prev.shape[-1])*noise,prev.shape)
            else:
                value = rng.normal(0,np.sqrt(noise),prev.shape) if kind=="noisy_gaussian" else np.zeros(prev.shape)
            d.arrays[blob] = value.astype(prev.dtype)
        return
    blob = factor.param_blobs[c["role"]]
    prev = d.arrays[blob]
    params = c.get("parameters",{})
    if kind in ("random_spd","random_stochastic"):
        params = {**params,"seed":seed}
    d.arrays[blob] = inits.make(kind,prev.shape,**params).astype(prev.dtype)
    d.template.blobs[blob] = replace(d.template.blobs[blob],init=inits.spec(kind,**params))


def resize(d,c):
    t = d.template
    v = t.variables[c["name"]]
    size = int(c["size"])
    if not 1 <= size <= 4096 or v.ef:
        raise ValueError("Resize requires a categorical or Gaussian variable of dimension 1–4096")
    if (v.dim or v.cardinality)==size:
        return
    recipe = d.meta.get("recipe",{})
    axes = d.meta.get("axes",{})
    axis = axes.get(v.name)
    names = {v.name} if not axis else {n for n,a in axes.items() if a==axis and n in t.variables}
    more = True
    while more:
        before = len(names)
        for f in t.factors.values():
            if f.type=="equality" and names.intersection(f.edges.values()):
                names.update(f.edges.values())
        more = len(names)!=before
    for n in names:
        t.variables[n] = replace(t.variables[n],**{"dim" if v.dim else "cardinality":size})
    if c.get("policy","preserve") not in ("preserve","reinitialize"):
        raise ValueError("Choose preserve or reinitialize")
    baseline = None
    if axis and recipe.get("kind") in ("rslds","kalman"):
        recipe[axis] = size
        baseline = starter(**recipe,name=t.name)
    changed = set()
    for blob,prev in list(d.arrays.items()):
        reference = baseline.arrays.get(blob) if baseline else None
        shape = reference.shape if reference is not None else _blob_shape(t,blob)
        if shape is None or tuple(shape)==prev.shape:
            continue
        changed.add(blob)
        roles = {role for f in t.factors.values() for role,b in f.param_blobs.items() if b==blob}
        if reference is not None:
            out = reference.astype(prev.dtype,copy=True)
        elif "cov" in roles:
            out = np.broadcast_to(np.eye(shape[-1]),shape).astype(prev.dtype,copy=True)
        elif any(f.type in ("categorical","matrix_cat") and blob in f.param_blobs.values() for f in t.factors.values()):
            out = np.full(shape,1/shape[0],dtype=prev.dtype)
        else:
            out = np.zeros(shape,dtype=prev.dtype)
        if c.get("policy","preserve")=="preserve":
            region = tuple(slice(0,min(a,b)) for a,b in zip(prev.shape,shape,strict=True))
            out[region] = prev[region]
            if any(f.type in ("categorical","matrix_cat") and blob in f.param_blobs.values() for f in t.factors.values()):
                totals = out.sum(axis=0,keepdims=True)
                if np.any(totals<=0):
                    raise ValueError("Resize removes all probability mass. Choose reinitialize.")
                out = out/totals
        d.arrays[blob] = out
    owners = {name for name,l in t.learnables.items() if changed.intersection(l.blobs)}
    if owners:
        if c.get("policy")!="reinitialize" or baseline is None or baseline.learning is None:
            raise ValueError("Resize affects learned priors and posteriors: "+", ".join(sorted(owners))+
                             ". Choose explicit reinitialization for a supported recipe or supply a complete replacement bundle.")
        priors = dict(d.learning.initial_params)
        posts = dict(d.learning.params)
        for name in owners:
            priors[name] = baseline.learning.initial_params[name]
            posts[name] = baseline.learning.params[name]
        d.learning = Learnables.from_persistent_state(Engine(compile_template(t,arrays=d.arrays)),priors,posts)


def apply_commands(d, commands, actor, base):
    if not 1 <= len(commands) <= 256:
        raise ValueError("A transaction needs 1–256 commands")
    t = d.template
    for index,c in enumerate(commands):
        try:
            op = c["action"]
            if op=="rename_variable":
                old = c["name"]
                name = new_name(c["to"],t.variables)
                t.variables[name] = replace(t.variables.pop(old),name=name)
                rename_refs(t,old,name)
                anchors = d.meta.get("anchors",{})
                if "var:"+old in anchors:
                    anchors["var:"+name] = anchors.pop("var:"+old)
                axes = d.meta.get("axes",{})
                if old in axes:
                    axes[name] = axes.pop(old)
            elif op=="resize_variable":
                resize(d,c)
            elif op=="initialize_factor":
                factor_init(d,c)
            elif op=="add_channel":
                add_channel(d,c)
            elif op=="connect_ports":
                left,right = (t.ports[n] for n in (c["source"],c["target"]))
                if left==right or t.variables[left.variable].domain_key!=t.variables[right.variable].domain_key:
                    raise ValueError("Choose two distinct boundary ports with matching domains")
                del t.ports[left.name],t.ports[right.name]
                rename_refs(t,right.variable,left.variable)
                del t.variables[right.variable]
            elif op=="set_array":
                blob = c["name"]
                if blob not in t.blobs:
                    raise ValueError("Declare the blob before supplying its values")
                value = np.asarray(c["values"],dtype=c.get("dtype","float64"))
                if value.dtype.kind not in "fibu" or not np.isfinite(value).all():
                    raise ValueError("Supply finite numeric parameter values")
                d.arrays[blob] = value
            elif op=="model_name":
                t.name = new_name(c["name"],{})
            elif op in ("annotate","critique","propose"):
                ref = c.get("ref","")
                if ref and ref not in d.meta.get("anchors",{}):
                    raise ValueError(f"Unknown anchor {ref!r}")
                body = c.get("text","").strip()
                if not body or len(body)>16000:
                    raise ValueError("Enter a note of 1–16000 characters")
                entry = {"id":uuid4().hex,"anchor":d.meta.get("anchors",{}).get(ref),"text":body,"author":actor,
                         "base":base,"created":time(),"status":"open","type":op,"replies":[]}
                if op=="propose":
                    entry["commands"]=c["commands"]
                d.meta.setdefault("notes" if op=="annotate" else "reviews",[]).append(entry)
            elif op in ("reply","resolve","decline","accept"):
                matches = [r for r in d.meta.get("reviews",[]) if r["id"]==c["id"]]
                if not matches:
                    raise ValueError("Unknown critique or proposal")
                r = matches[0]
                if op=="reply":
                    r["replies"].append({"text":c["text"],"author":actor,"created":time()})
                else:
                    if r["status"]!="open":
                        raise ValueError("Review is already closed")
                    if op=="accept":
                        if r["type"]!="propose":
                            raise ValueError("Select a proposed edit")
                        if any(a.get("action") in ("accept","propose") for a in r["commands"]):
                            raise ValueError("Nested proposals cannot be applied")
                        apply_commands(d,r["commands"],actor,base)
                    r["status"]={"accept":"accepted","resolve":"resolved","decline":"declined"}[op]
            else:
                action_from_dict(c).apply(t)
        except (TypeError,KeyError,ValueError,IndexError,OverflowError) as e:
            raise ValueError(f"Command {index+1} ({c.get('action','?')}): {e}") from e
    return d
