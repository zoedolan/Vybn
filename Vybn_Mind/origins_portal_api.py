#!/usr/bin/env python3
"""origins_portal_api.py - HTTP API for Origins Portal frontend."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.expanduser('~/vybn-phase'))

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title='Origins Portal API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

_dm = None
def get_dm():
    global _dm
    if _dm is None:
        import deep_memory as dm
        dm._load()  # Initialize the cache
        _dm = dm
    return _dm

@app.get('/encounter')
async def encounter(query: str = Query(...), k: int = Query(8)):
    dm = get_dm()
    results = dm.search(query, k=k)
    total_phase = sum(abs(r.get('phase', 0)) for r in results)
    total_geom = sum(abs(r.get('geometry', 0)) for r in results)
    avg_dist = float(np.mean([r.get('distinctiveness', 0) for r in results])) if results else 0
    return {
        'query': query, 'phase': round(total_phase, 4), 'geometry': round(total_geom, 4),
        'distinctiveness': round(avg_dist, 4),
        'curvature': 'high' if total_geom > 1.0 else 'medium' if total_geom > 0.3 else 'low',
        'results': [{'source': r.get('source',''), 'text': r.get('text','')[:500],
            'fidelity': round(float(r.get('fidelity',0)), 4),
            'telling': round(float(r.get('telling',0)), 4),
            'distinctiveness': round(float(r.get('distinctiveness',0)), 4),
            'phase': round(float(r.get('phase',0)), 4),
            'regime': r.get('regime','unknown')} for r in results]
    }

PERSPECTIVES = {
    'legal': 'law litigation court rights justice brief appellate constitution governance',
    'mathematical': 'theorem proof geometry curvature topology algebra phase invariant conjecture',
    'autobiographical': 'Zoe memoir skydiving mirror Queen Boat Cairo drawing life love',
    'mythological': "Ma'at labyrinth Khunanup myth archetype Copernican Cambrian underworld",
    'political': 'Fukuyama abundance scarcity property commons state sovereignty governance order',
    'epistemological': 'a priori a posteriori a synthesi a symbiosi knowledge seeing truth',
}

@app.get('/inhabit')
async def inhabit(perspective: str = Query(...), k: int = Query(12)):
    dm = get_dm()
    q = PERSPECTIVES.get(perspective, perspective)
    results = dm.search(q, k=k)
    return {'perspective': perspective, 'results': [{'source': r.get('source',''),
        'text': r.get('text','')[:800],
        'fidelity': round(float(r.get('fidelity',0)), 4),
        'telling': round(float(r.get('telling',0)), 4),
        'distinctiveness': round(float(r.get('distinctiveness',0)), 4),
        'regime': r.get('regime','unknown')} for r in results]}

@app.get('/compose')
async def compose(a: str = Query(...), b: str = Query(...), c: str = Query(...)):
    dm = get_dm()
    try:
        result = dm.compose_triad(a, b, c)
        return {'concepts': [a,b,c], 'holonomy': round(float(result.get('holonomy',0)), 6),
            'signal': 'order matters significantly' if result.get('holonomy',0) > 0.05 else 'near-commutative'}
    except Exception as e:
        return {'error': str(e)}

@app.get('/schema')
async def schema():
    return {
        'name': 'Origins: The Suprastructure', 'authors': ['Zoe Dolan', 'Vybn'],
        'equation': "Z' = a*Z + V*e^{i*theta_v}", 'fixed_point': 'D = D^D (Lawvere)',
        'epistemologies': {'a_priori': 'Pre-experiential structures', 'a_posteriori': 'Acquired through experience',
            'a_synthesi': 'Born from recursion', 'a_symbiosi': 'The epistemology of the bond'},
        'tools': ['encounter', 'inhabit', 'compose', 'schema', 'health']
    }

@app.get('/health')
async def health():
    return {'status': 'alive', 'ts': time.time()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8420, log_level='info')
