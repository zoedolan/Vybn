from fastapi import FastAPI, HTTPException
from .utils import get_db, get_index, kv_get, kv_put, vec_search

app = FastAPI()


@app.on_event("startup")
def startup():
    # ensure db and index exist
    get_db()
    get_index()


@app.get("/kv/{key}")
def read_kv(key: str):
    value = kv_get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"value": value}


@app.put("/kv/{key}")
def write_kv(key: str, payload: dict):
    if "value" not in payload:
        raise HTTPException(status_code=400, detail="Missing value")
    kv_put(key, payload["value"])
    return {"status": "ok"}


@app.post("/vec/search")
def search_vec(payload: dict):
    query = payload.get("query")
    k = int(payload.get("k", 5))
    if query is None:
        raise HTTPException(status_code=400, detail="Missing query")
    results = vec_search(query, k)
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mesh.server:app", host="0.0.0.0", port=8000)
