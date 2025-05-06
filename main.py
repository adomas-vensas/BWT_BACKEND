import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import FastAPI
from API import endpoints, websockets

app = FastAPI()

app.include_router(endpoints.router)
app.include_router(websockets.router)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
app.add_middleware(GZipMiddleware, minimum_size=1000)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=7910, reload=True)