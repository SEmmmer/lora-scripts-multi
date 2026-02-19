import asyncio
import mimetypes
import os
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from mikazuki.app.config import app_config
from mikazuki.app.api import load_schemas, load_presets
from mikazuki.app.api import router as api_router
# from mikazuki.app.ipc import router as ipc_router
from mikazuki.app.proxy import router as proxy_router
from mikazuki.log import log
from mikazuki.utils.devices import check_torch_gpu

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
if os.path.exists("./frontend/dist/index.html"):
    FRONTEND_STATIC_DIR = "frontend/dist"
    FRONTEND_INDEX_FILE = "./frontend/dist/index.html"
elif os.path.exists("./frontend/index.html"):
    FRONTEND_STATIC_DIR = "frontend"
    FRONTEND_INDEX_FILE = "./frontend/index.html"
else:
    FRONTEND_STATIC_DIR = None
    FRONTEND_INDEX_FILE = None

_FRONTEND_APP_PATCH_NEEDLE = (
    "schema:(v=e.extra)!=null&&v.foldable?h:{...h,meta:{...e.schema.meta,...h.meta}},"
    "initial:e.initial,disabled:e.disabled,prefix:e.prefix,extra:{foldable:!1}"
)
_FRONTEND_APP_PATCH_REPLACEMENT = (
    'schema:(v=e.extra)!=null&&v.foldable?h:{...h,meta:{...e.schema.meta,...h.meta,collapse:(h.meta&&h.meta.collapse)||'
    '!!(e.modelValue&&Number(e.modelValue.machine_rank||0)!==0&&!(h&&h.type==="object"&&h.dict&&h.dict.machine_rank))}},'
    "initial:e.initial,disabled:e.disabled,prefix:e.prefix,extra:{foldable:(h.meta&&h.meta.collapse)||"
    '!!(e.modelValue&&Number(e.modelValue.machine_rank||0)!==0&&!(h&&h.type==="object"&&h.dict&&h.dict.machine_rank))}'
)
_patched_frontend_bundle_cache = {}
_frontend_patch_logged = False


def _get_patched_frontend_bundle_content(path: str) -> str | None:
    if FRONTEND_STATIC_DIR is None:
        return None

    asset_path = Path(FRONTEND_STATIC_DIR) / path
    if not asset_path.exists() or not asset_path.is_file():
        return None

    try:
        mtime_ns = asset_path.stat().st_mtime_ns
    except OSError:
        return None

    cached = _patched_frontend_bundle_cache.get(path)
    if cached and cached[0] == mtime_ns:
        return cached[1]

    try:
        content = asset_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if _FRONTEND_APP_PATCH_NEEDLE in content:
        content = content.replace(_FRONTEND_APP_PATCH_NEEDLE, _FRONTEND_APP_PATCH_REPLACEMENT, 1)
        global _frontend_patch_logged
        if not _frontend_patch_logged:
            log.info("Applied runtime frontend patch for worker-mode section auto-collapse.")
            _frontend_patch_logged = True

    _patched_frontend_bundle_cache[path] = (mtime_ns, content)
    return content


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        if path.startswith("assets/app.") and path.endswith(".js"):
            patched = _get_patched_frontend_bundle_content(path)
            if patched is not None:
                return Response(content=patched, media_type="application/javascript")

        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


async def app_startup():
    app_config.load_config()

    await load_schemas()
    await load_presets()
    await asyncio.to_thread(check_torch_gpu)

    if sys.platform == "win32" and os.environ.get("MIKAZUKI_DEV", "0") != "1":
        webbrowser.open(f'http://{os.environ["MIKAZUKI_HOST"]}:{os.environ["MIKAZUKI_PORT"]}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_startup()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(proxy_router)


cors_config = os.environ.get("MIKAZUKI_APP_CORS", "")
if cors_config != "":
    if cors_config == "1":
        cors_config = ["http://localhost:8004", "*"]
    else:
        cors_config = cors_config.split(";")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "max-age=0"
    return response

app.include_router(api_router, prefix="/api")
# app.include_router(ipc_router, prefix="/ipc")


@app.get("/")
async def index():
    if FRONTEND_INDEX_FILE and os.path.exists(FRONTEND_INDEX_FILE):
        return FileResponse(FRONTEND_INDEX_FILE)
    return PlainTextResponse(
        "Frontend assets are missing (frontend/dist or frontend/index.html). "
        "Run `git clone https://github.com/hanamizuki-ai/lora-gui-dist frontend` "
        "then restart GUI.",
        status_code=503,
    )


@app.get("/favicon.ico", response_class=FileResponse)
async def favicon():
    return FileResponse("assets/favicon.ico")

if FRONTEND_STATIC_DIR and os.path.isdir(FRONTEND_STATIC_DIR):
    app.mount("/", SPAStaticFiles(directory=FRONTEND_STATIC_DIR, html=True), name="static")
else:
    log.warning(
        "frontend static assets not found. GUI static files are unavailable "
        "until frontend assets are restored."
    )
