import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = {path.stem: path for path in sorted((PROJECT_ROOT / "configs").glob("*.json"))}
LABELS = {
    "pretrain": "预训练",
    "sft": "监督微调 SFT",
    "dpo": "偏好训练 DPO",
    "data_pipeline": "数据流水线",
    "label": "语料清洗",
}

app = FastAPI(title="训练配置面板")


class ConfigUpdate(BaseModel):
    data: dict


def config_file(config_id: str) -> Path:
    if config_id not in CONFIG_FILES:
        raise HTTPException(404, "配置不存在")
    return CONFIG_FILES[config_id]


def config_info(config_id: str) -> dict:
    path = config_file(config_id)
    return {"id": config_id, "label": LABELS.get(config_id, config_id), "filename": path.name}


@app.get("/api/configs")
def list_configs():
    return {"configs": [config_info(config_id) for config_id in CONFIG_FILES]}


@app.get("/api/configs/{config_id}")
def read_config(config_id: str):
    path = config_file(config_id)
    return {**config_info(config_id), "data": json.loads(path.read_text(encoding="utf-8"))}


@app.put("/api/configs/{config_id}")
def save_config(config_id: str, update: ConfigUpdate):
    path = config_file(config_id)
    path.write_text(json.dumps(update.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**config_info(config_id), "data": update.data}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")
