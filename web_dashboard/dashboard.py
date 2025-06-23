from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
uploaded_images = []

@app.post("/upload_result/")
async def upload_result(
    image: UploadFile = File(...),
    filename: str = Form(...),
    matches: str = Form(...)
):
    save_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    uploaded_images.append((image.filename, matches))
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def home():
    html = """
    <html>
    <head>
        <title>📸 이상 이미지 대시보드</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background-color: #f5f7fa;
                color: #333;
                padding: 20px;
            }
            h1 {
                font-size: 2.2em;
                color: #d9534f;
            }
            h2 {
                margin-top: 40px;
                color: #5a5a5a;
            }
            .imgbox {
                display: inline-block;
                width: 300px;
                margin: 15px;
                text-align: center;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 10px;
                transition: transform 0.2s ease-in-out;
            }
            .imgbox:hover {
                transform: scale(1.05);
            }
            img {
                max-width: 100%;
                border-radius: 6px;
                border: 2px solid #ccc;
            }
            .info {
                margin-top: 8px;
                font-size: 0.9em;
                color: #666;
            }
            .refresh-note {
                font-size: 0.8em;
                color: #888;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <h1>🚨 이상 이미지 대시보드</h1>
        <div class="refresh-note">※ 이 페이지는 5초마다 자동 새로고침됩니다.</div>
    """

    abnormal_imgs = [(f, t) for f, t in uploaded_images if t == "이상 감지"]
    failed_imgs = [(f, t) for f, t in uploaded_images if t == "촬영 실패"]

    html += "<h2>🔴 이상 감지된 이미지</h2>"
    if abnormal_imgs:
        for fname, _ in abnormal_imgs:
            html += f'''
            <div class="imgbox">
                <img src="/uploads/{fname}">
                <div class="info">{fname}</div>
            </div>
            '''
    else:
        html += "<p>없음</p>"

    html += "<h2>⚠️ 촬영 실패 이미지</h2>"
    if failed_imgs:
        for fname, _ in failed_imgs:
            html += f'''
            <div class="imgbox">
                <img src="/uploads/{fname}">
                <div class="info">{fname}</div>
            </div>
            '''
    else:
        html += "<p>없음</p>"

    html += "</body></html>"
    return HTMLResponse(content=html)
