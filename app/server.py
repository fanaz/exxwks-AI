from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from pathlib import Path
import hashlib

from fastai.vision import *

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def setup_learner():
    learn = load_learner('app/models', 'export.pkl')
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)
    
    filename = data['file'].filename
    prediction_str = str(prediction[0])
    with open('datasets/count_app/' + prediction_str + '/' + prediction_str + '_' + hashlib.md5(img_bytes).hexdigest() + '_' + filename, 'wb') as filehandle:  
        filehandle.write(img_bytes)
    
    return JSONResponse({
        'result': str(prediction[0]),
        'scores': sorted(zip(learn.data.classes, map(float, prediction[2])), key=lambda p: p[1], reverse=True)
    })

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8088)