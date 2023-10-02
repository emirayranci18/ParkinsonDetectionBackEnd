from flask import Flask, request
from demo import func
import requests

app = Flask(__name__)
veri = ""

@app.route('/upload', methods=['POST'])
def upload_wav():
    wav_file = request.files['file']
    wav_file.save('recordingParkinsonPulled.pcm')

    try:
        veri = func()
        return veri
    except Exception as e:
        return "Hata oluştu. Farklı bir ses kaydı göndermeyi deneyin"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
