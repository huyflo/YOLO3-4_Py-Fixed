import time

from pydarknet import Detector, Image
import cv2
import argparse
from flask import Flask, render_template, Response
from utils.app_utils import FPS, WebcamVideoStream

app = Flask(__name__)
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video',
                help = 'path to video file containing class names',default='rtsp://192.168.1.13:8554/CH001.sdp')
                # help = 'path to video file containing class names',default='rtmp://sentirlite.com/live/f65731a782ec563842978e8472da0043')
                #help = 'path to video file containing class names',default='rtsp://vanbinh:Admin@2019@113.160.219.167')
                #help = 'path to video file containing class names',default='rtsp://vanbinh:Admin@2019@113.160.158.229')
args = ap.parse_args()


@app.route('/')
def index():
    return render_template('index.html')



net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("weights/yolov3.weights", encoding="utf-8"), 0,bytes("cfg/coco.data", encoding="utf-8"))
# net = Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0,bytes("cfg/coco.data", encoding="utf-8"))

def gen():
    # Optional statement to configure preferred GPU. Available only in GPU version.
    # pydarknet.set_cuda_device(0)
    while True:
        frame = video_capture.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        start_time = time.time()

        # Only measure the time taken by YOLO and API Call overhead

        dark_frame = Image(frame)
        results = net.detect(dark_frame)
        del dark_frame

        end_time = time.time()
        print("Elapsed Time  :",end_time-start_time)

        for cat, score, bounds in results:
            x, y, w, h = bounds
            cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
            cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
        ret, jpeg = cv2.imencode('.jpeg', frame)
        jpeg = jpeg.tobytes()
        # jpeg = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    video_capture = WebcamVideoStream(src=args.video, width=600, height=600).start()
    app.run(host='0.0.0.0', port=4000,debug=False)
