#################################################################
FROM python:3.13-slim

WORKDIR /app

# COPY ./resources/requirements.txt /app
# COPY ./resources/copilote-AV.py /app/main.py
COPY ./resources/entrypoint.sh /app

# RUN apt-get update && \
#     apt-get install -y python3-opencv ffmpeg && \
#     pip install -r /app/requirements.txt

RUN apt-get update && \
    apt-get install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
      gir1.2-gst-rtsp-server-1.0 python3-gi python3-gi-cairo &&\
    python3 -m pip install opencv-python ultralytics numpy

ENTRYPOINT [ "./entrypoint.sh" ]
CMD [ "python" ]
# CMD [ "/usr/local/bin/python", "main.py", "--in 'rtsp://admin:tEsTgfhjkm1729@10.0.0.21:554/cam/realmonitor?channel=1&subtype=1'", "--mount '/yolo'", "--listen 0.0.0.0", "--port 8554", "--model yolov8n.pt", "--conf 0.4", "--iou 0.5" ]