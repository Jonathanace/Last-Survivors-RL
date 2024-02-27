FROM tensorflow/tensorflow

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt \
  && pip install --user tf-agents[reverb] \
  && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
  && apt-get install python3-tk -y 
  # && apt-get update && apt-get install -y firefox \
  # && apt-get update && apt-get install -y firefox x11vnc xvfb \
  # && echo "exec firefox" > ~/.xinitrc && chmod +x ~/.xinitrc 

CMD ["v11vnc", "-create", "-forever"]

COPY . .

# CMD ["python", "./your-daemon-or-script.py"]