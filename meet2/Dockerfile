FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-avatar.txt ./
ARG AVATAR=1
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$AVATAR" = "1" ]; then pip install --no-cache-dir -r requirements-avatar.txt; fi
COPY meet ./meet
COPY web ./web
COPY config ./config
ENV MEET_HOST=0.0.0.0 MEET_PORT=8600 MEET_KB_DIR=/data/kb MEET_LOG_DIR=/data/rooms
VOLUME /data
EXPOSE 8600
CMD ["python", "-m", "meet.server"]
