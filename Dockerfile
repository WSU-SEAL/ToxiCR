FROM python:3.8.20-bullseye

COPY requirements.txt /
RUN cd /; pip install -r requirements.txt

VOLUME /toxicr
WORKDIR /toxicr

ENTRYPOINT ["/bin/bash"]
