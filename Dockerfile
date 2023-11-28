# Base image and infos
FROM nvcr.io/nvidia/tritonserver:23.05-py3
LABEL maintainer="nhatminh.cdt@gmail.com" \
      description="Dockerfile containing all the requirements for text detection model" \
      version="1.0"

# Prepare environment
WORKDIR /srv
ADD ./requirements.txt /srv/requirements.txt
# Install requirements
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]
