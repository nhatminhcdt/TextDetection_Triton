# Base image and infos
FROM nvcr.io/nvidia/tritonserver:23.05-py3
LABEL maintainer="nhatminh.cdt@gmail.com" \
      description="Dockerfile containing all the requirements for TextDetection_Triton" \
      version="1.0"

# Prepare environment
RUN mkdir /app
WORKDIR /app/server
COPY ./model_repository /app/server/model_repository
COPY ./requirements.txt /app/server/requirements.txt

# Install requirements
RUN pip install -r requirements.txt

# Run Triton Server
CMD ["tritonserver", "--model-repository=/models"]
