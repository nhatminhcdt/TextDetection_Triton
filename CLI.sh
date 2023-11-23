# --GCLOUD--
# For login:
gcloud auth login
gcloud auth configure-docker
gcloud projects list
gcloud services enable containerregistry.googleapis.com --project=mles-class-01


#---DOCKER--
# Notice: Assume that the gcloud project is already set to mles-class-01
# To build docker image
docker build ./ -f Dockerfile -t asia.grc.io/mles-class-01/textdetection_triton:v1.0
# To run docker image locally
docker run --name TextDetection_Triton --gpus=all -it --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models asia.grc.io/mles-class-01/textdetection_triton:v1.0
# To push docker image to Google Cloud Registry (GCR)
docker push asia.grc.io/mles-class-01/textdetection_triton:v1.0
