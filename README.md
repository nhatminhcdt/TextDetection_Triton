## Deploying Text Detection Model with Triton Inference Server

This project is to demonstrate how to use NVIDIA Triton Inference Server to deploy a **text detection** and **recognition model** ensemble. The text detection model is based on [OpenCV Text Detection (EAST text detector)](https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/) and the text recognition model is based on [Deep Text Recognition](https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/README.md)

## Triton Model Ensembles

To reduce the number of network calls, we can combine the text detection and text recognition models as well as the image pre/postprocessing into a single ensemble model. The ensemble model will take the full image as input and return the parsed text as output. The ensemble model will execute multiple models in a single network call.


**Network Call**

```mermaid
sequenceDiagram
    Client ->> Triton: Full Image
    activate Triton
    activate Triton
    Note right of Triton: Text Detection
    deactivate Triton
    activate Triton
    Note right of Triton: Image Cropping (Serverside)
    Note left of Triton: Ensemble Model
    deactivate Triton
    activate Triton
    Note right of Triton: Text Recognition
    Triton ->> Client: Parsed Text
    deactivate Triton
    deactivate Triton
```

**Directed Acyclic Graph**

Below is a graphical representation of our model pipeline. The diamonds represent the final input and output of the ensemble, which is all the client will interact with. The circles are the different deployed models, and the rectangles are the tensors that get passed between models.

![DAG](./docs/pics/TextDetection_DAG.png "Text Detection - DAG")

The configuration of model ensembles is defined in `/ensemble_model/config.pbtxt`.

<details>
<summary>Expand for ensemble config file</summary>

```
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 0
input [
  {
    name: "input_image"
    data_type: TYPE_UINT8
    dims: [ -1, -1 ]
  }
]
output [
  {
    name: "recognized_text"
    data_type: TYPE_STRING
    dims: [ -1, -1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "detection_preprocessing"
      model_version: -1
      input_map {
        key: "detection_preprocessing_input"
        value: "input_image"
      }
      output_map {
        key: "detection_preprocessing_output"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "text_detection"
      model_version: -1
      input_map {
        key: "input_images:0"
        value: "preprocessed_image"
      }
      output_map {
        key: "feature_fusion/Conv_7/Sigmoid:0"
        value: "Sigmoid:0"
      },
      output_map {
        key: "feature_fusion/concat_3:0"
        value: "concat_3:0"
      }
    },
    {
      model_name: "detection_postprocessing"
      model_version: -1
      input_map {
        key: "detection_postprocessing_input_1"
        value: "Sigmoid:0"
      }
      input_map {
        key: "detection_postprocessing_input_2"
        value: "concat_3:0"
      }
      input_map {
        key: "detection_postprocessing_input_3"
        value: "preprocessed_image"
      }
      output_map {
        key: "detection_postprocessing_output"
        value: "cropped_images"
      }
    },
    {
      model_name: "text_recognition"
      model_version: -1
      input_map {
        key: "input.1"
        value: "cropped_images"
      }
      output_map {
        key: "307"
        value: "recognition_output"
      }
    },
    {
      model_name: "recognition_postprocessing"
      model_version: -1
      input_map {
        key: "recognition_postprocessing_input"
        value: "recognition_output"
      }
      output_map {
        key: "recognition_postprocessing_output"
        value: "recognized_text"
      }
    }
  ]
}
```

</details>

## Deployment

There are 3 different python backend models that will be served with Triton:
>1. `detection_preprocessing`
>2. `detection_postprocessing`
>3. `recognition_postprocessing`

To deploy Text Detection and Text Recognition models, we need to convert the models to ONNX format and then create a folder for each model in the `model_repository` folder.

**Step 1: Deploy the Text Detection and Text Recognition Models**

- Launch NGC TensorFlow container environment with **docker** (Recommended)

`docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:22.01-tf2-py3`

then execute:

`bash utils/export_text_detection.sh`

- On another terminal, launch NGC TensorFlow container environment with **docker** (Recommended)

`docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:22.01-py3`

then execute:

`bash utils/export_text_recognition.sh`


**Step 2: Launch Triton**
- Launching the triton server

```
docker run --gpus=all -it --shm-size=1g --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.05-py3
```

- Install a couple of dependencies for our Python backend scripts.

`pip install opencv-python-headless Pillow`

- Launch Triton

`tritonserver --model-repository=/models`


## Inference

On command line, run the client script.

`python3 client.py`

Note: Using port 8000 for HTTP, 8001 for GRPC, and 8002 for metrics.


## Tips

1. An easy way to see the structure with a graphical interface is by using tools like [netron](https://netron.app/).


## Troubleshooting

**1. Error when converting ***text_detection*** model to ONNX format**
>`python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx`
>`ValueError: Could not infer attribute explicit_paddings type from empty iterator`

**Solution:**
Installing tf2onnx via pip will install onnx:1.15.0. You can verify that by running pip list | grep onnx
In order to downgrade the version of your onnx dependency, just run `pip install onnx==1.14.1`. ([REF](https://github.com/onnx/tensorflow-onnx/issues/2262))

**2. Error when make inference**

>`UNAVAILABLE - Inference request for unknown model 'ensemble_model'`

**Solution:**
Create version folder for the ensemble model. The version folder can be empty as ensemble doesn't require any model file.

>`tritonclient.utils.InferenceServerException: [400] in ensemble 'ensemble_model', Failed to process the request(s) for model instance 'detection_preprocessing_0', message: Failed to increase the shared memory pool size for key 'triton_python_backend_shm_region_6' to 67108864 bytes.`

**Solution:**
Increase `--shm-size` value when launching triton server, e.g: `--shm-size=1g`

3. Error when pushing docker image to GCR
>`docker push asia.grc.io/mles-class/textdetection_triton:v1.0`
>-> net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)

**Solution:**


## For more information
| [Triton Model Deployment](https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_1-model_deployment/README.md) | [Triton Model Ensembles](https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_5-Model_Ensembles/README.md) |
| :--- | :--- |


## Logs

- 2023/11/22: Package the project into a docker container
- 2023/11/21: Replace torchvision with Pillow for image preprocessing
- 2023/11/20: Initial commit