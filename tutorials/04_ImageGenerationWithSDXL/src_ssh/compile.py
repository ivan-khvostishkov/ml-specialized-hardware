# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import subprocess
subprocess.check_call(['apt-mark', 'hold', 'ssh-import-id'])  # Temp measure to avoid python3-distro error on trn

# See https://github.com/aws-samples/sagemaker-ssh-helper#inference
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import sagemaker_ssh_helper
sagemaker_ssh_helper.setup_and_start_ssh()


import os
os.environ['NEURON_RT_NUM_CORES'] = '2'
import io
import json
import time
import shutil
import argparse
from filelock import FileLock
from optimum.neuron import NeuronStableDiffusionPipeline

lock_path = '/tmp/new_packages.lock'
lock = FileLock(lock_path)


def model_fn(model_dir, context=None):
    print("Waiting for the lock acquire...")    
    lock.acquire()
    print("Loading model...")
    t = time.time()    
    model = NeuronStableDiffusionPipeline.from_pretrained(model_dir,  device_ids=[0, 1])
    print(f"Model loaded. Elapsed: {time.time()-t}s")
    lock.release()
    return model

    
def input_fn(request_body, request_content_type, context=None):
    if request_content_type == 'application/json':
        req = json.loads(request_body)
        prompt = req.get('prompt')
        num_inference_steps = req.get('num_inference_steps', 50)
        guidance_scale = req.get('guidance_scale', 7.5)
        if prompt is None or type(prompt) != str or len(prompt) < 5:
            raise("Invalid prompt. It needs to be a string > 5")
        if type(num_inference_steps) != int:
            raise("Invalid num_inference_steps. Expected int. default = 50")
        if type(guidance_scale) != float:
            raise("Invalid guidance_scale. Expected float. default = 7.5")
        return prompt, num_inference_steps, guidance_scale
    else:
        raise Exception(f"Unsupported mime type: {request_content_type}. Supported: application/json")


def predict_fn(input_req, model, context=None):
    prompt, num_inference_steps, guidance_scale = input_req
    return model(
        prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
    ).images[0]


def output_fn(image, accept, context=None):
    if accept != 'image/jpeg':
        raise Exception(f'Invalid data type. Expected image/jpeg, got {accept}')

    buffer = io.BytesIO()
    image.save(buffer, 'jpeg', icc_profile=image.info.get('icc_profile'))
    buffer.seek(0)
    return buffer.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.    
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--auto_cast", type=str, default="matmul")
    parser.add_argument("--auto_cast_type", type=str, default="bf16")

    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args, _ = parser.parse_known_args()

    model_id = args.model_id
    compiler_args = {"auto_cast": args.auto_cast, "auto_cast_type": args.auto_cast_type}
    input_shapes = {"batch_size": args.batch_size, "height": args.height, "width": args.width}

    print("Compiling model...")    
    t = time.time()
    stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(
        model_id, export=True, **compiler_args, **input_shapes
    )
    print(f"Done. Elapsed time: {(time.time()-t) * 1000}s")
    print("Saving model...")
    t = time.time()
    stable_diffusion.save_pretrained(args.model_dir)
    print(f"Done. Elapsed time: {(time.time()-t) * 1000}s")

    code_path = os.path.join(args.model_dir, "code")
    os.makedirs(code_path, exist_ok=True)
    shutil.copy("compile.py", os.path.join(code_path, "inference.py"))
    shutil.copy("requirements.txt", os.path.join(code_path, "requirements.txt"))
    print(f"Job done!")
