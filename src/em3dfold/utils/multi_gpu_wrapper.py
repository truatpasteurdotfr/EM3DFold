import torch
import torch.nn as nn
from typing import List
from collections import namedtuple
import os
from contextlib import nullcontext
import queue
import threading

from em3dfold.utils.misc_utils import filter_useless_warnings

# Currently hard-coded, can change later
if os.environ.get("MASTER_ADDR") is None:
    os.environ["MASTER_ADDR"] = "localhost"
if os.environ.get("MASTER_PORT") is None:
    os.environ["MASTER_PORT"] = "29500"


InferenceData = namedtuple(
    "InferenceData",
    ["data", "status"]
)


def is_iterable(object) -> bool:
    try:
        iter(object)
        return True
    except:
        return False


def send_dict_to_device(dictionary, device: str):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            dictionary[key] = dictionary[key].to(device)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(device) for x in dictionary[key]]
    return dictionary


def clone_dict_to_cpu(dictionary):
    output = {}
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            output[key] = dictionary[key].detach().to("cpu")
        elif is_iterable(dictionary[key]):
            output[key] = [
                x.detach().to("cpu") if torch.is_tensor(x) else x
                for x in dictionary[key]
            ]
        else:
            output[key] = dictionary[key]
    return output


def cast_dict_to_half(dictionary):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            if dictionary[key].dtype == torch.float32:
                dictionary[key] = dictionary[key].to(torch.float16)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(torch.float16) for x in dictionary[key] if x.dtype == torch.float32]
    return dictionary


def cast_dict_to_full(dictionary):
    for key in dictionary:
        if torch.is_tensor(dictionary[key]):
            if dictionary[key].dtype == torch.float16:
                dictionary[key] = dictionary[key].to(torch.float32)
        elif is_iterable(dictionary[key]):
            dictionary[key] = [x.to(torch.float32) for x in dictionary[key] if x.dtype == torch.float16]
    return dictionary

def init_model(model_class, model_args, state_dict_path: str, device: str) -> nn.Module:
    model = model_class(**model_args).eval()
    checkpoint = torch.load(state_dict_path, map_location="cpu")
    if "model" not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model

def run_inference(
        rank_id: int,
        model_class,
        model_args,
        state_dict_path: str,
        devices: List[str],
        input_queue,
        output_queue,
        dtype: torch.dtype = torch.float32,
):
    device = devices[rank_id]
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    model = init_model(model_class, model_args, state_dict_path, device)
    filter_useless_warnings()

    while True:
        with torch.no_grad():
            try:
                inference_data = input_queue.get()
                if inference_data.status != 1:
                    break
                model_inputs = send_dict_to_device(inference_data.data, device)
                if dtype == torch.float16:
                    model_inputs = cast_dict_to_half(model_inputs)
                autocast_ctx = (
                    torch.cuda.amp.autocast(dtype=dtype)
                    if str(device).startswith("cuda") and dtype == torch.float16
                    else nullcontext()
                )
                with autocast_ctx:
                    output = model(**model_inputs)
                output = output.to("cpu").to(torch.float32)
                output_queue.put(output)
            except Exception as e:
                output_queue.put(e)
                return


class MultiGPUWrapper(nn.Module):
    def __init__(
            self,
            model_class,
            model_args,
            state_dict_path: str,
            devices: List[str],
            fp16: bool = False
    ):
        super().__init__()
        self.input_queues = []
        self.output_queues = []
        self.workers = []
        self.world_size = len(devices)
        self.devices = devices
        self.dtype = torch.float32 if not fp16 else torch.float16

        if self.world_size > 1:
            for rank_id in range(self.world_size):
                input_queue = queue.Queue(maxsize=1)
                output_queue = queue.Queue(maxsize=1)
                worker = threading.Thread(
                    target=run_inference,
                    args=(
                        rank_id,
                        model_class,
                        model_args,
                        state_dict_path,
                        devices,
                        input_queue,
                        output_queue,
                        self.dtype,
                    ),
                    daemon=True,
                )
                worker.start()
                self.input_queues.append(input_queue)
                self.output_queues.append(output_queue)
                self.workers.append(worker)
        else:
            self.model = init_model(model_class, model_args, state_dict_path, devices[0])

    def forward(self, data_list: List) -> List:
        output_list = []
        for i, data in enumerate(data_list):
            device = self.devices[i]
            if self.world_size > 1:
                input_queue = self.input_queues[i]
                input_queue.put(
                    InferenceData(data=clone_dict_to_cpu(data), status=1)
                )
            else:
                model_inputs = send_dict_to_device(clone_dict_to_cpu(data), device)
                if self.dtype == torch.float16:
                    model_inputs = cast_dict_to_half(model_inputs)
                autocast_ctx = (
                    torch.cuda.amp.autocast(dtype=self.dtype)
                    if str(device).startswith("cuda") and self.dtype == torch.float16
                    else nullcontext()
                )
                with autocast_ctx, torch.no_grad():
                    output_list.append(
                        self.model(**model_inputs).to("cpu").to(torch.float32)
                    )
        if self.world_size > 1:
            for output_queue, _ in zip(self.output_queues, data_list):
                output = output_queue.get()
                if isinstance(output, Exception):
                    raise output
                output_list.append(output)
        return output_list


    def __del__(self):
        if self.world_size > 1:
            for input_queue in self.input_queues:
                try:
                    input_queue.put(InferenceData(data=None, status=0))
                except:
                    pass
            for worker in self.workers:
                worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.__del__()


