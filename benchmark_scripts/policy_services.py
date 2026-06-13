import zmq
# import torch
from io import BytesIO
from typing import Any, Callable, Dict
import pickle
# class TorchSerializer:
#     @staticmethod
#     def to_bytes(data: dict) -> bytes:
#         buffer = BytesIO()
#         torch.save(data, buffer)
#         return buffer.getvalue()

#     @staticmethod
#     def from_bytes(data: bytes) -> dict:
#         buffer = BytesIO(data)
#         obj = torch.load(buffer, weights_only=False)
#         return obj

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return pickle.dumps(data)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return pickle.loads(data)

class BaseInferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        # self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        # self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    # def call_endpoint(
    #     self, endpoint: str, data: dict | None = None, requires_input: bool = True
    # ) -> dict:
    def call_endpoint(
        self, endpoint: str, data: dict = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
     
        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)



    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """
# dict_keys(['video.left_view', 'video.right_view', 'video.wrist_view', 
# 'state.end_effector_position_relative', 'state.end_effector_rotation_relative', 
# 'state.gripper_qpos', 'state.base_position', 'state.base_rotation', 'annotation.human.action.task_description'])
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)
    
    # def getadress(self, observations: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Get the action from the server.
    #     The exact definition of the observations is defined
    #     by the policy, which contains the modalities configuration.
    #     """
    #     return self.call_endpoint("get_action", observations)