import argparse
import logging
import pickle
import socketserver
import struct
import threading
from typing import Any

import numpy as np

from astra_controller.astra_controller import AstraController


JOINT_ACTION_DIM = 18
ALLOWED_METHODS = {"connect", "disconnect", "wait_for_reset", "get_observation", "get_action", "send_action"}


def _recv_exact(sock, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock) -> Any:
    header = _recv_exact(sock, 4)
    (size,) = struct.unpack("!I", header)
    return pickle.loads(_recv_exact(sock, size))


def _send_message(sock, message: Any) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _as_float32(values):
    return np.asarray(values, dtype=np.float32)


class AstraLeRobotBridge:
    def __init__(self, space: str = "joint"):
        if space != "joint":
            raise ValueError("Only joint space is supported by the LeRobot bridge.")

        self.controller = AstraController(space=space)
        self.lock = threading.Lock()

    def connect(self):
        with self.lock:
            self.controller.connect()
        return {"connected": True}

    def disconnect(self):
        with self.lock:
            self.controller.disconnect()
        return {"connected": False}

    def wait_for_reset(self):
        with self.lock:
            self.controller.wait_for_reset()
        return {"reset": True}

    def get_observation(self):
        with self.lock:
            (
                state,
                state_arm_l,
                state_gripper_l,
                state_arm_r,
                state_gripper_r,
                state_base,
                state_eef_l,
                state_eef_r,
                state_odom,
                state_head,
            ) = self.controller.read_present_position()
            images = self.controller.read_cameras()
            done = bool(self.controller.done)
            reset = bool(self.controller.reset)
            self.controller.done = False

        return {
            "state": _as_float32(state),
            "state_arm_l": _as_float32(state_arm_l),
            "state_gripper_l": _as_float32(state_gripper_l),
            "state_arm_r": _as_float32(state_arm_r),
            "state_gripper_r": _as_float32(state_gripper_r),
            "state_base": _as_float32(state_base),
            "state_eef_l": _as_float32(state_eef_l),
            "state_eef_r": _as_float32(state_eef_r),
            "state_odom": _as_float32(state_odom),
            "state_head": _as_float32(state_head),
            "images": images,
            "done": done,
            "reset": reset,
        }

    def get_action(self):
        with self.lock:
            (
                action,
                action_arm_l,
                action_gripper_l,
                action_arm_r,
                action_gripper_r,
                action_base,
                action_eef_l,
                action_eef_r,
                action_head,
            ) = self.controller.read_leader_present_position()

        return {
            "action": _as_float32(action),
            "action_arm_l": _as_float32(action_arm_l),
            "action_gripper_l": _as_float32(action_gripper_l),
            "action_arm_r": _as_float32(action_arm_r),
            "action_gripper_r": _as_float32(action_gripper_r),
            "action_base": _as_float32(action_base),
            "action_eef_l": _as_float32(action_eef_l),
            "action_eef_r": _as_float32(action_eef_r),
            "action_head": _as_float32(action_head),
        }

    def send_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (JOINT_ACTION_DIM,):
            raise ValueError(f"Expected {JOINT_ACTION_DIM} joint actions, got shape {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError("Action contains NaN or Inf.")

        with self.lock:
            self.controller.write_goal_position(action.tolist())

        return {"action": action}


class BridgeRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        bridge: AstraLeRobotBridge = self.server.bridge
        while True:
            try:
                request = _recv_message(self.request)
            except ConnectionError:
                return

            try:
                method = request["method"]
                params = request.get("params", {})
                if method not in ALLOWED_METHODS:
                    raise ValueError(f"Unknown bridge method: {method}")
                result = getattr(bridge, method)(**params)
                response = {"ok": True, "result": result}
            except Exception as exc:
                logging.exception("Astra LeRobot bridge request failed")
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            _send_message(self.request, response)


class ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, request_handler_class, bridge):
        super().__init__(server_address, request_handler_class)
        self.bridge = bridge


def main():
    parser = argparse.ArgumentParser(description="Astra ROS bridge for LeRobot Python 3.12 clients.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--space", default="joint", choices=["joint"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bridge = AstraLeRobotBridge(space=args.space)
    server = ThreadingTCPServer((args.host, args.port), BridgeRequestHandler, bridge)
    logging.info("Astra LeRobot bridge listening on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
