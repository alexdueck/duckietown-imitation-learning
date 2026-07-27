#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Forward a macOS-connected PS4 controller to the Duckietown ROS container."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import socket
import sys
from time import monotonic, sleep

from dt_utils.cli_completion import parse_args_with_completion
from dt_utils.duckiebot_teleop_input import SDLControllerInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a PS4 controller on the Mac and serve its state to "
            "physical_duckiebot_teleop.py in Docker Desktop."
        )
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--controller-index", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="List SDL controllers and exit.")
    parser.add_argument("--throttle-axis", type=int, default=1)
    parser.add_argument("--steering-axis", type=int, default=0)
    parser.add_argument("--arm-button", type=int, default=0)
    parser.add_argument("--record-button", type=int, default=6)
    parser.add_argument("--emergency-button", type=int, default=1)
    parser.add_argument("--clear-button", type=int, default=3)
    return parse_args_with_completion(parser)


def _controllers(pygame_module, controller_module) -> list:
    controller_module.init()
    controllers = []
    for index in range(controller_module.get_count()):
        if controller_module.is_controller(index):
            controllers.append(controller_module.Controller(index))
    return controllers


def _wait_for_connection(server, pygame_module):
    server.settimeout(0.1)
    while True:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT or (
                event.type == pygame_module.KEYDOWN
                and event.key == pygame_module.K_ESCAPE
            ):
                return None
        try:
            return server.accept()
        except socket.timeout:
            continue


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        print("error: --port must be between 1 and 65535", file=sys.stderr)
        return 2
    if args.rate <= 0:
        print("error: --rate must be positive", file=sys.stderr)
        return 2
    try:
        import pygame
        from pygame._sdl2 import controller as sdl_controller
    except ImportError:
        print(
            "error: pygame is required on the Mac. In this repository, use "
            "python3.11 or install requirements/gym-duckietown.txt.",
            file=sys.stderr,
        )
        return 2

    pygame.init()
    try:
        pygame.display.set_mode((680, 140))
        pygame.display.set_caption("PS4 → Duckiebot bridge")
        controllers = _controllers(pygame, sdl_controller)
        if args.list:
            if not controllers:
                print("No SDL controllers found.")
            for index, controller in enumerate(controllers):
                print(
                    f"{index}: {controller.name} "
                    f"({pygame.CONTROLLER_AXIS_MAX} standardized axes, "
                    f"{pygame.CONTROLLER_BUTTON_MAX} standardized buttons)"
                )
            return 0 if controllers else 1
        if not 0 <= args.controller_index < len(controllers):
            raise RuntimeError(
                f"Controller index {args.controller_index} is unavailable; "
                f"SDL found {len(controllers)} controller(s)"
            )
        device = controllers[args.controller_index]
        controller = SDLControllerInput(
            pygame,
            device,
            throttle_axis=args.throttle_axis,
            steering_axis=args.steering_axis,
            arm_button=args.arm_button,
            recording_button=args.record_button,
            emergency_button=args.emergency_button,
            clear_button=args.clear_button,
        )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", args.port))
            server.listen(1)
            print(
                f"Controller: {device.name} "
                f"({pygame.CONTROLLER_AXIS_MAX} standardized axes, "
                f"{pygame.CONTROLLER_BUTTON_MAX} standardized buttons)"
            )
            print(f"Waiting for Duckietown container on TCP port {args.port} ...")
            accepted = _wait_for_connection(server, pygame)
            if accepted is None:
                return 0
            connection, address = accepted
            with connection:
                print(f"Container connected from {address[0]}.")
                print(
                    "Cross: arm  Options: record  Circle: E-stop  "
                    "Triangle: clear  Escape: quit"
                )
                next_update_at = monotonic()
                while True:
                    events = pygame.event.get()
                    state = controller.poll(events)
                    triggered = [
                        label
                        for enabled, label in (
                            (state.arm_toggle, "ARM TOGGLE"),
                            (state.recording_toggle, "RECORD TOGGLE"),
                            (state.emergency_stop, "EMERGENCY STOP"),
                            (state.clear_emergency_stop, "CLEAR E-STOP"),
                        )
                        if enabled
                    ]
                    if triggered:
                        print(
                            "Controller event: "
                            + ", ".join(triggered)
                            + f" (throttle={state.throttle:+.2f}, "
                            f"steering={state.steering:+.2f})",
                            flush=True,
                        )
                    payload = json.dumps(
                        asdict(state), separators=(",", ":")
                    ).encode() + b"\n"
                    connection.sendall(payload)
                    pygame.display.set_caption(
                        "PS4 → Duckiebot | "
                        f"throttle={state.throttle:+.2f} "
                        f"steering={state.steering:+.2f}"
                    )
                    if state.quit:
                        break
                    next_update_at += 1.0 / args.rate
                    sleep(max(0.0, next_update_at - monotonic()))
    except KeyboardInterrupt:
        print("\nBridge stopped.")
        return 0
    except (BrokenPipeError, ConnectionError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
