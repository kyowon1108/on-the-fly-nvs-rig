#
# Standalone viewer for saved on-the-fly-nvs models
#

import os
import time
import argparse
from threading import Thread
from socketserver import TCPServer
from http.server import SimpleHTTPRequestHandler

from scene.scene_model import SceneModel
from webviewer.webviewer import WebViewer


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "webviewer"), **kwargs)


def main():
    parser = argparse.ArgumentParser(description="View saved on-the-fly-nvs model")
    parser.add_argument("scene_dir", type=str, help="Path to saved model directory")
    parser.add_argument("--anchor_overlap", type=float, default=0.3)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--http_port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading model from {args.scene_dir}...")
    scene_model = SceneModel.from_scene(args.scene_dir, args)
    print(f"Loaded {len(scene_model.keyframes)} keyframes, {len(scene_model.anchors)} anchors")

    # Start HTTP server for web viewer
    http_server = TCPServer((args.ip, args.http_port), ViewerHandler)
    http_thread = Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    print(f"HTTP server started at http://{args.ip}:{args.http_port}/")

    # Start WebSocket viewer
    viewer = WebViewer(scene_model, args.ip, args.port)
    viewer.trainer_state = "finish"  # Mark as finished training
    print(f"WebSocket viewer started at ws://{args.ip}:{args.port}")
    print(f"\nOpen your browser and go to: http://{args.ip}:{args.http_port}/")

    viewer.run()


if __name__ == "__main__":
    main()
