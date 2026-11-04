#!/usr/bin/env python3
"""Minimal MQTT 3.1.1 broker for in-sandbox experiments (amqtt).

The deliverable broker is Eclipse Mosquitto (docker-compose.yml, auth+ACL).
This script exists because the agent sandbox has no Docker/mosquitto binary;
it provides a REAL MQTT-over-TCP broker so X5/X7 measurements are real [M]
network measurements. Usage: python x5_broker.py [--port 1883]
"""
import argparse
import asyncio

from amqtt.broker import Broker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1883)
    args = ap.parse_args()
    config = {
        "listeners": {
            "default": {"type": "tcp", "bind": f"127.0.0.1:{args.port}"},
        },
        "sys_interval": 0,
        "auth": {"allow-anonymous": True},
        "plugins": {
            "amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True},
        },
    }

    async def run():
        broker = Broker(config)
        await broker.start()
        print(f"[broker] amqtt listening on 127.0.0.1:{args.port}", flush=True)
        while True:
            await asyncio.sleep(3600)

    asyncio.run(run())


if __name__ == "__main__":
    main()
