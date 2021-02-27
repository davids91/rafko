#!/bin/bash
protoc "../proto/logger.proto" --proto_path="../proto" --python_out="."
