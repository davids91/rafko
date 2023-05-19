#!/bin/bash
# protoc "../cxx/rafko_protocol/logger.proto" --proto_path="../proto" --python_out="."
PROTO_LOCATION="../../cxx/rafko_protocol"
protoc "$PROTO_LOCATION/training.proto" --proto_path="$PROTO_LOCATION" --python_out="."
protoc "$PROTO_LOCATION/rafko_net.proto" --proto_path="$PROTO_LOCATION" --python_out="."
