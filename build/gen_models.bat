protoc common.proto --proto_path="../proto/" --cpp_out="../cxx/gen/"
protoc sparse_net.proto --proto_path="../proto/" --proto_path="../proto/models/" --cpp_out="../cxx/gen/"
protoc solution.proto --proto_path="../proto/" --proto_path="../proto/models/" --cpp_out="../cxx/gen/"
protoc training.proto --proto_path="../proto/" --proto_path="../proto/models/" --cpp_out="../cxx/gen/"

mkdir "../lib"