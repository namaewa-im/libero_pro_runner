## Overview

This repository provides a lightweight client utility for interacting with the
`openvla-oft` server (https://github.com/moojink/openvla-oft) and is compatible
with LIBERO checkpoints.  
The model is hosted as an HTTP server, and this code acts as the client that communicates with the server via HTTP requests.

## Usage

1. Start the model server from `openvla-oft` (serves on port 8777 by default).
2. Run this client to send images, language instructions, and retrieve actions
   through HTTP communication.

