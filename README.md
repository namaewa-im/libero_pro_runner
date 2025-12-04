## Overview

This repository provides a lightweight client utility for interacting with the
`openvla-oft` server (https://github.com/moojink/openvla-oft) and supports
LIBERO checkpoints.  
The model is hosted as an HTTP server (default port: **8777**), and this code
acts as the client that communicates with the server via HTTP requests.

## Requirements

This project is designed to work **with LIBERO-PRO**.  
Please install LIBERO-PRO first:

https://github.com/Zxy-MLlab/LIBERO-PRO

After installing LIBERO-PRO, ensure that `openvla-oft` is installed and its
model server is running.

## Usage

1. Install and set up **LIBERO-PRO**.
2. Start the model server from `openvla-oft` (serves on port 8777 by default).
3. Run this client to send images and language instructions to the server and
   receive predicted actions via HTTP.

