# UCX Transport (Test)

Test project for connecting two nodes over a combination of TCP and UCX.

## Client
A sample client that contains a chunk of data readable by the server.

## Server
The server listens for TCP connections containing UCX setup data.
Clients can call to the server to register their shared regions, or prompt the server to copy data from the shared regions.
UCX facilitates RMA data transfer, allowing the client to progress while the server copies data.

## Deps
- pthread
- UCx
    - UCP
    - UCT
    - UCS
