
## Scaling in a nutshell

This document outlines the thestrategy for scaling thi detection API to meet demanding throughput requirements, particularly focusing to achieve sub-10ms response times for repeat traffic. The solution is built upon two core principles: **In-Memory Caching** and **Horizontal Scaling (Sharding)**.

---

### 1. Horizontal Sharding with Nginx

#### A. Strategy: Multi-Instance Deployment

Instead of running the API as a single process, the strategy is to launch multiple, independent instances of the `main.py` application.

| Component | Responsibility | Resources |
| :--- | :--- | :--- |
| **Instance 1** | Loads one copy of the model. | CPU Core 1, Port 8001 |
| **Instance 2** | Loads one copy of the model. | CPU Core 2, Port 8002 |
| **Instance *N*** | Loads one copy of the model. | CPU Core *N*, Port 800*N* |

#### Now that the app is running on multiple instances we can use nginx as a load balancer that distributes all the requests between each of the instances

**Tip:** A general best practice is to run **1 Uvicorn worker process per physical CPU core** on the host machine.


#### The idea is that we distribute the traffic equally and use nignx as a load balancer for high sub-10ms SOTA perfomance.

**Nginx Configuration Example:**

```nginx
upstream profanity_api_pool {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
}


server {
    listen 8000;
    server_name api.yourdomain.com;

    location / {
        # Pass the request to one of the backend servers
        proxy_pass http://profanity_api_pool;
        
        # Standard header configuration
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 15s;
    }
}
```