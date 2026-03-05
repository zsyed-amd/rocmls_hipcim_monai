import subprocess
import sys
import argparse

def start_vllm_server(
    model="llava-hf/llava-1.5-7b-hf",
    host="0.0.0.0", 
    port=8000,
    gpu_memory_utilization=0.8,
    trust_remote_code=True
):
    """Start vLLM server with vision model for pathology analysis"""
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"Server will be available at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start vLLM server for pathology analysis")
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="Vision model to load")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization")
    
    args = parser.parse_args()
    
    start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory
    )