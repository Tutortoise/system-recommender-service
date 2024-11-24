![latency_vs_users.png](benchmark_results/latency_vs_users.png)


everything pre-configured using [uv](https://docs.astral.sh/uv/), just run `uv sync` and `uv granian recommender.main:app --host 0.0.0.0 --port 8000 --interface asgi --workers {number_of_workers}` to start the service.