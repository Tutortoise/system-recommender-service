import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmark.load_test import LoadTestConfig, EnhancedLoadTester
from datetime import datetime


async def main():
    config = LoadTestConfig(
        initial_users=5,
        max_users=500,
        step_size=5,
        requests_per_user=20,
        step_duration=30,
        request_timeout=10,
        think_time_min=0.5,
        think_time_max=2.0,
        enable_degradation_check=True,
        max_error_rate=15.0,
        max_p95_latency=3000.0
    )

    print(f"Starting load test at {datetime.now()}")
    print("=" * 50)
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print("=" * 50)

    try:
        tester = EnhancedLoadTester(
            base_url="http://localhost:8000",  # Adjust if needed
            config=config
        )

        print("Setting up test...")
        await tester.setup()

        print("Running load test...")
        await tester.run_incremental_load_test()

        print("Generating report...")
        tester.generate_report(output_dir="benchmark_results")

        print("Load test completed successfully!")

    except Exception as e:
        print(f"Error during load test: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
