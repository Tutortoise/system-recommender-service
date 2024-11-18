import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmark.load_test import LoadTestConfig, EnhancedLoadTester


async def main():
    # Configure the load test
    config = LoadTestConfig(
        initial_users=10,  # Start with fewer users for testing
        max_users=100,
        step_size=10,
        requests_per_user=20,
        step_duration=30,
        concurrent_users=10,
        test_duration=300,  # 5 minutes total test duration
        request_timeout=5,
        think_time_min=0.1,
        think_time_max=1.0
    )

    print("Starting load test...")
    print(f"Base configuration: {config}")

    try:
        tester = EnhancedLoadTester(config=config)
        await tester.setup()
        print("Setup completed successfully")

        print("\nRunning incremental load test...")
        await tester.run_incremental_load_test()

        print("\nGenerating report...")
        tester.generate_report()

        print("\nLoad test completed successfully")

    except Exception as e:
        print(f"Error during load test: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
