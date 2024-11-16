import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmark.load_test import LoadTestConfig, EnhancedLoadTester


async def main():
    config = LoadTestConfig(
        initial_users=100,
        max_users=1000,
        step_size=50,
        requests_per_user=100,
        step_duration=60,
    )

    tester = EnhancedLoadTester(config=config)
    await tester.setup()
    await tester.run_incremental_load_test()
    tester.generate_report()


if __name__ == "__main__":
    asyncio.run(main())
