import asyncio
from benchmark.load_test import LoadTestConfig, EnhancedLoadTester


async def main():
    config = LoadTestConfig(
        initial_users=500,
        max_users=2000,
        step_size=500,
        requests_per_user=200,
        step_duration=120,
    )

    tester = EnhancedLoadTester(config=config)
    await tester.setup()
    await tester.run_incremental_load_test()
    tester.generate_report()


if __name__ == "__main__":
    asyncio.run(main())
