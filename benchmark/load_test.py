import asyncio
import time
import numpy as np
import aiohttp
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random
import asyncpg
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from recommender.config import settings
import os


@dataclass
class LoadTestConfig:
    initial_users: int = 10
    max_users: int = 100
    step_size: int = 10
    requests_per_user: int = 50
    ramp_up_time: int = 5
    step_duration: int = 30
    request_timeout: int = 5
    think_time_min: float = 0.1
    think_time_max: float = 1.0


class EnhancedLoadTester:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        config: Optional[LoadTestConfig] = None,
    ):
        self.base_url = base_url
        self.config = config or LoadTestConfig()
        self.results: List[Dict] = []
        self.user_ids: List[str] = []
        self.error_details: Dict[int, List[str]] = {}
        self.step_results: List[Dict] = []
        self.start_time = None

    async def setup(self):
        # Get user IDs from database
        conn = await asyncpg.connect(settings.POSTGRES_URL)
        try:
            rows = await conn.fetch("SELECT user_id FROM user_features")
            if not rows:
                raise Exception(
                    "No users found in database. Please seed the database first."
                )
            self.user_ids = [str(row["user_id"]) for row in rows]

            # Verify service health
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status != 200:
                        health_data = await response.json()
                        raise Exception(f"Service unhealthy: {health_data}")
        finally:
            await conn.close()

    async def _run_user_session(self, session: aiohttp.ClientSession, user_id: str):
        for _ in range(self.config.requests_per_user):
            if datetime.now() - self.start_time > timedelta(
                seconds=self.config.step_duration
            ):
                break

            start_time = time.time()
            try:
                async with session.get(
                    f"{self.base_url}/recommendations/{user_id}",
                    timeout=self.config.request_timeout,
                ) as response:
                    response_data = await response.json()
                    latency = (time.time() - start_time) * 1000
                    status = response.status

                    if status != 200:
                        if status not in self.error_details:
                            self.error_details[status] = []
                        self.error_details[status].append(str(response_data))

            except asyncio.TimeoutError:
                latency = self.config.request_timeout * 1000
                status = 408
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                status = 500
                print(f"Request error for user {user_id}: {str(e)}")

            self.results.append(
                {
                    "timestamp": datetime.now(),
                    "latency": latency,
                    "status": status,
                    "user_id": user_id,
                    "elapsed_time": (datetime.now() - self.start_time).total_seconds(),
                }
            )

            # Add think time between requests
            think_time = random.uniform(
                self.config.think_time_min, self.config.think_time_max
            )
            await asyncio.sleep(think_time)

    async def run_load_test(self):
        if not self.user_ids:
            raise Exception("No user IDs available. Did you run setup()?")

        self.start_time = datetime.now()
        self.results = []

        connector = aiohttp.TCPConnector(limit=self.config.concurrent_users)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            # Create all user sessions at once
            for i in range(self.config.concurrent_users):
                user_id = random.choice(self.user_ids)
                # Calculate delay but don't await it here
                delay = (i / self.config.concurrent_users) * self.config.ramp_up_time
                # Create task with delay built into it
                task = asyncio.create_task(
                    self._delayed_user_session(session, user_id, delay)
                )
                tasks.append(task)

            # Wait for all tasks to complete or until test duration is reached
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks), timeout=self.config.test_duration
                )
            except asyncio.TimeoutError:
                # Test duration reached, this is expected
                pass

    async def _delayed_user_session(self, session, user_id, delay):
        if delay > 0:
            await asyncio.sleep(delay)
        await self._run_user_session(session, user_id)

    async def run_incremental_load_test(self):
        print("\nStarting Incremental Load Test")
        print("=" * 50)

        for num_users in range(
            self.config.initial_users, self.config.max_users + 1, self.config.step_size
        ):
            print(f"\nTesting with {num_users} concurrent users...")

            # Run test for this step
            step_metrics = await self._run_test_step(num_users)
            self.step_results.append({"concurrent_users": num_users, **step_metrics})

            # Print interim results
            self._print_step_results(step_metrics, num_users)

            # Check for degradation
            if step_metrics["error_rate"] > 10 or step_metrics["p95_latency"] > 2000:
                print("\nPerformance degradation detected. Stopping test.")
                break

            # Cool down period
            await asyncio.sleep(5)

    async def _run_test_step(self, num_users: int) -> Dict:
        self.start_time = datetime.now()  # Set start time for this test step
        self.results = []  # Reset results for this step

        connector = aiohttp.TCPConnector(limit=num_users)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(num_users):
                user_id = random.choice(self.user_ids)
                tasks.append(self._run_user_session(session, user_id))

            await asyncio.gather(*tasks)

        return self._calculate_step_metrics()

    def _calculate_step_metrics(self) -> Dict:
        df = pd.DataFrame(self.results)
        successful = df[df["status"] == 200]

        if len(df) == 0:
            return {"error": "No results collected"}

        return {
            "total_requests": len(df),
            "success_rate": len(successful) / len(df) * 100,
            "error_rate": (len(df) - len(successful)) / len(df) * 100,
            "avg_latency": successful["latency"].mean(),
            "p50_latency": successful["latency"].median(),
            "p95_latency": successful["latency"].quantile(0.95),
            "throughput": len(df) / (df["elapsed_time"].max() or 1),
        }

    def generate_report(self, output_dir: str = "benchmark_results"):
        os.makedirs(output_dir, exist_ok=True)

        # Convert results to DataFrame
        df = pd.DataFrame(self.step_results)

        # Generate summary table
        summary_table = tabulate(df, headers="keys", tablefmt="grid", floatfmt=".2f")

        # Create plots
        self._create_plots(df, output_dir)

        # Generate markdown report
        self._generate_markdown_report(df, summary_table, output_dir)

    def _create_plots(self, df: pd.DataFrame, output_dir: str):
        # Latency vs Users
        plt.figure(figsize=(10, 6))
        plt.plot(df["concurrent_users"], df["avg_latency"], marker="o", label="Average")
        plt.plot(df["concurrent_users"], df["p95_latency"], marker="o", label="P95")
        plt.xlabel("Concurrent Users")
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.title("Latency vs Concurrent Users")
        plt.savefig(f"{output_dir}/latency_vs_users.png")
        plt.close()

        # Throughput vs Users
        plt.figure(figsize=(10, 6))
        plt.plot(df["concurrent_users"], df["throughput"], marker="o")
        plt.xlabel("Concurrent Users")
        plt.ylabel("Throughput (requests/sec)")
        plt.title("Throughput vs Concurrent Users")
        plt.savefig(f"{output_dir}/throughput_vs_users.png")
        plt.close()

    def _generate_markdown_report(
        self, df: pd.DataFrame, summary_table: str, output_dir: str
    ):
        with open(f"{output_dir}/load_test_report.md", "w") as f:
            f.write("# Load Test Report\n\n")
            f.write(f"Test conducted at: {datetime.now().isoformat()}\n\n")

            f.write("## Test Configuration\n")
            for key, value in self.config.__dict__.items():
                f.write(f"- {key}: {value}\n")

            f.write("\n## Summary Results\n")
            f.write("```\n")
            f.write(summary_table)
            f.write("\n```\n\n")

            f.write("## Performance Metrics\n")
            f.write(
                f"- Maximum throughput: {df['throughput'].max():.2f} requests/second\n"
            )
            f.write(
                f"- Optimal concurrent users: {df.loc[df['throughput'].idxmax(), 'concurrent_users']}\n"
            )
            f.write(f"- Lowest error rate: {df['error_rate'].min():.2f}%\n")

            if len(self.error_details) > 0:
                f.write("\n## Error Details\n")
                for status, errors in self.error_details.items():
                    f.write(f"\n### Status {status}\n")
                    for error in errors[:5]:  # Show first 5 errors of each type
                        f.write(f"- {error}\n")

    def _print_step_results(self, metrics: Dict, num_users: int):
        print(f"\nResults for {num_users} concurrent users:")
        print("-" * 40)
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
        print(f"Error Rate: {metrics['error_rate']:.2f}%")
        print(f"Average Latency: {metrics['avg_latency']:.2f}ms")
        print(f"P95 Latency: {metrics['p95_latency']:.2f}ms")
        print(f"Throughput: {metrics['throughput']:.2f} requests/sec")

        if self.error_details:
            print("\nError Summary:")
            for status, errors in self.error_details.items():
                print(f"Status {status}: {len(errors)} occurrences")
