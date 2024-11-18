# Load Test Report

Test conducted at: 2024-11-17T12:40:20.653399

## Test Configuration
- initial_users: 10
- max_users: 100
- step_size: 10
- requests_per_user: 20
- ramp_up_time: 5
- step_duration: 30
- request_timeout: 5
- think_time_min: 0.1
- think_time_max: 1.0
- concurrent_users: 10
- test_duration: 300

## Summary Results
```
+----+--------------------+------------------+----------------+--------------+---------------+---------------+---------------+--------------+
|    |   concurrent_users |   total_requests |   success_rate |   error_rate |   avg_latency |   p50_latency |   p95_latency |   throughput |
+====+====================+==================+================+==============+===============+===============+===============+==============+
|  0 |              10.00 |            79.00 |          63.29 |        36.71 |       2362.79 |       2300.12 |       3631.78 |         2.37 |
+----+--------------------+------------------+----------------+--------------+---------------+---------------+---------------+--------------+
```

## Performance Metrics
- Maximum throughput: 2.37 requests/second
- Optimal concurrent users: 10
- Lowest error rate: 36.71%
