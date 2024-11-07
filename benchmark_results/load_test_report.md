# Load Test Report

Test conducted at: 2024-11-07T19:47:05.698676

## Test Configuration
- initial_users: 500
- max_users: 2000
- step_size: 500
- requests_per_user: 200
- ramp_up_time: 5
- step_duration: 120
- request_timeout: 5
- think_time_min: 0.1
- think_time_max: 1.0

## Summary Results
```
+----+--------------------+------------------+----------------+--------------+---------------+---------------+---------------+--------------+
|    |   concurrent_users |   total_requests |   success_rate |   error_rate |   avg_latency |   p50_latency |   p95_latency |   throughput |
+====+====================+==================+================+==============+===============+===============+===============+==============+
|  0 |             500.00 |         33960.00 |          95.54 |         4.46 |       1030.95 |        232.13 |       3985.58 |       279.81 |
+----+--------------------+------------------+----------------+--------------+---------------+---------------+---------------+--------------+
```

## Performance Metrics
- Maximum throughput: 279.81 requests/second
- Optimal concurrent users: 500
- Lowest error rate: 4.46%
