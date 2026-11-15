# Dataset Cohort Audit

Based on `source_code/config_v2.json`, `source_code/config.json`, and the saved `dataset_split_audit.csv`.

| Cohort category | Count |
|---|---:|
| Selected datasets | 23 |
| Original V1 datasets retained | 10 |
| Additional screened datasets | 13 |
| Event-evaluable test datasets | 17 |
| No-positive-test datasets | 6 |
| Complete test events | 24 |

The screening script selected 13 additions using test-event coverage and clean train/validation-window criteria. The original 10 V1 datasets were carried forward without reapplying the new screen. Four of those ten have complete positive test events in the held-out test partition; six have no positive test labels. The six remain useful as negative-only false-alarm cases but must not be included in event-recall or event-based hypothesis denominators.

## No-positive-test datasets

- `realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv`
- `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv`
- `realTraffic/TravelTime_387.csv`
- `realTraffic/TravelTime_451.csv`
- `realTweets/Twitter_volume_AAPL.csv`
- `realTweets/Twitter_volume_AMZN.csv`

## Event-evaluable datasets

- `realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv` — 2 complete test event(s)
- `realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv` — 1 complete test event(s)
- `realAWSCloudwatch/elb_request_count_8c0756.csv` — 1 complete test event(s)
- `realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv` — 1 complete test event(s)
- `realAdExchange/exchange-4_cpm_results.csv` — 1 complete test event(s)
- `realKnownCause/ambient_temperature_system_failure.csv` — 1 complete test event(s)
- `realKnownCause/cpu_utilization_asg_misconfiguration.csv` — 1 complete test event(s)
- `realKnownCause/ec2_request_latency_system_failure.csv` — 2 complete test event(s)
- `realKnownCause/machine_temperature_system_failure.csv` — 1 complete test event(s)
- `realKnownCause/nyc_taxi.csv` — 3 complete test event(s)
- `realTraffic/occupancy_t4013.csv` — 2 complete test event(s)
- `realTraffic/speed_6005.csv` — 1 complete test event(s)
- `realTraffic/speed_7578.csv` — 2 complete test event(s)
- `realTraffic/speed_t4013.csv` — 2 complete test event(s)
- `realTweets/Twitter_volume_CVS.csv` — 1 complete test event(s)
- `realTweets/Twitter_volume_IBM.csv` — 1 complete test event(s)
- `realTweets/Twitter_volume_KO.csv` — 1 complete test event(s)
