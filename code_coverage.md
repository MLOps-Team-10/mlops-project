## Code Coverage

Coverage was measured using `coverage.py`:

```bash
uv run coverage run --source=src -m pytest
uv run coverage report -m

### Code Coverage Summary

| File                                           | Statements | Missed | Coverage | Missing Lines |
|------------------------------------------------|------------|--------|----------|---------------|
| `data.py`                                      | 24         | 0      | 100%     | –             |
| `evaluate.py`                                  | 49         | 49     | 0%       | 1–104         |
| `model.py`                                     | 27         | 6      | 78%      | 81–95, 99     |
| `predict_folder.py`                            | 53         | 53     | 0%       | 1–169         |
| `scripts/download_data.py`                     | 51         | 40     | 22%      | 13–15, 23–33, 53–103, 107 |
| `test.py`                                      | 30         | 30     | 0%       | 3–89          |
| `train.py`                                     | 123        | 36     | 71%      | 28–44, 65–67, 86–108, 258–279, 296 |
| **TOTAL**                                      | **357**    | **214**| **40%**  |               |

_3 empty files skipped_