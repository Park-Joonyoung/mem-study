# LLM KV-Cache 메모리 스터디

LLM의 **KV-cache decode (GQA, group size=4) DRAM 트래픽**을 시뮬레이션으로 분석한다.
DRAM 채널 1개에서 발생하는 contention을 측정한다.

- Ramulator 2.1(DRAM 시뮬)에 합성 KV 트레이스를 넣어 sweep
- 실측 llama.cpp 캡처로 검증
  - HW 절대성능을 맞추는 게 아니라, llama.cpp의 KV 접근 패턴만 trace로 뽑아 같은 시뮬레이터에서 합성과 비교(apples-to-apples)
- **동시 생성** : 사용자 N명 동시 서빙 = 독립 생성 N개(N-core). 이들이 **DRAM 채널 1개를 나눠 씀**
- **포화 시작 지점 = power(throughput/latency) 최대 N** : N이 커지면 채널이 포화. 포화가 시작되는 N = 채널당 수용량.
- reread vs. naive
  - 초기 (naive) : 단순 다중 for loop으로 kv decode 구현. 포화 시작 N=8, demand 과소평가
  - 실제로는 같은 group의 Q head 4개가 K/V를 각각 읽어서 병렬적으로 4번 read가 발생하고 있었음 -> 개선 (reread)


## 결과

| 측정 | 값 | 뜻 |
|---|---|---|
| 생성 1개의 DRAM 수요 | ~3.16 GB/s | 채널 한계(19.2 GB/s)보다 훨씬 낮음 → 생성 1개는 DRAM 병목 아님 |
| **포화 시작 지점** | **N=5** | DDR4 채널 1개는 동시 생성 ~5개에서 포화 |
| batch (한 생성 내부 묶음) | 영향 ~0 | 동시성 축이 아님 — 코어 수만 contention 결정 |
| 합성 vs 실측 오차 | 1–7% | 합성 생성기가 실제 llama.cpp와 일치 |

상세: `report/TECH_REPORT.md` · 그림 `report/figures/` (11개).

## 디렉토리

| 경로 | 내용 |
|---|---|
| `gen/` | 트레이스 생성기 (합성 + 실측 KV 로그 전개) |
| `run/` | sweep 실행 + `ram21.py` (Ramulator 2.1 러너) |
| `analysis/` | stats 파싱 · 그림 생성 |
| `configs/` | Ramulator 설정 (`base_ddr4_kv.py`) |
| `results/` | sweep CSV · `.stats` · `hw_kv_demand/` |
| `report/` | 분석 문서 + 그림 |
| `sim_constants.py` | 공유 상수 (결합관계는 주석에) |
| `ramulator2/` | submodule — DRAM 시뮬 (`v2.1-kvstudy`: upstream v2.1 + simpleO3 int→size_t 패치) |
| `llama.cpp/` | submodule — 실측 계측 패치 (`kv-demand-perf` 브랜치) |
| `traces/` `models/` | gitignore (용량 큼 / 재생성 가능) |

## 셋업

```bash
# 1) 소스 + submodule 한 번에
git clone --recurse-submodules <this-repo>      # 또는 clone 후:
git submodule update --init --recursive

# 2) Ramulator 빌드 (nanobind .so 생성 — submodule은 소스만)
cmake -S ramulator2 -B ramulator2/build -DCMAKE_BUILD_TYPE=Release -DRAMULATOR_PYTHON_BINDINGS=ON
cmake --build ramulator2/build -j

# 3) Python 3.10 (nanobind 모듈이 3.10 빌드). 외부 빌드를 쓰려면 RAMULATOR_PYTHON 환경변수로 override.
```

- ram21.py는 기본적으로 `ramulator2/python`(빌드된 submodule)을 import. `RAMULATOR_PYTHON`으로 외부 경로 지정 가능.
- 실측 트레이스 캡처가 필요하면 llama.cpp도 빌드 (`gen/expand_kv_log.py` 참고).

## 실행

```bash
# 1) 합성 트레이스 + baseline
python3 gen/gen_kv_decode_trace.py
python3 configs/base_ddr4_kv.py > results/kv_ddr4_baseline.stats

# 2) core × batch sweep  (--cores = 동시 생성 수)
python3 run/run_kv_core_batch_sweep.py --batch 1 --cores 1 2 4 8

# 3) 그림
python3 analysis/make_report_figures.py
```

## 더 보기

| 주제 | 위치 |
|---|---|
| 전체 방법론 · 결과 · 한계 | `report/TECH_REPORT.md` |
| 하드웨어 수요 측정 (perf_event) | `results/hw_kv_demand/README.md` |
| 실측 llama.cpp 캡처 파이프라인 | `gen/expand_kv_log.py` · `run/run_real_trace_sweep.py` |
| 클럭 도메인 · LLC associativity 규칙 | `sim_constants.py` 주석 · `run/ram21.py` |
| GQA emission (`reread`/`naive`) | `gen/gen_kv_decode_trace.py` docstring |
