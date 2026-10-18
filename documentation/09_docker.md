# Docker Compose와 재현 환경

## 무엇을 구현했는가

`Dockerfile`은 CPU PyTorch 이미지와 선택적 CUDA PyTorch 이미지를 제공합니다. `compose.yaml`의 모든 연구 서비스는 저장소 루트를 `/workspace`에 마운트합니다. Python은 마운트한 `src/`를 바로 읽으므로 코드 수정마다 이미지를 다시 빌드할 필요가 없습니다. 의존성이 바뀌었을 때 다시 빌드합니다.

| 서비스 | 실행 장치 | 용도 |
|---|---|---|
| research | CPU | MuJoCo, 학습, 평가, 테스트 |
| warp | GPU 노출 + CPU PyTorch | Warp CUDA 시뮬레이션, CPU 학습 |
| cuda | GPU 노출 + CUDA PyTorch | Warp CUDA와 GPU 신경망 학습 |
| docs | CPU / localhost:8000 | HTML 문서 열람 |

`warp`는 `gpu` profile, `cuda`는 `cuda` profile에서 실행합니다. NVIDIA GPU와 Docker GPU 지원이 필요합니다. CUDA가 없다고 요청한 Warp 계산을 MuJoCo로 조용히 대체하지 않습니다.

이번 변경에는 `wandb`와 `tqdm` 의존성이 추가되었습니다. 이전 이미지를 사용 중이라면 연구·CUDA 이미지를 다시 빌드해야 합니다. 루트 mount는 Python 소스 수정은 즉시 반영하지만 이미지에 없는 패키지를 설치하지는 않습니다.

```bash
docker compose --profile cuda build research cuda
docker compose --profile cuda run --rm cuda python -m cube doctor
```

GPU 신경망 학습을 시작하기 전에 `doctor`의 `torch_cuda: true`와 GPU 이름을 확인합니다. 데이터 생성용 MuJoCo CPU worker와 신경망 CUDA 학습 장치는 독립적입니다.

## 왜 구현했는가

시뮬레이터·텐서 라이브러리 버전을 고정하고, 호스트 Python 환경을 변경하지 않으면서 동일한 명령을 재실행하기 위해서입니다. Warp 최초 실행에는 kernel compilation 시간이 포함됩니다. 성능 비교는 최초 컴파일과 정상 실행을 분리해야 합니다.

상위 패키지는 `pyproject.toml`에 고정합니다. 실행 시 `doctor`와 각 결과 metadata로 실제 버전을 기록합니다. 하위 의존성 및 Docker base digest 전체를 고정한 bitwise 재현 이미지까지 제공하는 것은 아닙니다.

## 사용 명령

```bash
docker compose build research
docker compose run --rm research python -m cube doctor
docker compose run --rm research python -m pytest -q -m "not warp"
docker compose run --rm research python -m cube docs
docker compose up -d docs
docker compose logs docs
docker compose down
```

브라우저에서 [문서 열기](http://localhost:8000). 포트를 바꾸려면 `.env`에 `DOCS_PORT=8080`을 설정합니다. 문서 서비스는 호스트의 loopback 주소에만 바인딩합니다. `docker compose down`은 실험 데이터나 cache volume을 삭제하지 않습니다.

## W&B 온라인 기록과 로컬 설정

저장소 루트의 `.env.example`을 참고하여 로컬 `.env`를 준비합니다. Compose가 아래 값을 컨테이너로 전달합니다.

| 설정 | 용도 |
|---|---|
| `WANDB_API_KEY` | W&B 계정 인증 |
| `WANDB_ENTITY` | 사용자 계정 또는 팀 |
| `WANDB_PROJECT` | config/CLI에서 project를 지정하지 않을 때의 기본 project |
| `DOCS_PORT` | 호스트에서 HTML 문서를 여는 포트, 기본 8000 |

API 키는 로컬 `.env`에만 넣고 JSON config·명령행·문서에 기록하지 않습니다. `.env`는 Git에서 제외합니다. Compose의 `WANDB_DIR`은 `/workspace/outputs/wandb`이며, 학습 tracker는 각 run의 output 디렉터리도 SDK에 지정하므로 실행별 로컬 기록이 호스트에 남습니다.

`train_gpu.json`은 `wandb.mode: online`을 사용합니다. online 초기화에는 인증과 네트워크가 필요하며 실패하면 오류를 내고 실행을 중단합니다. 정상 실행의 `wandb_run.json`에는 run ID와 온라인 URL을 기록합니다. 학습·검증 loss, learning rate, 처리량, GPU 메모리와 최종 평가 지표로 구조·seed별 결과를 비교할 수 있습니다.

```bash
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-nerd --config configs/train_gpu.json --model hierarchical --seed 0 --output outputs/cube-gpu/hierarchical
```

계정 인증 없이 로컬 기록 경로를 확인할 때는 별도 output과 `--wandb-mode offline`을 사용합니다. offline 실행은 온라인 업로드가 완료된 실행이 아닙니다. API 키가 준비되지 않은 상태에서 online 성공을 표시하지 않습니다.

```bash
docker compose --profile cuda run --rm cuda python -m cube train --dataset data/cube-nerd --config configs/train_gpu.json --model hierarchical --seed 0 --wandb-mode offline --output outputs/cube-gpu-offline/hierarchical
```

## 긴 실행의 tqdm 표시

tqdm은 데이터 생성·정규화·학습·검증·평가·suite의 진행률과 처리 속도를 표시합니다. 학습에는 loss와 learning rate도 표시합니다. 일반 대화형 터미널은 Compose `run`으로 실행합니다. 진행률이 없는 로그가 필요하면 `TQDM_DISABLE=1`, TTY가 없는 환경에서도 진행률을 남기려면 `TQDM_FORCE=1`을 컨테이너 환경으로 전달합니다. `TQDM_DISABLE`이 우선합니다.

```bash
docker compose run --rm -e TQDM_DISABLE=1 research python -m cube generate --config configs/cube_nerd_scale.json --output data/cube-nerd --resume

docker compose --profile cuda run --rm -e TQDM_FORCE=1 cuda python -m cube suite --config configs/suite_gpu.json --output outputs/cube-gpu-suite --resume
```

진행률과 ETA는 현재 처리량의 추정치입니다. 전체 데이터·모델 비교의 실행 순서와 재개 절차는 [GPU 실험 안내](12_gpu_training.html)에 정리했습니다.

```bash
docker compose --profile gpu run --rm warp python -m cube doctor
docker compose --profile gpu run --rm warp python -m cube validate --backend warp --device cuda:0 --output outputs/warp_validation
docker compose --profile cuda build cuda
```

## 사용한 공식 API

- [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [MuJoCo Warp 소스](https://github.com/google-deepmind/mujoco_warp)

MuJoCo Warp는 Warp로 실제 physics step을 수행합니다. 일반 MuJoCo는 독립적인 CPU 평가 경로입니다. 양자의 수치 결과를 bitwise 동일하다고 가정하지 않습니다.
