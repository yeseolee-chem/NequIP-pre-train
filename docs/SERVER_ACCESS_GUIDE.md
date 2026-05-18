# UBAI 서버 접속 및 작업 모니터링 가이드

이 문서는 `halo8-nequip-pretrain` 사전학습 작업을 모니터링·관리하기 위한
실용 가이드입니다. (HPC: `gate1.hpc`, 사용자: `yeseo1ee`)

---

## 1. 서버 접속

```bash
ssh yeseo1ee@gate1.hpc            # 로그인 노드
# 또는 VSCode Remote-SSH 프로파일 사용
```

프로젝트 경로:

```bash
cd /gpfs/home1/yeseo1ee/projects/halo8-nequip-pretrain
```

Conda 환경 (NequIP 0.6.2 / Torch 2.2.1+cu118 / e3nn 0.5.9 사전 설치됨):

```bash
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
```

---

## 2. 클러스터 자원 확인

| 명령 | 용도 |
|---|---|
| `sinfo -o "%P %a %D %T %G %N"` | 파티션별 노드 상태/GRES 요약 |
| `sinfo -N -o "%N %P %T %G" \| grep gpu` | GPU 노드 단위 상태 |
| `scontrol show node n007` | 특정 노드 상세 (`State=IDLE` 이면 가용) |
| `squeue -u $USER` | 내가 제출한 job 목록 |
| `sacct -u $USER --starttime $(date -d '1 day ago' +%F)` | 최근 24h 내 job 이력 |

GPU 파티션 요약:

| Partition | GPU 종류 | 메모 |
|---|---|---|
| `gpu1` | RTX 3090 × 4 / node | 24 GB · n007이 자주 idle |
| `gpu2`, `gpu6` | A10 × 4 / node | 24 GB |
| `gpu3` | A6000 ada × 4 / node | 48 GB |
| `gpu4`, `gpu5` | A6000 × 4 / node | 48 GB |

---

## 3. 현재 사전학습 작업 모니터링

작업 ID는 `submit_pretrain.sh` 안에서 자동 chain 됩니다. 가장 최근 job
번호와 상태를 보려면:

```bash
squeue -u $USER -n halo8_nequip -o "%i %T %N %M %l %R"
```

전체 진행 상태:

```bash
WORK=/gpfs/home1/yeseo1ee/projects/halo8-nequip-pretrain/output/halo8_nequip_v1

# 현재 chain 의 재제출 횟수
cat $WORK/resubmit_count.txt

# 학습 로그(실시간)
tail -f $WORK/logs/training.log

# 최신 SLURM stdout
ls -t $WORK/logs/slurm/ | head -1 | xargs -I{} tail -f $WORK/logs/slurm/{}

# 완료 여부 (이 파일이 있으면 자동 재제출이 중단됨)
ls $WORK/training_complete.flag 2>/dev/null && echo "COMPLETE" || echo "running"

# 체크포인트 상태 (last.ckpt 가 매 epoch 마다 갱신되어야 정상)
ls -la --time-style=full-iso $WORK/checkpoints/
```

체크포인트가 6시간 이상 갱신되지 않으면 학습이 멈춘 신호이므로
`SERVER_ACCESS_GUIDE.md §6` 의 복구 절차를 따르세요.

---

## 4. 데이터 준비 단계 (job 시작 직후 ≈30–60분)

`prepare_data.py` 는 `Halo_1.db` … `Halo_10.db` 를 1회만 스트리밍합니다.
완료되면 `output/halo8_nequip_v1/data/.prepare_done` 가 생기고, 이후
job 에서는 자동으로 skip 합니다.

```bash
cat $WORK/data/.prepare_done           # 완료 시각, 프레임 수
cat $WORK/data/used_reaction_ids.json  # 선택된 반응 + provenance
cat $WORK/data/dataset_hash.txt        # SHA-256
```

---

## 5. 실시간 GPU 상태 확인 (compute node 에 직접 들어가기)

작업이 돌고 있는 노드에서 nvidia-smi 보기:

```bash
NODE=$(squeue -u $USER -n halo8_nequip -h -o "%N")
ssh $NODE -- nvidia-smi
```

또는 SLURM 의 `srun --jobid` 로 동일 컨테이너 안에 명령 실행:

```bash
JID=$(squeue -u $USER -n halo8_nequip -h -o "%i")
srun --jobid=$JID --pty nvidia-smi
```

---

## 6. 비상 정지 / 재시작

| 상황 | 명령 |
|---|---|
| 체인 정지 (현재 job 은 그대로) | `touch $WORK/training_complete.flag` |
| 현재 job 도 즉시 종료 | `scancel -u $USER -n halo8_nequip` |
| 카운터 리셋 후 재시작 | `echo 0 > $WORK/resubmit_count.txt && sbatch scripts/submit_pretrain.sh` |
| Config 만 바꿔서 이어 학습 | `configs/halo8_nequip_v1.yaml` 수정 → `sbatch scripts/submit_pretrain.sh` (last.ckpt 로 resume) |

---

## 7. 자주 발생하는 실패와 대응

| 증상 | 원인 | 대응 |
|---|---|---|
| `CUDA out of memory` | batch_size 가 GPU 메모리 초과 | `configs/halo8_nequip_v1.yaml` 의 `batch_size` 를 16 으로 낮추고 resume |
| `NaN loss` | LR 과대 / 데이터 이상 | `learning_rate` 를 1e-3 으로 낮추고 `best.ckpt` 에서 resume |
| `epoch_NNN.ckpt` 만 있고 `last.ckpt` 없음 | SIGKILL 시 쓰기 중단 | 가장 최근 `epoch_*.ckpt` 를 `last.ckpt` 로 복사 후 resume |
| `resubmit_count > 30` | 실패 루프 가능성 | chain 정지 후 슬럼 stderr 확인 |
| n007 이 점유됨 | 다른 사용자가 할당 | `submit_pretrain.sh` 의 `--nodelist=n007` 줄을 주석 처리하면 `gpu1` 의 다른 노드도 허용 |

---

## 8. 결과물 사용 (학습 종료 후)

```bash
$WORK/checkpoints/best.ckpt    # 최저 val loss
$WORK/checkpoints/last.ckpt    # 마지막 epoch
$WORK/data/halo8_train.extxyz  # 학습 데이터(재현용)
$WORK/data/dataset_hash.txt    # 데이터 SHA-256
$WORK/data/used_reaction_ids.json  # 사용된 17,574 reaction ID
```

Active learning fine-tuning (Stage 6) 에서는 `last.ckpt` 를 백본으로 로드합니다.
