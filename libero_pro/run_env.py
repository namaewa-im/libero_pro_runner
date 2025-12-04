import json, time, argparse
import os

import numpy as np
import requests
import yaml
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

# OpenVLA 서버와 동일하게 numpy를 JSON으로 직렬화/역직렬화하기 위해 json_numpy 사용
import json_numpy

json_numpy.patch()

# 프로젝트 내부 유틸 import를 위해 레포 루트를 sys.path에 추가
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import perturbation  # LIBERO-PRO perturbation 생성 유틸

# perturbation 조합은 기본 4개 suite에 대해서만 지원
ALLOWED_BASE_SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]


def _json_default(obj):
    """numpy 타입을 JSON으로 직렬화하기 위한 기본 변환기."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def json_dumps(obj):
    """numpy 배열을 포함한 객체를 JSON 문자열로 인코딩."""
    return json.dumps(obj, default=_json_default)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """쿼터니언(w, x, y, z)을 axis-angle 3벡터로 변환."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must have 4 elements, got shape {q.shape}")
    w = np.clip(q[0], -1.0, 1.0)
    xyz = q[1:]
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = xyz / s
    return (axis * angle).astype(np.float32)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    그리퍼 스칼라 입력을 [0, 1] → [-1, +1] 범위로 변환하고 필요하면 이진화.
    action 배열의 마지막 요소가 그리퍼 값이라고 가정.
    """
    act = np.asarray(action, dtype=np.float32).copy()
    if act.shape[-1] < 1:
        raise ValueError("action vector must have at least 1 dimension")
    gripper = act[-1]
    gripper = np.clip(gripper, 0.0, 1.0) * 2.0 - 1.0
    if binarize:
        gripper = 1.0 if gripper >= 0.0 else -1.0
    act[-1] = gripper
    return act


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """그리퍼 부호를 뒤집는다."""
    act = np.asarray(action, dtype=np.float32).copy()
    if act.shape[-1] < 1:
        raise ValueError("action vector must have at least 1 dimension")
    act[-1] = -act[-1]
    return act


def interactive_choose_benchmark_and_task():
    """터미널에서 벤치마크/태스크를 골라서 선택하는 간단한 메뉴."""
    benchmark_dict = benchmark.get_benchmark_dict()

    # perturbation 조합은 기본 4개 suite에 대해서만 허용
    names = [name for name in ALLOWED_BASE_SUITES if name in benchmark_dict]
    if not names:
        raise RuntimeError(
            "기본 LIBERO suite(libero_10/libero_spatial/libero_object/libero_goal)을 찾지 못했습니다."
        )

    print("\n=== LIBERO 벤치마크 선택 ===")
    for i, name in enumerate(names, start=1):
        print(f"[{i}] {name}")
    print("example) 2 or libero_object")

    while True:
        raw = input(
            f"벤치마크 번호/이름 입력 (기본: {names[0]}): "
        ).strip()
        if raw == "":
            bench_name = names[0]
            break
        # 숫자 인덱스
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(names):
                bench_name = names[idx - 1]
                break
        # 이름 직접 입력
        key = raw.lower()
        if key in names:
            bench_name = key
            break

        print("input is invalid. please try again.")

    benchmark_instance = benchmark_dict[bench_name]()
    task_names = benchmark_instance.get_task_names()

    print(f"\n=== select task: {bench_name} ({len(task_names)} tasks) ===")
    for i, tname in enumerate(task_names, start=1):
        print(f"[{i}] {tname}")

    while True:
        raw = input(f"input task index (1 ~ {len(task_names)}, default: 1): ").strip()
        if raw == "":
            task_index = 0
            break
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(task_names):
                task_index = idx - 1
                break
        print("input is invalid. please try again.")

    return bench_name, benchmark_instance, task_index


def interactive_choose_perturbation():
    """
    어떤 perturbation들을 적용할지 인터랙티브하게 선택.
    - 0: 원본 LIBERO (perturbation 없음)
    - 1: Environment
    - 2: Swap (position)
    - 3: Object
    - 4: Language
    - 5: Task (단독 사용만 가능, 다른 옵션과 함께 사용할 수 없음)
    """
    print("\n=== Perturbation 선택 ===")
    print("[0] None (원본 LIBERO 그대로 사용)")
    print("[1] Environment (환경 교체)")
    print("[2] Swap / Position (객체 위치 교환)")
    print("[3] Object (객체 외형/종류 변경)")
    print("[4] Language (언어 명령 파라프레이즈)")
    print("[5] Task (태스크 정의 자체 변경, 단독 사용만 가능)")
    print("예시) 1,3  /  2,3,4  / 5  / 엔터=0")

    while True:
        raw = input("적용할 perturbation 번호들을 쉼표로 입력: ").strip()
        if raw == "":
            # 아무 것도 선택하지 않음 → 원본 환경
            return {
                "use_environment": False,
                "use_swap": False,
                "use_object": False,
                "use_language": False,
                "use_task": False,
            }

        tokens = [t.strip() for t in raw.split(",") if t.strip() != ""]
        if any(t not in {"0", "1", "2", "3", "4", "5"} for t in tokens):
            print("유효하지 않은 번호가 있습니다. 0~5 중에서 다시 입력해 주세요.")
            continue

        # 0이 포함되어 있으면서 다른 것도 있으면 애매하므로 다시 입력
        if "0" in tokens and len(tokens) > 1:
            print("0(None)은 단독으로만 사용할 수 있습니다. 다시 입력해 주세요.")
            continue

        use_env = "1" in tokens
        use_swap = "2" in tokens
        use_object = "3" in tokens
        use_language = "4" in tokens
        use_task = "5" in tokens

        # Task perturbation은 단독으로만 허용 (코드 구현 상 제약)
        if use_task and (use_env or use_swap or use_object or use_language):
            print("Task perturbation(use_task)은 다른 perturbation과 함께 사용할 수 없습니다.")
            print("5만 단독으로 넣거나, 1/2/3/4만 조합해서 다시 선택해 주세요.")
            continue

        return {
            "use_environment": use_env,
            "use_swap": use_swap,
            "use_object": use_object,
            "use_language": use_language,
            "use_task": use_task,
        }


def main():
    # numpy를 포함한 payload 직렬화를 위해 커스텀 덤프 사용

    # 실행 중에 벤치마크 / 태스크를 고르는 인터랙티브 메뉴
    bench_name, benchmark_instance, task_index = interactive_choose_benchmark_and_task()
    task = benchmark_instance.get_task(task_index)

    # 어떤 perturbation을 적용할지 선택
    perturb_flags = interactive_choose_perturbation()
    use_any_perturb = any(perturb_flags.values())

    # LIBERO 경로들
    bddl_root = get_libero_path("bddl_files")
    init_root = get_libero_path("init_states")

    # 기본(원본) BDDL / init 경로
    suite_name = task.problem_folder  # 예: libero_object, libero_spatial, ...
    bddl_dir = os.path.join(bddl_root, suite_name)
    init_dir = os.path.join(init_root, suite_name)

    if use_any_perturb:
        # 프로젝트 루트 기준으로 evaluation_config.yaml 불러오기
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_cfg_path = os.path.join(project_root, "evaluation_config.yaml")
        with open(eval_cfg_path, "r", encoding="utf-8") as f:
            eval_cfg = yaml.safe_load(f)

        # 이 스크립트에서는 하나의 벤치마크(suite)에 대해서만 perturbation을 수행하므로
        # bddl_files_path를 해당 suite 하위 디렉토리로 맞춰준다.
        eval_cfg["bddl_files_path"] = os.path.join(bddl_root, suite_name)

        # init 파일이 생성될 기본 디렉토리 (create_env 내부에서 <suite_name>_temp가 붙음)
        eval_cfg["init_file_dir"] = init_root + "/"

        # init state 생성 스크립트 경로 지정
        eval_cfg["script_path"] = os.path.join(
            project_root, "notebooks", "generate_init_states.py"
        )

        # perturbation 시 참조할 task suite 이름 (YAML 키와 로그 등에 사용)
        eval_cfg["task_suite_name"] = suite_name

        # 플래그 반영
        for key in ["use_environment", "use_swap", "use_object", "use_language", "use_task"]:
            eval_cfg[key] = perturb_flags[key]

        print("\n[run_env] 선택한 perturbation 설정으로 BDDL / init 파일을 생성합니다...")
        perturbation.create_env(configs=eval_cfg)

        # create_env는 항상 <원래_폴더명>_temp 아래에 결과를 저장하도록 구현되어 있음
        perturbed_suite_name = suite_name + "_temp"
        bddl_dir = os.path.join(bddl_root, perturbed_suite_name)
        init_dir = os.path.join(init_root, perturbed_suite_name)
        print(f"[run_env] perturbation 적용된 BDDL 디렉토리: {bddl_dir}")
        print(f"[run_env] perturbation 적용된 init 디렉토리:  {init_dir}")

    # 최종적으로 사용할 BDDL 파일 경로
    bddl_file_path = os.path.join(bddl_dir, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_file_path,
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "render_camera": "frontview",
    }

    env = ControlEnv(**env_args)
    env.seed(0)

    # init state 설정 (perturbation을 쓴 경우에는 새로 생성된 init 파일에서 읽어옴)
    init_states = None
    if use_any_perturb:
        init_path = os.path.join(init_dir, task.init_states_file)
        try:
            init_states = torch.load(init_path)
            print(f"[run_env] perturbation init state 로드: {init_path}")
        except Exception as e:
            print(f"[run_env] perturbation init state 로드 실패({init_path}): {e}")
    else:
        init_states = benchmark_instance.get_task_init_states(task_index)

    if init_states is not None and len(init_states) > 0:
        env.set_init_state(init_states[0])

    # OpenVLA 서버 설정
    openvla_url = "http://localhost:8777/act"
    instruction = "put the red block on the blue plate"

    # 첫 관측
    obs = env.reset()
    sim_step = 0
    action_buffer = []  # 서버에서 받은 action 시퀀스를 저장하는 로컬 버퍼
    payload_dumped = False  # 첫 요청의 전체 payload를 한 번만 덤프하기 위한 플래그

    while True:
        # 버퍼가 비어 있으면 서버에 새 action 시퀀스를 요청
        if len(action_buffer) == 0:
            # 1) obs에서 OpenVLA 입력 구성
            # 학습 / 공식 평가 파이프라인(libero_utils.get_libero_image 등)과 동일하게
            # 180도 회전된 이미지를 사용하여 분포를 맞춘다.
            full_image = obs["agentview_image"][::-1, ::-1]
            wrist_image = obs["robot0_eye_in_hand_image"][::-1, ::-1]

            # state 8차원: [eef_pos(3), axis-angle(3), gripper_qpos(2)]
            # => LIBERO 평가 코드(run_libero_eval.py) 및 학습 파이프라인과 동일한 포맷
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            gripper_qpos = obs["robot0_gripper_qpos"]
            state = np.concatenate(
                [eef_pos, quat2axisangle(eef_quat), gripper_qpos], axis=0
            ).astype("float32")

            # OpenVLA 서버 예제와 동일하게, numpy를 포함한 dict를 한 번 더 JSON 문자열로 감싼
            # double-encoding 형식을 사용한다.
            #
            #   obs_for_vla = {
            #       "instruction": str,
            #       "full_image": np.ndarray(H, W, 3, uint8),
            #       "wrist_image": np.ndarray(H, W, 3, uint8),
            #       "state": np.ndarray(8, float32),
            #   }
            #   encoded = json.dumps(obs_for_vla)  # json_numpy.patch() 덕분에 numpy 직렬화 가능
            #   payload = {"encoded": encoded}
            obs_for_vla = {
                "instruction": instruction,
                "full_image": full_image.astype("uint8"),
                "wrist_image": wrist_image.astype("uint8"),
                "state": state.astype("float32"),
            }
            encoded = json.dumps(obs_for_vla)  # json_numpy.patch() 적용
            payload = {"encoded": encoded}

            # 디버깅용: 서버로 보내는 payload 일부 + 전체 JSON 덤프 (한 번만)
            print(
                f"[run_env] request payload: "
                f"instruction='{instruction}', "
                f"full_image_shape={full_image.shape}, "
                f"wrist_image_shape={wrist_image.shape}, "
                f"state_shape={state.shape}"
            )
            if not payload_dumped:
                debug_path = os.path.join(REPO_ROOT, "debug_openvla_payload.json")
                try:
                    with open(debug_path, "w", encoding="utf-8") as f:
                        # 디버그 파일에는 디코딩 전/후를 모두 남겨 둔다.
                        json.dump(
                            {
                                "encoded_wrapper": payload,
                                "decoded_obs_preview": {
                                    "instruction": obs_for_vla["instruction"],
                                    "full_image_shape": list(full_image.shape),
                                    "wrist_image_shape": list(wrist_image.shape),
                                    "state_shape": list(state.shape),
                                },
                            },
                            f,
                        )
                    print(f"[run_env] 전체 POST payload(wrapper+preview)를 '{debug_path}'에 저장했습니다.")
                except Exception as e:
                    print(f"[run_env] payload 저장 실패: {e}")
                # 터미널에서도 바로 복사할 수 있도록 JSON 문자열을 한 번 출력
                try:
                    print("[run_env] === BEGIN FULL PAYLOAD JSON ===")
                    print(json.dumps(payload))
                    print("[run_env] === END FULL PAYLOAD JSON ===")
                except Exception as e:
                    print(f"[run_env] payload 직렬화 출력 실패: {e}")
                payload_dumped = True

            try:
                resp = requests.post(openvla_url, json=payload, timeout=300)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                print(f"[run_env] OpenVLA 요청 실패: {e}")
                break

            # 서버 응답 디버깅
            print(f"[run_env] raw server response: {result!r}")

            # 서버는 double-encoding 모드에서 numpy 정보를 문자열로 반환한다.
            # 문자열이면 다시 json.loads로 파싱해서 리스트로 복원한다.
            if isinstance(result, str):
                try:
                    decoded = json.loads(result)  # json_numpy.patch() 적용 덕분에 np.ndarray로 복원 가능
                except Exception as e:
                    print(f"[run_env] 서버 응답 디코딩 실패: {e}")
                    break
            else:
                decoded = result

            # decoded는 action 벡터들의 리스트 (각 원소 shape (7,))
            if not isinstance(decoded, (list, tuple)) or len(decoded) == 0:
                print(f"[run_env] 예상치 못한 action 형식: {type(decoded)}, 값={decoded}")
                break

            # 전체 시퀀스를 버퍼에 저장
            action_buffer = [np.array(a, dtype=np.float32) for a in decoded]
            print(f"[run_env] 새 action 시퀀스 수신: {len(action_buffer)}개")

        # 버퍼에서 하나 꺼내 사용
        raw_action = action_buffer.pop(0)

        # 2) 액션 후처리 (그리퍼 정규화 및 OpenVLA 부호 보정)
        #    - normalize_gripper_action: [0, 1] → [-1, +1] 및 binarize
        #    - invert_gripper_action: OpenVLA에서 사용한 그리퍼 부호 convention을 환경과 맞추기 위해 다시 뒤집기
        action = normalize_gripper_action(raw_action, binarize=True)
        action = invert_gripper_action(action)  # OpenVLA 모델 호출을 가정

        # 3) 환경 스텝
        obs, reward, done, info = env.step(action.tolist())
        env.env.render()
        sim_step += 1
        print(
            f"step: {sim_step}, "
            f"buffer_remaining={len(action_buffer)}, "
            f"RawAction: {raw_action}, "
            f"ProcessedAction: {action}, reward={reward}, done={done}"
        )
        if done:
            obs = env.reset()
            action_buffer = []


if __name__ == "__main__":
    main()
