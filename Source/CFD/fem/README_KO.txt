유한요소법 원통 후류 코드 — 2026-11-01

이 패키지는 Python으로 2차원 비압축성 Navier–Stokes 방정식을 계산한다.
ROM, SINDy, KvN, 양자 회로는 유동 생성 과정에 사용하지 않는다.
OpenFOAM 실행 코드가 아니며 Geneva–Zabaras 공개 데이터의 원 계산 설정을 재현한다고 주장하지 않는다.

의존성 설치: python -m pip install -r requirements.txt
Re100 전체 계산: python run_fem.py --reynolds 100
Re100/200/300 전체 계산: python run_fem.py --reynolds 100 200 300
기존 Re100 입력으로 정확한 원본 실행 재현: python run_fem.py --reference-replay

run_fem.py는 정상장을 새로 계산한 뒤 비정상 Navier–Stokes를 적분한다.
reference-replay는 원래 저장한 계산망·정상장 입력을 사용하며, 입력 경로만 공통 resolver에 연결한 legacy solver로 시간 전파한다. 수치 계산식은 유지한다.
두 재현 검사는 별개다. 정상장 재계산의 수치 일치와 전체 스냅숏의 일치 결과는 함께 제공한 보고서에 있다.

무차원화: 원통 지름 D=1, 자유류 속도 U=1. Re=U D/nu이며 nu=1/Re.
전체 계산 영역: -10<=x/D<=20, -10<=y/D<=10. 원통 반경=0.5.
입구·위·아래: 속도(1,0) 고정. 원통: 속도0. 출구: nu du/dn - p n =0의 자연 조건.
정상장: 상반부에서 대칭 조건과 Newton 반복으로 계산하고, 반사하여 전체 장을 구성한다.
공간 이산화: 삼각형의 P2 속도/P1 압력(Taylor–Hood), 적분 차수5.
시간 이산화: 시작1단계 IMEX Euler, 이후 BDF2/AB2. 정상장 대류 Jacobian은 암시적으로 처리하고 비선형 나머지는 매 단계 새로 계산한다.
simulate.py는 상수 점성만 Re에 따라 바꾸며, 비선형 대류 전체를 유지한다.
Re200/300에서 기존 dt=0.02는 불안정하여 더 작은 시간 간격을 검증한다. 최종 사용값은 verified_timesteps.json에 기록한다.
완료는 시간적·공간적 수렴의 일반 증명이 아니다. 이 코드는 원고의 명시된 유한 영역 벤치마크다.

초기조건:
  u_initial = u_steady + delta_u.
  u_steady는 해당 Re의 독립 정상 Navier–Stokes 해다.
  교란 유선함수 psi =0.02 chi(x) exp(-((x-2)^2+y^2)/0.7^2).
  delta_u=(dpsi/dy,-dpsi/dx)이며, 고정 속도 경계에서는 교란을0으로 둔다.
  q=clip((x-0.7)/0.5,0,1), chi=q^3(10-15q+6q^2).
  이 절단함수는 원통 위에서 교란이0이 되도록 한다.
  마지막으로 질량행렬을 이용해 이산 발산0 공간으로 투영한다.
  0.02는 유선함수 계수이며 최대 속도 교란의 크기가 아니다.

초기조건의 의미:
원통 주변의 대칭 박리 정상장에서 작은 비대칭 교란이 성장하여 주기 와류 방출로 이어지는 과도현상을 보는 실험이다.
균일 유입장에서 장애물 주변 유동을 처음 형성하는 시작 실험과는 초기 과도구간이 다르다.
정상해 근처에서 시작하는 구성은 원통 후류 연구에 사용된다. 다만 여기의 Gaussian 모양·계수는 명시한 벤치마크 선택이며,
문헌의 선형 불안정 고유모드를 초기 교란으로 그대로 사용한 것은 아니다.
각 Re의 초기장은 해당 Re의 정상해로 따로 계산한다. Re100 정상장을 Re200/300에 재사용하지 않는다.

참고:
Noack et al. (2003), A hierarchy of low-dimensional models for the transient and post-transient cylinder wake.
https://doi.org/10.1017/S0022112003006694
Ehlert et al. (2019), Locally linear embedding for transient cylinder wakes.
https://arxiv.org/abs/1906.07822
위 논문의 초기 교란은 불안정 고유모드다. 본 코드의 국소 유선함수 교란과 구분해야 한다.
Barkley & Henderson (1996), Three-dimensional Floquet stability analysis of the wake of a circular cylinder.
https://doi.org/10.1017/S0022112096002777
Re200/300 결과는 2차원으로 제한한 Navier–Stokes 벤치마크다. 실제 3차원 후류의 검증 결과로 해석하지 않는다.

출력:
results/Re*_coarse/finite_element_solution.npz: 정상장 전체 유한요소 계수와 계산망.
cfd/Re*_coarse_dt*_train/snapshots.npz: 공통 관측 격자128x64의 두 속도 성분.
cfd/.../final_state.npz: 마지막 두 시간의 전체 속도 계수.
cfd/.../config.json, diagnostics.json, summary.json: 설정과 수치 진단.
관측 격자[-1,7]x[-2,2]는 계산망 자체가 아니다.
정상장과 비정상장은 동일한 계산망·외부 경계조건을 사용한다.

파일 구성:
steady.py: Re를 매개변수로 받는 정상장 계산 및 계산망 생성.
simulate.py: Re를 매개변수로 받는 시간 전파.
../replay_fem/: Re100 시간 전파 코드. 정상장 입력은 공통 Source_Data에서 찾는다.
../replay_fem/inputs/steady_coarse.npz: 과거 입력 사본. 현재 실행의 기본 입력 경로로 사용하지 않는다.
계산 결과 전체와 양자 관련 파일은 이 코드 전용 ZIP에 포함하지 않는다.

2026-11-25 통합 반영:
- reference-replay는 공통 Source_Data의 Shared_source_data/CFD_ROM/FEM/inputs/steady_coarse.npz를 사용한다.
- --data-root /path/to/Source_Data 또는 KVN_SOURCE_DATA로 위치를 지정할 수 있다.
- 로컬 자료가 없으면 기존 Zenodo resolver가 캐시/다운로드를 처리한다. 명시한 데이터 경로가 잘못되었거나 정상장 파일을 찾지 못하면 solver 출력 폴더 생성 전에 종료한다.
- 시간별 항력·양력의 원본 재현: 코드 루트에서 python -m Reproduction.reproduce_cfd_forces --verify
- 이 명령은 복구한 별도 원본 해석기로 압력·점성 항력과 양력을 모두 계산한다.
- 일반 simulate.py는 검수된 속도 시간발전 코드로 보존한다. 원래 Fig. 4의 힘 재현에는 위 명령을 사용한다.

공통 입력 경로:
- simulate.py와 replay_fem/cfd_benchmark.py는 --input /path/to/steady.npz로 생성한 정상장을 직접 받을 수 있다.
- --input을 생략하면 manifest를 통해 공개 Re100 coarse 정상장을 읽는다. 다른 Re/mesh에는 대응하는 --input이 필요하다.
- run_fem.py 일반 실행은 새로 계산한 정상장을 --input으로 전달하며 코드 내부 inputs 폴더에 복사하지 않는다.
