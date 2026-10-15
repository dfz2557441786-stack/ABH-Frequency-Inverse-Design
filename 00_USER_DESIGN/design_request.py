"""用户只需修改本文件：设置目标频率与是否进行独立 Abaqus 验证。"""
from __future__ import print_function

# 正常检索只修改以下频率和容差；无需修改任何 Abaqus 建模脚本。
TARGET_FREQUENCY_HZ = 125
FREQUENCY_TOLERANCE_PERCENT = 2.0
TOP_N = 10
# 当前使用的样本库。FIXED_BAND_15MM_V2 对应中心平台固支、中心平台打孔、
# 横向和纵向阻尼均向内延伸 15 mm 的独立 ABH 数据。
DATA_LIBRARY = "FIXED_BAND_15MM_V2"
FIXED_BAND_DATASET_CSV = r"C:\Users\ipc\Desktop\ABH_Frequency_Dataset_V2_2\output\dataset.csv"
TARGET_DIRECTION = "Z"
PREFER_HIGH_EFFECTIVE_MODAL_MASS = True
USE_MASS_FILTER = False
MAX_MASS = None
DESIGN_MODE = "GENERATE"  # 可选：RETRIEVAL / GENERATE / MODEL_CHECK / VALIDATE / GENERATE_AND_VALIDATE
# MODEL_CHECK：按几何隔离训练/校准/测试，比较原模型和目标相关模型；绝不提交 Abaqus。
# 不改本配置也可运行：python main.py --design-mode MODEL_CHECK
# 检查结果存入独立 model_check 子目录，不覆盖候选、真实验证结果和生产模型。
# ANCHOR_CHECK：复用已完成的同数据库MODEL_CHECK，比较真实锚点+频率增量修正。
# 命令：python main.py --design-mode ANCHOR_CHECK；不训练新GPR、不提交Abaqus。
GENERATIVE_ADDITIONAL_TARGETS_HZ = (150.0, 250.0)

# 想只检索 129.3 Hz：
# TARGET_FREQUENCY_HZ = 129.3
# DESIGN_MODE = "RETRIEVAL"

# 想生成 50 Hz、34.41 Hz、129.30 Hz、213.63 Hz 的新构型，但不计算 Abaqus：
# TARGET_FREQUENCY_HZ = 50.0
# DESIGN_MODE = "GENERATE"
# GENERATIVE_ADDITIONAL_TARGETS_HZ = (34.41, 129.30, 213.63)

# 阻尼布局：新样本库使用固定实体阻尼带宽时设为 FIXED_BAND_WIDTH。
# X、Y 都是“单侧向内延伸”的物理长度，单位 mm；当前样本库与验证入口
# 锁定为 15 mm / 15 mm。若改成其他值，必须重新建立对应物理带宽的样本库。
DAMPING_LAYOUT_MODE = "FIXED_BAND_WIDTH"
FIXED_DAMPING_BAND_WIDTH_X_MM = 15.0
FIXED_DAMPING_BAND_WIDTH_Y_MM = 15.0

# 单个新构型的独立 Abaqus 验证入口。
# 只支持当前固定 15 mm 样本库的 FB15_... 新构型；横向、纵向阻尼带宽
# 始终分别固定为上面的 15 mm，而非把旧 D02 的无量纲 rx/ry 当作输入。
# 默认 False：普通检索和生成绝不提交 Abaqus。
# VALIDATE_ONE_NEW_CANDIDATE = False
# CANDIDATE_ID_TO_VALIDATE = ""

# 默认保持关闭：只有生成结果明确给出可信候选后，才手动改为 True 并填写候选 ID。
# 这样运行 main.py 不会意外提交旧候选或无关的 Abaqus 作业。
VALIDATE_ONE_NEW_CANDIDATE = False
CANDIDATE_ID_TO_VALIDATE = "FB15EXP_7ba26d2b7cfc"


# 留空时自动读取本项目最新的 fixed_band_15mm_new_abh_candidates.csv。
# 如要验证其他已保存的 FB15 候选表，再填写该 CSV 的完整路径。
SINGLE_VALIDATION_CANDIDATE_CSV = ""

# 批量独立 Abaqus 验证入口（默认关闭）。
# 使用方法：把需要验证的 Candidate_ID 逐行粘贴到下面的元组中，再将开关改为 True，
# 然后只运行：python main.py。
# 程序严格串行运行：前一个候选完成并写入台账后，才会提交下一个候选。
# 已经成功验证的候选会自动跳过；已经尝试过但失败的候选默认也跳过，避免无意重复计算。
VALIDATE_BATCH_NEW_CANDIDATES = False
BATCH_CANDIDATE_IDS = (
    # "FB15EXP_aaaaaaaaaaaa",
    # "FB15EXP_bbbbbbbbbbbb",
    "FB15EXP_2a872f0abe97",
    "FB15EXP_92f8f49570b7",
    "FB15EXP_ea3268d2669e",
    "FB15EXP_2c171806879b",
    "FB15EXP_399b39fddb7c",
    "FB15EXP_623aede8c808",
    "FB15EXP_bdabfa216592",
    "FB15EXP_5eb4cd204793",
    "FB15EXP_ee955cfedb88",
    "FB15EXP_6db628c5aa88",
    "FB15EXP_3437dbaadfcf",
    "FB15EXP_5cd741e400a4",
    "FB15EXP_511ca05a69bb",
    "FB15EXP_fa67f053ea4b",
    "FB15EXP_b861577e23cb",
    "FB15EXP_84f7ed95173e",
    "FB15EXP_5a5c3cf068af",
    "FB15EXP_cf5d3bcd957d",
    "FB15EXP_8a7765a8778a",
    "FB15EXP_2713ef43f578",
    "FB15EXP_cdafe363d8f7",
    "FB15EXP_fa400bd49adf",
    "FB15EXP_143ebbc72bb4",
    "FB15EXP_86afe28df238",
    "FB15EXP_643fd7f5be6b",
    "FB15EXP_3e97a5275227",
    "FB15EXP_35998311e4b1",
    "FB15EXP_bb2177ff5642",
    "FB15EXP_0b47d3ed0832",
    "FB15EXP_b73b7c4db1fe",


)
# 留空时自动读取 fixed_band_15mm_exploratory_candidates.csv。
# 如果 Candidate_ID 位于其他候选表，填写该 CSV 的完整路径。
BATCH_VALIDATION_CANDIDATE_CSV = ""
# 只有确认要重新尝试历史失败工况时才改为 True；默认绝不自动重试。
BATCH_RETRY_PREVIOUSLY_ATTEMPTED = False
# 旧 Host 耦合/动态任务保持关闭。
RUN_COUPLED_VALIDATION = False
RUN_DYNAMIC_VALIDATION = False
