"""用户只需修改本文件：设置目标频率与是否进行独立 Abaqus 验证。"""
from __future__ import print_function

# 正常检索只修改以下频率和容差；无需修改任何 Abaqus 建模脚本。
TARGET_FREQUENCY_HZ = 100.0
FREQUENCY_TOLERANCE_PERCENT = 2.0
TOP_N = 10
# ABH 铝壳体外轮廓的硬约束，单位 mm。X/Y 尺寸由当前 V2 的实际几何公式
# 计算，不包含阻尼层或约束层。检索与新构型生成都会遵守此范围。
USE_ABH_SHELL_SIZE_FILTER = True
ABH_SHELL_OVERALL_LENGTH_X_MM_RANGE = (0.0, 150.0)
ABH_SHELL_OVERALL_WIDTH_Y_MM_RANGE = (0.0, 120.0)
# 当前使用的样本库。FIXED_BAND_15MM_V2 对应中心平台固支、中心平台打孔、
# 横向和纵向阻尼均向内延伸 15 mm 的独立 ABH 数据。
DATA_LIBRARY = "FIXED_BAND_15MM_V2"
# 合并库：原 2215 组 + 首批100组 + 第二批119组小外壳样本，共2434组。
# 第二批 S00079 完成状态无法确认，明确不纳入训练。所有原始 CSV 均保留。
FIXED_BAND_DATASET_CSV = r"C:\Users\ipc\Desktop\ABH_Frequency_Dataset_V2_2\output\dataset_2434_fixed_band_15mm_with_small_shell_20261017.csv"
TARGET_DIRECTION = "Z"
PREFER_HIGH_EFFECTIVE_MODAL_MASS = True
USE_MASS_FILTER = False
MAX_MASS = None
DESIGN_MODE = "RETRIEVAL"  # 验证开关开启时优先执行验证；本值无需改动。
# MODEL_CHECK：按几何隔离训练/校准/测试，比较原模型和目标相关模型；绝不提交 Abaqus。
# 不改本配置也可运行：python main.py --design-mode MODEL_CHECK
# 检查结果存入独立 model_check 子目录，不覆盖候选、真实验证结果和生产模型。
# ANCHOR_CHECK：复用已完成的同数据库MODEL_CHECK，比较真实锚点+频率增量修正。
# 命令：python main.py --design-mode ANCHOR_CHECK；不训练新GPR、不提交Abaqus。
# 本轮需要生成并验证的四个目标频率。生成阶段不提交 Abaqus；验证由后续
# 明确选择的 Candidate_ID 串行完成。
GENERATIVE_ADDITIONAL_TARGETS_HZ = (150.0, 200.0, 250.0)

# 较远新构型探索：python main.py --design-mode GENERATE_FAR
# 或把 DESIGN_MODE 改为 "GENERATE_FAR"。此模式绝不调用 Abaqus。
# 距离 = 九个参数分别除以数据库取值跨度后，计算欧氏距离；不是米。
# 最小距离针对整个数据库，而不是只针对某个锚点。阈值是探索要求，不保证频率精度。
# 结果放在独立 far_generated/<运行标识>/；不覆盖原候选。
FAR_SEARCH_POLICY = {
    "minimum_database_distance": 0.10,
    "maximum_anchor_distance": 0.25,
    "minimum_pair_distance": 0.10,
    "candidate_count": 6,  # 每个目标最多保留几个远距离构型
    "proposal_count": 3000,
    "optimization_starts": 24,
    "optimizer_maxiter": 80,
    "training_maxiter": 40,
    "prediction_tolerance_percent": 2.0,
    "maximum_prediction_std_percent": 12.0,
    "seed": 20261015,
}
# 远距离结果只是待验证探索点。以后批量验证时，显式填写结果CSV路径及候选ID，
# 切回 GENERATE 模式再开启批量验证开关；不要把预测命中当作真实FE命中。

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
CANDIDATE_ID_TO_VALIDATE = ""


# 留空时自动读取本项目最新的 fixed_band_15mm_new_abh_candidates.csv。
# 如要验证其他已保存的 FB15 候选表，再填写该 CSV 的完整路径。
SINGLE_VALIDATION_CANDIDATE_CSV = ""

# 批量独立 Abaqus 验证入口（默认关闭）。
# 使用方法：把需要验证的 Candidate_ID 逐行粘贴到下面的元组中，再将开关改为 True，
# 然后只运行：python main.py。
# 程序严格串行运行：前一个候选完成并写入台账后，才会提交下一个候选。
# 已经成功验证的候选会自动跳过；已经尝试过但失败的候选默认也跳过，避免无意重复计算。
# 验证“far_candidates.csv”中的远距离新构型时设为 True。
# 程序会严格串行运行；每个候选完成后写入检查点，已完成项不会重复提交。
VALIDATE_BATCH_NEW_CANDIDATES = False
BATCH_CANDIDATE_IDS = (
    # 本轮使用下方候选 CSV 的全部 19 个探索点，故此处无需逐个填写。
)
# 本批是“远距离探索”候选，必须读取其原始完整候选表；不要将它们复制到旧的
# exploratory_candidates.csv，因为两张表的字段顺序不同，复制会造成列错位。


BATCH_VALIDATION_CANDIDATE_CSV = (
    r"C:\Users\ipc\Desktop\abh-abaqus-abh-damping-frequency-design"
    r"\ABH_Damping_Frequency_Design\07_RESULTS\inverse_design"
    r"\frequency_9d\fixed_band_15mm\generated"
    r"\fixed_band_15mm_exploratory_candidates.csv"
)
# True：验证本轮最新的 19 个探索候选（100Hz×2、150Hz×3、200Hz×8、250Hz×6）。
# 严格串行；每个结果即时写入 checkpoint。已成功项目不会重复提交。
VALIDATE_ALL_CANDIDATES_FROM_CSV = True
# 只有确认要重新尝试历史失败工况时才改为 True；默认绝不自动重试。
BATCH_RETRY_PREVIOUSLY_ATTEMPTED = False
# 旧 Host 耦合/动态任务保持关闭。
RUN_COUPLED_VALIDATION = False
RUN_DYNAMIC_VALIDATION = False
