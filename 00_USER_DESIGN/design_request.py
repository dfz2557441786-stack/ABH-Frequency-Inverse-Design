"""用户只需修改本文件：设置目标频率与是否进行独立 Abaqus 验证。"""
from __future__ import print_function

# 正常检索只修改以下频率和容差；无需修改任何 Abaqus 建模脚本。
TARGET_FREQUENCY_HZ = 213.63
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
DESIGN_MODE = "GENERATE"  # 可选：RETRIEVAL / GENERATE / VALIDATE / GENERATE_AND_VALIDATE
GENERATIVE_ADDITIONAL_TARGETS_HZ = (34.41, 129.30, 213.63)

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

VALIDATE_ONE_NEW_CANDIDATE = True
CANDIDATE_ID_TO_VALIDATE = "FB15_2a401ad636cb"

# 留空时自动读取本项目最新的 fixed_band_15mm_new_abh_candidates.csv。
# 如要验证其他已保存的 FB15 候选表，再填写该 CSV 的完整路径。
SINGLE_VALIDATION_CANDIDATE_CSV = ""
# 旧 Host 耦合/动态任务保持关闭。
RUN_COUPLED_VALIDATION = False
RUN_DYNAMIC_VALIDATION = False
