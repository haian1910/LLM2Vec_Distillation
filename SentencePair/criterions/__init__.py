from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd import DualSpaceKD
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD
from .uld_att_mined import ULD_ATT_MINED
from .uld_att_mined_cka import ULD_ATT_MINED_CKA
from .dskd_cma_att_mined import DSKD_CMA_ATT_MINED
from .dskd_cma_att_mined_cka import DSKD_CMA_ATT_MINED_CKA
from .rmse_cka import RMSE_CKA
from .rmse import RMSE
from .ot import OT
from .ot_rmse_cka import OT_RMSE_CKA
from .ot_pro import OT_PRO
from .ot_pro_rmse_cka import OT_PRO_RMSE_CKA
from .multi_level_ot import MULTI_LEVEL_OT

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd": DualSpaceKD,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "uld_att_mined": ULD_ATT_MINED,
    "uld_att_mined_cka": ULD_ATT_MINED_CKA,
    "dskd_cma_att_mined": DSKD_CMA_ATT_MINED,
    "dskd_cma_att_mined_cka": DSKD_CMA_ATT_MINED_CKA,
    "rmse_cka": RMSE_CKA,
    "rmse": RMSE,
    "ot": OT,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "ot_rmse_cka": OT_RMSE_CKA,
    "ot_pro": OT_PRO,
    "ot_pro_rmse_cka": OT_PRO_RMSE_CKA,
    "multi_level_ot": MULTI_LEVEL_OT
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")
