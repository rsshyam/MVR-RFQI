from rlkit.torch.sac.policies.seq2act.sd_policy import SDPolicy, SDPolicyAdapter
from rlkit.torch.sac.policies.seq2act.hardcoded_sd_policy import (
    HardCodedSDPolicy,
    HardCodedSDPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.hardcoded_sid_policy import (
    HardCodedSIDPolicy,
    HardCodedSIDPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.sid_policy import SIDPolicy, SIDPolicyAdapter
from rlkit.torch.sac.policies.seq2act.sl_policy import SLPolicy, SLPolicyAdapter
from rlkit.torch.sac.policies.seq2act.frame_stacked_policy import (
    FrameStackPolicy,
    FrameStackPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.frame_diff_policy import (
    FrameDiffPolicy,
    FrameDiffPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.gaussian_seq import (
    SeqGaussianPolicy,
    SeqGaussianPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.indv_sid_policy import (
    IndvSIDPolicy,
    IndvSIDPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.rproj_sid_policy import (
    RprojSIDPolicy,
    RprojSIDPolicyAdapter,
)
from rlkit.torch.sac.policies.seq2act.seq_encoder_policy import (
    SeqEncoderPolicy,
    SeqEncoderPolicyAdapter,
)


ADAPTER_DICT = {
    'sd': SDPolicyAdapter,
    'sid': SIDPolicyAdapter,
    'sl': SLPolicyAdapter,
    'frame_stack': FrameStackPolicyAdapter,
    'frame_diff': FrameDiffPolicyAdapter,
    'gaussian': SeqGaussianPolicyAdapter,
    'hardcoded_sd': HardCodedSDPolicyAdapter,
    'hardcoded_sid': HardCodedSIDPolicyAdapter,
    'indv_sid': IndvSIDPolicyAdapter,
    'rproj': RprojSIDPolicyAdapter,
}
