import transformers
from optimum.intel.openvino import OVModelForCausalLM

from typing import Optional, Union
from lm_eval.base import BaseLM


class OptimumIntelAutoCausalLM(BaseLM):
    _DEFAULT_MAX_LENGTH: int = 2048
    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[Union[int, str]] = 1,
        max_gen_toks: Optional[int] = 256,
        device: Optional[Union[int, str]] = "cpu",
        trust_remote_code: Optional[bool] = False
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int,str))

        self._device = device

        revision = revision + ("/" + subfolder if subfolder is not None else "")

        ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
        self.model = OVModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            ov_config=ov_config
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = int(batch_size)
        self._max_gen_toks = int(max_gen_toks)


    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """

        attention_mask = inps.clone()
        attention_mask[:] = 1.0
        return self.model(inps, attention_mask)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {'do_sample': False, 'max_length': max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)
