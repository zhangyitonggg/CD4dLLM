from constrained_diffusion.constrain_utils import CompiledLexMap, LexMap
from constrained_diffusion.eval.dllm.datasets.generic import Instance
from rustformlang.cfg import CFG


class Model(object):
    """
    Base class for all models to be evaluated.
    """

    def tokenizer(self, device):
        """
        Returns the tokenizer for the model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def model(self, device):
        """
        Returns the model for the specified device.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_unconstrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        steps: int,
        gen_length: int,
        temperature: int,
        alg: str = "low_confidence",
        trace: bool = False,
        block_length=32,
    ) -> tuple[str, str, str, bool, float]:
        """
        Generates a response from the model based on the provided instance.
        Returns
        - The derived prompt for the model
        - The generated code as a string.
        - The extracted result from the instance.
        - A boolean indicating whether the generation timed out.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_constrained(
        self,
        instance: Instance,
        model,
        tokenizer,
        steps: int,
        gen_length: int,
        temperature: float,
        lang: CFG,
        lex_map: CompiledLexMap,
        orig_lex_map: LexMap,
        subtokens,
        additional_stuff,
        max_total_injections: int = 0,
        inject_gap_size: int = 0,
        prelex: str | None = None,
        alg: str = "low_confidence",
        timeout: int = 60,
        trace: bool = False,
        block_length=32,
    ) -> tuple[str, str, list[str], str, bool, list, str, str, float, float]:
        """
        Generates a response from the model based on the provided messages and additional constraints.
        Returns
        - The derived prompt for the model
        - The generated code as a string.
        - The generated code as a string (including special tokens).
        - The extracted result from the instance.
        - A boolean indicating whether the generation timed out.
        - A valid (raw) completion derived from the constraint language (if the main method timed out).
        - A valid completion derived from the constraint language (if the main method timed out).
        - The time taken to derive the autocompletion.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_ours(
        self,
        instance: Instance,

        model,
        tokenizer,
        
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: int,
        timeout: int = 120,
        trace: bool = False,
        change_logits: bool = False,

        top_k_per_mask: int = 5,
        top_n_beam: int = 30,
        random_n_beam: int = 20,
        max_retry_num_total: int = 1000, 
    ):
        """
        New version of our constrained generation method.
        """
        raise NotImplementedError("Subclasses must implement this method.")