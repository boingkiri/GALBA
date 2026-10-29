import abc

class GuidanceBasedGenerativeOracle(abc.ABC):
    """
    Base class for generative oracles.
    """
    @abc.abstractmethod
    def create_generative_oracle(self, cfg):
        """
        Create a generative oracle based on the configuration.
        """
        pass


    @abc.abstractmethod
    def create_guidance_model(self, cfg):
        """
        Create a guidance model for the generative oracle.
        This method can be overridden by subclasses if needed.
        """
        pass

    @abc.abstractmethod
    def load_guidance_model(self, cfg):
        """
        Load the guidance model from a checkpoint or other source.
        This method can be overridden by subclasses if needed.
        """
        pass

    @abc.abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generate samples from the oracle.
        """
        pass

    # @abc.abstractmethod
    # def get_metrics(self, *args, **kwargs):
    #     """
    #     Compute metrics based on the generated samples.
    #     """
    #     pass