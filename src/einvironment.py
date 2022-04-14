

class Environment:

    def __init__(self, config: Config):
        """
        The environment should be able to do the following things:
        - create random map, if not specified (params: size, nbr of enemies, nbr of merchants)
        - create specific map
        - ...


        :param config:
        """



        self.Config = config


    def step(self, action: int):
        pass
