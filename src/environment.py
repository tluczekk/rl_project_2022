from Config import Config
import numpy as np
import logging
from enums import Action



class Environment:

    def __init__(self, config: Config):
        """
        The environment should be able to do the following things:
        - create random map, if not specified (params: size, nbr of enemies, nbr of merchants)
        - create specific map
        - ...
        :param config:
        """
        self.config = config

        # set encoding for pirate, merchants, enemies
        # nbr --> field type
        # 0       empty sea
        # 1       pirate ship
        # 2       merchant ship
        # 3       enemy ship
        self._pirate_code = 1
        self._merchant_code = 2
        self._enemy_code = 3
        self._outside_map_code = -1

        # initialize map
        self.map = self.__create_map()

        # set visibility of pirate (change this value only if absolutely necessary)
        self._visibilty_of_pirate = 2       # how many fields (in each direction) does the pirate see (i.e. 2 means the pirate sees
                                            # a square of 25 fields)

    def step(self, action: int):
        # move merchants and enemies

        # move pirate

        # get reward

        #
        self.__get_state()

    def __move_ships(self, action):
        """
        This function moves pirate, merchants and enemies.
        The pirate will be moved according to the action of the agent,
        the other ships will be moved randomly.

        Rules for overlapping ships:
        - merchants and enemies cannot be on the same field (if the random movement results in this the stay where they are)
        - pirate and enemies or pirate and merchants can be on the same field
            -- pirate and merchant --> merchant dissapears, pirate gets reward
            -- pirate and enemy --> game over, negative reward
        """

        # we need 2 maps to dissol

        # move pirate
        # random case
        if np.random.uniform(0,1,1) > self.config.env_action_success_prop:
            pass
        # else move pirate according to action
        else:
            pass
            # get pirate location
            res = np.where(self._visibility_map == self._pirate_code)
            pirate_location = int(res[0]), int(res[1])
            # get new location
            new_location = self.__get_new_location(action, pirate_location)
            # if new location not on map --> stay
            if self._visibility_map[new_location] != -1:


        # move merchands and enemies

    def __get_new_location(self, action, position):
        """
        This function takes the location of a player (pirate, merchant, enemy), the action this player should execute
        and returns the new location as index.
        :param action:
        :param position:
        :return:
        """
        new_position = position
        if action == Action.LEFT.value:
            new_position[1] = position[1]-1
        elif action == Action.DOWN.value:
            new_position[0] = position[0]-1
        elif action == Action.RIGHT.value:
            new_position[1] = position[1]+1
        elif action == Action.UP.value:
            new_position[0] = position[1]+1
        else:
            raise ValueError("Wrong action value. An action can only take the values 0, 1, 2, 3.")

        return new_position


    def __get_state(self):
        """
        This function returns an array which represents the visible area around the pirate ship.
        The area around the pirate ship is defined as a square with sidelenth = 5, where the pirate is in the middle
        field of this square (see example below)

        00000
        00000
        00100   --> 1 = Pirate
        00000
        00000

        The array returned is the flattened version of the values of this square (value definition see self.__init__()).
        So this array has always the form [n1, n2, ..., n13, ..., n25], where n1 - n25 (except n13) represent the field
        type (sea, merchant, enemy, outside map), and n13 always represents the pirate itself (this value could have been
        omitted because it does never change, but for reasons of readability it has been kept).

        The above visibility matrix would be returned like this:
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,]

        """

        # expand map (pirate has to see outside map = -1)
        visib_side_len = self.config.env_size + self._visibilty_of_pirate*2
        self._visibility_map = np.full((visib_side_len, visib_side_len), -1)

        # place actual map inside of visible map
        # source: https://stackoverflow.com/questions/40833073/insert-matrix-into-the-center-of-another-matrix-in-python
        lenbig = self._visibility_map.shape[0]  # side length of big (visibility) map
        lower = self._visibilty_of_pirate
        upper = lenbig - self._visibilty_of_pirate
        self._visibility_map[lower:upper, lower:upper] = self.map

        # get location of pirate
        res = np.where(self._visibility_map == self._pirate_code)
        pirate_location = int(res[0]), int(res[1])

        # get visible area
        i_row_upper = pirate_location[0]-self._visibilty_of_pirate
        i_row_lower = pirate_location[0]+self._visibilty_of_pirate + 1
        i_col_left = pirate_location[1]-self._visibilty_of_pirate
        i_col_right = pirate_location[1]+self._visibilty_of_pirate + 1

        vis_area_matrix = self._visibility_map[i_row_upper:i_row_lower, i_col_left:i_col_right]
        logging.info(f'Visibility area (not flattened):\n {vis_area_matrix}\n')

        # return flattened matrix
        return vis_area_matrix.flatten()

    def __create_map(self):
        """
        :param config:
        :return: returns a 2D-array (square) with location of pirate, merchants, enemies:
        """

        # create empty map
        map_side_len = self.config.env_size
        map = np.zeros((map_side_len, map_side_len))

        # create starting point for the pirate (always upper left corner)
        map[0,0] = self._pirate_code

        # possible locations (1 is excluded because pirate starts there)
        possible_locations = np.arange(1,map_side_len**2)
        # select location of merchants
        merchants = np.random.choice(possible_locations, size=self.config.env_nbr_merchants)
        # possible locations except merchant ships
        possible_locations = np.setdiff1d(possible_locations, merchants)
        # enemy locations
        enemies = np.random.choice(possible_locations, size=self.config.env_nbr_enemies)

        # mark merchants and enemies on map
        for m in merchants:
            # convert index to 2d-index
            map[np.unravel_index(m, (map_side_len, map_side_len))] = self._merchant_code

        for m in enemies:
            # convert index to 2d-index
            map[np.unravel_index(m, (map_side_len, map_side_len))] = self._enemy_code

        logging.info(f'Map side length: {self.config.env_size}\n'
                 f'Number of merchants : {self.config.env_nbr_merchants}\n'
                 f'Number of enemies: {self.config.env_nbr_enemies}\n'
                 f'Map:\n {map}\n')

        return map
