from Config import Config
import numpy as np
import logging
from enums import Action
import copy


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
        # -1      outside map
        self._empty_sea_code = 0
        self._pirate_code = 1
        self._merchant_code = 2
        self._enemy_code = 3
        self._outside_map_code = -1

        ## initialize parameters which are independant of config
        # set visibility of pirate (change this value only if absolutely necessary)
        self._visibility_of_pirate = 2       # how many fields (in each direction) does the pirate see (i.e. 2 means the pirate seesa square of 25 fields)
        self._enemy_neg_reward = -10
        self._step_neg_reward = 0
        self._merchant_pos_reward = 10
        self._step_counter = 0          # this parameter is used to determine every other step (merchant and enemies move only every other step)

        ## params for external use
        self.observation_space = (self._visibility_of_pirate*2+1)**2
        self.action_space = len(Action)

        # initialize map
        self._map = self.__create_map()

    def print_map(self):
        i = self.config.env_size + self._visibility_of_pirate
        print(self._map[2:i, 2:i], "\n")

    def step(self, action: int):
        """
        This function moves the pirate. If the flag "move_enemies_merchants" is set to TRUE then these ships are moved too.
        Otherwise they stay at their initial place.

        :return:
        new_state = what the agent sees in the new step (square area around himself)
        reward = reward for this action
        done = boolean, if game is over
        info = empty string, only to keep the same interface as openAI
        """
        # 1. #############################################################################

        # initialize return values:
        new_state = 0
        reward = 0
        done = False
        info = ""
        

        ##### move pirate #####
        # get pirate action for random case
        if np.random.uniform(0,1,1) > self.config.env_action_success_prob:
            action = np.random.randint(0,4)
            logging.info(f'Random action: {action}\n')

        # get pirate position
        res = np.where(self._map == self._pirate_code)
        old_pirate_position = int(res[0]), int(res[1])
        # get new position
        new_pirate_position = self.__get_new_position(action, old_pirate_position)

        # if new position not on map --> stay
        if self._map[new_pirate_position] == self._outside_map_code:
            new_pirate_position = old_pirate_position

        ##### move other ships #####
        #if self.config.env_move_enemies_merchants:
        if True:
            self._map = self.__get_tmp_map_with_moved_enemies_merchants()

        ##### compute reward and place pirate to new position #####
        # if pirate is on enemy position
        if self._map[new_pirate_position] == self._enemy_code:
            new_state = np.zeros(self.observation_space)
            reward = self._enemy_neg_reward
            done = True
            info = ""
            return (new_state, reward, done, info)
        # if pirate is on merchant position
        elif self._map[new_pirate_position] == self._merchant_code:
            self._map[old_pirate_position] = self._empty_sea_code
            self._map[new_pirate_position] = self._pirate_code
            new_state = self.__get_state()
            reward = self._merchant_pos_reward
            done = True
        else:
            self._map[old_pirate_position] = self._empty_sea_code
            self._map[new_pirate_position] = self._pirate_code
            new_state = self.__get_state()
            reward = self._step_neg_reward


        logging.info(f'Reward: {reward}\n'
                     f'Action: {action}\n'
                     f'Done: {done}\n'
                     f'Map:\n {self._map}\n')

        return new_state, reward, done, info

    def __get_tmp_map_with_moved_enemies_merchants(self):
        """
        This function moves the enemy and merchant ships. It will only be activated if the flag (move_enemies_merchants)
        is set to true in the config.
        Otherwise the enemy and merchant ships stay where they are.


        Rules for overlapping ships:
        - stay if
            -- outside map
            -- on new position is an enemy or merchant (with new position)
            -- on new position is an enemy or merchant (with old position)


        :return:
        map with moved enemies and merchants (but no pirate)

        """

        # temporary map
        tmp_map = copy.deepcopy(self._map)
        # delete pirate (pirate will be added in the step function
        tmp_map[tmp_map == self._pirate_code] = 0

        # move enemies and merchants only every other step (and handle possible collisions of pirate or merchants)
        self._step_counter+=1
        # move merchants and enemies every other step
        if self._step_counter%2 == 0:

            # set everything inside map to 0
            tmp_map[tmp_map > 0] = 0

            # get positions
            res_enemies = np.asarray(np.where(self._map == self._enemy_code))
            res_merchants = np.asarray(np.where(self._map == self._merchant_code))

            ############# move enemies first (merchants will stay if field is occupied by enemy) #############

            for i in range(res_enemies.shape[1]):
                tmp_old_enemy_position = res_enemies[0,i], res_enemies[1,i]
                # get a random action
                tmp_action = np.random.choice(list(Action)).value
                tmp_new_enemy_position = self.__get_new_position(tmp_action, tmp_old_enemy_position)

                # we have to delete the enemy_code on the old map for this enemy because below we have to check if
                # on the new position there is an enemy (and we dont want to compare the same enemy)
                self._map[tmp_old_enemy_position] = self._empty_sea_code

                # move enemy only if new position is
                    # 1. not outside map
                    # 2. not like old position of other enemy or merchant
                    # 3. not like new position of other enemy or merchant
                if(
                    # 1. not outside map
                    tmp_map[tmp_new_enemy_position] == self._outside_map_code or
                    # 2. not like old position of other enemy or merchant
                    self._map[tmp_new_enemy_position] == self._enemy_code or
                    self._map[tmp_new_enemy_position] == self._merchant_code or
                    # 3. not like new position of other enemy or merchant
                    tmp_map[tmp_new_enemy_position] == self._enemy_code or
                    tmp_map[tmp_new_enemy_position] == self._merchant_code
                ):
                    # stay
                    tmp_map[tmp_old_enemy_position] = self._enemy_code
                else:
                    # move
                    tmp_map[tmp_new_enemy_position] = self._enemy_code

            ## exactly the same loop as above but for the merchant
            for i in range(res_merchants.shape[1]):
                tmp_old_merchant_position = res_merchants[0,i], res_merchants[1,i]
                # get a random action
                tmp_action = np.random.choice(list(Action)).value
                tmp_new_merchant_position = self.__get_new_position(tmp_action, tmp_old_merchant_position)

                # we have to delete the merchant_code on the old map for this merchant because below we have to check if
                # on the new position there is a merchant (and we dont want to compare the same merchant)
                self._map[tmp_old_merchant_position] = self._empty_sea_code

                # move merchant only if new position is
                    # 1. not outside map
                    # 2. not like old position of other merchant or merchant
                    # 3. not like new position of other merchant or merchant
                if(
                    # 1. not outside map
                    tmp_map[tmp_new_merchant_position] == self._outside_map_code or
                    # 2. not like old position of other merchant or merchant
                    self._map[tmp_new_merchant_position] == self._enemy_code or
                    self._map[tmp_new_merchant_position] == self._merchant_code or
                    # 3. not like new position of other merchant or merchant
                    tmp_map[tmp_new_merchant_position] == self._enemy_code or
                    tmp_map[tmp_new_merchant_position] == self._merchant_code
                ):
                    # stay
                    tmp_map[tmp_old_merchant_position] = self._merchant_code
                else:
                    # move
                    tmp_map[tmp_new_merchant_position] = self._merchant_code

        return tmp_map

    def __get_new_position(self, action, position):
        """
        This function takes the position of a player (pirate, merchant, enemy), the action this player should execute
        and returns the new position as index.
        :param action:
        :param position:
        :return:
        """
        if action == Action.LEFT.value:
            new_position = (position[0], position[1]-1)
        elif action == Action.DOWN.value:
            new_position = (position[0]+1, position[1])
        elif action == Action.RIGHT.value:
            new_position = (position[0], position[1]+1)
        elif action == Action.UP.value:
            new_position = (position[0]-1, position[1])
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

        :param:
        - none --> checks for itself, where the agent is on the map

        """
        # get position of pirate
        res = np.where(self._map == self._pirate_code)
        pirate_position = int(res[0]), int(res[1])

        # get visible area
        i_row_upper = pirate_position[0]-self._visibility_of_pirate
        i_row_lower = pirate_position[0]+self._visibility_of_pirate + 1
        i_col_left = pirate_position[1]-self._visibility_of_pirate
        i_col_right = pirate_position[1]+self._visibility_of_pirate + 1

        vis_area_matrix = self._map[i_row_upper:i_row_lower, i_col_left:i_col_right]
        # logging.info(f'Visibility area (not flattened):\n {vis_area_matrix}\n')

        # return flattened matrix
        return vis_area_matrix.flatten()

    def get_state(self):
        return np.array(self.__get_state())

    def __create_map(self):
        """
        :param config:
        :return: returns a 2D-array (square) with position of pirate, merchants, enemies:
        """

        # create empty map
        map_side_len = self.config.env_size
        map = np.zeros((map_side_len, map_side_len))

        # create starting point for the pirate (always upper left corner)
        map[0,0] = self._pirate_code

        # possible positions (1 is excluded because pirate starts there)
        possible_positions = np.arange(1,map_side_len**2)
        # select position of merchants
        merchants = np.random.choice(possible_positions, size=self.config.env_nbr_merchants)
        # possible positions except merchant ships
        possible_positions = np.setdiff1d(possible_positions, merchants)
        # enemy positions
        enemies = np.random.choice(possible_positions, size=self.config.env_nbr_enemies)

        # mark merchants and enemies on map
        for m in merchants:
            # convert index to 2d-index
            map[np.unravel_index(m, (map_side_len, map_side_len))] = self._merchant_code

        for m in enemies:
            # convert index to 2d-index
            map[np.unravel_index(m, (map_side_len, map_side_len))] = self._enemy_code

        # expand map (pirate has to see outside map = -1)
        visib_side_len = self.config.env_size + self._visibility_of_pirate*2
        visibility_map = np.full((visib_side_len, visib_side_len), -1)

        # place actual map inside of visible map
        # source: https://stackoverflow.com/questions/40833073/insert-matrix-into-the-center-of-another-matrix-in-python
        lenbig = visibility_map.shape[0]  # side length of big (visibility) map
        lower = self._visibility_of_pirate
        upper = lenbig - self._visibility_of_pirate
        visibility_map[lower:upper, lower:upper] = map

        # logging.info(f'Map side length: {self.config.env_size}\n'
        #          f'Number of merchants : {self.config.env_nbr_merchants}\n'
        #          f'Number of enemies: {self.config.env_nbr_enemies}\n'
        #          f'Map:\n {visibility_map}\n')

        return visibility_map

    def reset(self):
        """This function resets the environment to an initial state. That means:
        - create new map
        - reset _step_counter to 0
        """
        self._step_counter = 0
        self._map = self.__create_map()
        return self.__get_state()