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
        self._pirate_code = 1
        self._merchant_code = 2
        self._enemy_code = 3
        self._outside_map_code = -1

        ## initialize parameters which are independant of config
        # set visibility of pirate (change this value only if absolutely necessary)
        self._visibility_of_pirate = 2       # how many fields (in each direction) does the pirate see (i.e. 2 means the pirate seesa square of 25 fields)
        self._enemy_neg_reward = -10
        self._step_neg_reward = -1
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
        This function moves pirate, merchants and enemies, and returns the new state, reward, done.
        The pirate will be moved according to the action of the agent,
        the other ships will be moved randomly but only every other step.

        Rules for overlapping ships:
        - merchants and enemies cannot be on the same field (if the random movement results in this the stay where they are)
            -- if a pirate ship wants to go where a merchant was in the last step --> pirate stays
            -- if a merchant ship wants to go where a pirate is in the new step --> merchant stays
        - pirate and enemies or pirate and merchants can be on the same field
            -- pirate and merchant --> merchant dissapears, pirate gets reward
            -- pirate and enemy --> game over, negative reward

        The function does this as follows:
        1. move pirate to new location (stay if new location would be outside of map)
        2.
            2.1: move enemies and handle collisions for every other step
                2.1.1: move enemies (stay if merchant was there last step)
                2.1.2: move merchants (stay if enemy is there in new step)
            2.2: hanlde collisions for every other step
                2.2.1: for merchants
                2.2.2: for enemies
        3. return values

        :param:
        new_state = what the agent sees in the new step (square area around himself)
        reward = reward for this action
        done = boolean, if game is over
        info = empty string, only to keep the same interface as openAI
        """
        # we need 2 maps to dissolve cases, where 2 ships are on the same field
        new_visibility_map = copy.deepcopy(self._map)
        # set all fields on the map to 0
        new_visibility_map = np.where(new_visibility_map > 0, 0, new_visibility_map)

        # 1. #############################################################################
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
        if new_visibility_map[new_pirate_position] == self._outside_map_code:
            new_visibility_map[old_pirate_position] = self._pirate_code
        # else move pirate to new position
        else:
            new_visibility_map[new_pirate_position] = self._pirate_code

        # initialize return values (in some cases, reward and done will be overriden below)
        reward = self._step_neg_reward
        done = False
        info = ""

        # 2. #############################################################################

        # 2.1 ## move enemies and merchants only every other step (and handle possible collisions of pirate or merchants)
        self._step_counter+=1
        # move merchants and enemies every other step
        if self._step_counter%2 == 0:

            ############# move merchands and enemies #############
            action_values = list(map(lambda x: x.value, Action._member_map_.values()))

            # 2.1.1 ############ move enemies first (merchants will stay if field is occupied by enemy) #############
            res_enemies = np.asarray(np.where(self._map == self._enemy_code))

            for i in range(res_enemies.shape[1]):
                tmp_old_enemy_position = res_enemies[0,i], res_enemies[1,i]
                # get a random action
                tmp_action = np.random.choice(list(Action)).value
                tmp_new_enemy_position = self.__get_new_position(tmp_action, tmp_old_enemy_position)

                # end game if enemy and pirate are on same position
                if tmp_new_enemy_position == new_pirate_position:
                    new_state = np.zeros(25)
                    reward = self._enemy_neg_reward
                    done = True
                    return (new_state, reward, done, info)
                # if enemy whants to go to a position where a merchant was in old step --> enemy stays
                elif self._map[new_pirate_position] == self._merchant_code:
                    new_visibility_map[tmp_old_enemy_position] = self._enemy_code
                # else: place enemy on new_position
                else:
                    # don't move enemy if new position is outside map
                    if new_visibility_map[tmp_new_enemy_position] == self._outside_map_code:
                        new_visibility_map[tmp_old_enemy_position] = self._enemy_code
                    else:
                        new_visibility_map[tmp_new_enemy_position] = self._enemy_code

            # 2.1.2 ########## move merchants ##############
            res_merchants = np.asarray(np.where(self._map == self._merchant_code))

            for i in range(res_merchants.shape[1]):
                tmp_old_merchant_position = res_merchants[0,i], res_merchants[1,i]
                # get a random action
                tmp_action = np.random.choice(list(Action)).value
                tmp_new_merchant_position = self.__get_new_position(tmp_action, tmp_old_merchant_position)

                # eliminate merchant if pirate is on same position as merchant
                if tmp_new_merchant_position == new_pirate_position:
                    reward = self._merchant_pos_reward
                # if enemy is where merchant wants to go --> merchant stays
                elif new_visibility_map[tmp_new_merchant_position] == self._merchant_code:
                    new_visibility_map[tmp_old_merchant_position] = self._merchant_code
                # else: place merchant on new_position
                else:
                    # don't move merchant if new position is outside map
                    if new_visibility_map[tmp_new_merchant_position] == self._outside_map_code:
                        new_visibility_map[tmp_old_merchant_position] = self._merchant_code
                    else:
                        new_visibility_map[tmp_new_merchant_position] = self._merchant_code

        # 2.2 ## handle collisions for the case that the merchants and enemies did not move
            # (we moved the pirate above but handled the eliminations for the merchant-enemy-move case before)
        else:
            # 2.2.1 ## handle merchant collisions

            # get merchant positions
            res_merchants = np.asarray(np.where(self._map == self._merchant_code))

            for i in range(res_merchants.shape[1]):
                # get merchant position
                tmp_old_merchant_position = res_merchants[0,i], res_merchants[1,i]

                # if collision: eliminate merchant if pirate is on same position as merchant
                if tmp_old_merchant_position == new_pirate_position:
                    reward = self._merchant_pos_reward
                    # don't copy merchant position to new_visibility_map
                # if no collision: place merchant on new_position
                else:
                    new_visibility_map[tmp_old_merchant_position] = self._merchant_code

            # 2.2.2 ## handle enemy collisions
            res_enemies = np.asarray(np.where(self._map == self._enemy_code))

            for i in range(res_enemies.shape[1]):
                # get merchant position
                tmp_old_enemy_position = res_enemies[0,i], res_enemies[1,i]

                # if collision: eliminate enemy if pirate is on same position as merchant
                if tmp_old_enemy_position == new_pirate_position:
                    new_state = np.zeros(25)
                    reward = self._enemy_neg_reward
                    done = True
                    return (new_state, reward, done, info)
                # if no collision: place enemy on new_position
                else:
                    new_visibility_map[tmp_old_enemy_position] = self._enemy_code

        # 3. #############################################################################

        # assign the new positions to the old map
        self._map = new_visibility_map
        # get new state
        new_state = self.__get_state()

        logging.info(f'Reward: {reward}\n'
                     f'Action: {action}\n'
                     f'Done: {done}\n'
                     f'Map:\n {self._map}\n')

        return new_state, reward, done, info

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

    # For external use in training 
    def get_state(self):
        res = self.__get_state()
        print(len(res))
        return np.array(res)

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
        pirate_position = res[0].astype(int), res[1].astype(int)

        # get visible area
        i_row_upper = (pirate_position[0]-self._visibility_of_pirate)[0]
        i_row_lower = (pirate_position[0]+self._visibility_of_pirate + 1)[0]
        i_col_left = (pirate_position[1]-self._visibility_of_pirate)[0]
        i_col_right = (pirate_position[1]+self._visibility_of_pirate + 1)[0]

        tmp_map = np.array(self._map)
        vis_area_matrix = tmp_map[i_row_upper:i_row_lower, i_col_left:i_col_right]
        # logging.info(f'Visibility area (not flattened):\n {vis_area_matrix}\n')

        # return flattened matrix
        return vis_area_matrix.flatten()

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
