import math
from collections import deque
from typing import Tuple, List

import gym
import numpy as np
import heapq as heap
from gym_slap.envs.core_events import (Retrieval, RetrievalFirstLeg,
                                       DeliverySecondLeg, Travel, Delivery,
                                       Event, DeliveryFirstLeg,
                                       RetrievalSecondLeg)
from gym_slap.envs.core_state import State, \
    storage_keys, vehicle_keys
from gym_slap.envs.core_warehouse import SlapWarehouse
from gym_slap.envs.core_logger import SlapLogger
from gym_slap.envs.core_events import SlapEnvEvents
from gym_slap.envs.helpers import faster_deepcopy
from gym_slap.envs.interface_input import Input
# from storage_policies import ClassBasedStorage
from use_case import UseCase


class SlapEnv(gym.Env):
    """

    """

    def render(self, mode='human'):
        pass

    def __init__(self, parameters: Input):
        """
        initializes an environment that has a size of
        n_rows x n_columns x n_levels.
        the slap instance is represented by three n_rows x n_columns x n_levels
        matrices.
        parameters: dict
            dictionary of values that are used to create SlapEnv and
            its object parameters (SlapWarehouse, SlapOrders, etc.). Keys
            include parameters such as number of rows, number of columns,
            initial pallet parameters (storage strategy if any, sku counts if
            any)
        seeds: dict
            random seeds that are used to control stochasticity. seeds are
            used in numpy and random modules
        warehouse: SlapWarehouse
        S: np.array
            storage matrix - numpy array where values represent types of tiles
            (walls, aisles, source, sink, and storage)
        V: np.array
            vehicle matrix - numpy array where values represent AGVs and their
            status (busy, free, etc.)
        T: np.array
            arrival time matrix - numpy array where values represent at what time
            an sku was placed on that tile
        state: State
        orders: SlapOrders
        resetting: bool
           parameter used for debugging - becomes true once reset()
            is executed
        verbose: bool
            parameter used for debugging - if set to true, then all
            print() statements are activated
        SKU_counts: dict
            dictionary where keys are SKU numbers and values are
            how many pallets with that SKU are in the warehouse - at times, may
            not be 100% accurate, i.e. when a retrieval order is placed, this
            gets updated, and not exactly when the pallet gets picked up
        events: SlapEvents
        previous_event: Event
            saves the previous event so its parameters can be
            accessed in future
        current_order: str
            used to keep track of what type of event needs to be
            created next and what type of actions are legal -
            can be either delivery or retrieval
        legal_actions: list of 3D tuples
            tuples represent one location in the numpy
            matrices. can represent what pallets can be retrieved during
            retrieval orders, or what storage locations are empty during
            delivery orders
        state_stack_size: int
            how many states should be stored in state_stack
        state_stack: deque
            saves concatenated state and kicks out
            past states if it doesn't fit
        storage_matrix_history: list of np.arrays
            saves each state's storage matrix into a list
        initial_state: np.array
            first value in state_stack
        logger: Logger
        """
        self.parameters = parameters
        self.seed = parameters.seed
        if self.seed:
            self.set_seed()
        self.use_case = None
        if parameters.environment['use_case']:
            self.use_case = UseCase(parameters.environment['use_case'])
            parameters.environment['n_skus'] = self.use_case.n_skus
        self.warehouse = SlapWarehouse(parameters.environment['n_rows'],
                                       parameters.environment['n_columns'],
                                       parameters.environment['n_levels'],
                                       parameters.environment['n_agvs'],
                                       parameters.environment['n_sources'],
                                       parameters.environment['n_sinks'],
                                       self.use_case)
        S, V, T = SlapEnv._init_matrices(self.warehouse, self.use_case)
        self.state = State(S, V, T, parameters.environment['n_skus'])
        self.orders = SlapOrders(parameters.environment['n_skus'],
                                 parameters.environment['n_orders'],
                                 parameters.environment['generate_orders'],
                                 parameters.environment['desired_fill_level'],
                                 parameters.environment['order_list'],
                                 parameters.environment['initial_pallets_'
                                                        'sku_counts'],
                                 parameters.environment['initial_pallets_'
                                                        'storage_strategy'],
                                 self.state.get_n_storage_locations(),
                                 self.use_case)
        self.resetting = parameters.environment['resetting']
        self.verbose = parameters.environment['verbose']
        self.SKU_counts = {i: 0 for i in range(1, self.orders.n_SKUs + 1)}
        self.events = SlapEnvEvents()
        self.previous_event = None
        self.current_order = "delivery"
        self.legal_actions = self.get_legal_actions()
        self.state_stack_size = parameters.environment['state_stack_size']
        self.state_stack = deque(#TODO ask alexandru if we will use this, delete if not
            parameters.environment['state_stack_size'] *
            [self.state.concatenate()],
            # np.concatenate([self.S, self.V, self.T], axis=1)],
            parameters.environment['state_stack_size'])  # only state after step
        # includes state after reset(), step_no_action(), and step()
        self.storage_matrix_history = [np.copy(S)]
        self.logger = SlapLogger(parameters.logfile_path,
                                 self.state)

    def set_seed(self):
        np.random.seed(self.seed)


    @staticmethod
    def _init_matrices(warehouse, use_case=None):
        """creates initial empty storage, vehicle, and time numpy matrices"""
        n_rows, n_columns, n_levels, n_agvs, n_sources, n_sinks = \
            warehouse.n_rows,\
            warehouse.n_columns, \
            warehouse.n_levels, \
            warehouse.n_AGVs, \
            warehouse.n_sources, \
            warehouse.n_sinks
        if use_case is None:
            S = SlapEnv._init_storage_matrix(n_rows, n_columns,
                                             n_levels, storage_keys,
                                             n_sources, n_sinks)
            S = S.astype(int)

        else:
            S = np.zeros((n_rows, n_columns, n_levels))
            S = S.astype(int)
            S[:, :, 0] = use_case.layout
            S[:, :, 1] = use_case.layout
            S[:, :, 2] = use_case.layout
        V = SlapEnv._init_vehicle_matrix(n_rows, n_columns,
                                         n_agvs, vehicle_keys, S)
        T = SlapEnv._init_arrival_time_matrix(n_rows, n_columns, n_levels)
        return S, V, T

    @staticmethod
    def _init_vehicle_matrix(n_rows: int, n_columns: int, n_agvs: int, keys, storage_matrix) \
            -> np.ndarray:
        """ creates np array and initializes the appropriate number of free
        AGVs
        """
        v = np.full((n_rows, n_columns), keys['nAGV'])
        source_sink_row = int(n_rows / 2)
        storage_locations = np.argwhere(storage_matrix[:,:,0] == storage_keys['empty'])
        index = np.random.randint(0, len(storage_locations))
        storage_location = storage_locations[index]
        for i in range(0, n_agvs):
            index = np.random.randint(0, len(storage_locations))
            storage_location = storage_locations[index]
            v[storage_location[0]][storage_location[1]] = keys['free']
        return v

    @staticmethod
    def _init_storage_matrix(n_rows: int, n_columns: int,
                             n_levels: int, keys: dict,
                             n_sources:int, n_sinks: int) -> np.ndarray:
        """ creates np array using keys that represent aisles, walls, etc.
        """
        s = np.zeros((n_rows, n_columns, n_levels))
        source_rows = [int(n_rows / 2)]
        sink_rows = source_rows.copy()
        s[:, 1:4, :] = keys['aisle']
        # 3 vertical aisles at right of warehouse
        s[:,(n_columns-4): (n_columns-1), :] = keys['aisle']
        # 3 horizontal aisles in middle of warehouse
        s[(source_rows[0]-1):(source_rows[0]+2), 0:n_columns - 1, :] = keys['aisle']
        # middle vertical aisle at right of warehouse
        s[:, n_columns-3, :] = keys['middle_aisle']
        # middle vertical aisle at right of warehouse
        s[:, 2, :] = keys['middle_aisle']
        # traversable aisle tile in front of access tiles
        s[source_rows[0], 0:n_columns - 1, :] = keys['middle_aisle']
        s[:, 0, :] = keys['wall']
        s[:, n_columns-1, :] = keys['wall']
        s[0, :, :] = keys['wall']
        s[n_rows - 1, :, :] = keys['wall']

        s[source_rows[0], 0, :] = keys['source']
        counter = 0
        direction = 1
        scale = 1
        for i in range(0, n_sources-1):
            if i == 2*scale:
                scale += 1
            middle_row = source_rows[0]
            source_rows.append(middle_row+2*direction*scale)
            s[middle_row+2*direction*scale, 0, :] = keys['source']
            s[middle_row+2*direction*scale, 1, :] = keys['middle_aisle']
            direction = direction*-1
            counter += 1

        s[sink_rows[0], n_columns - 1, :] = keys['sink']
        counter = 0
        direction = 1
        scale = 1
        for i in range(0, n_sinks-1):
            if i == 2*scale:
                scale += 1
            middle_row = sink_rows[0]
            sink_rows.append(middle_row+2*direction*scale)
            s[middle_row+2*direction*scale,  n_columns - 1, :] = keys['sink']
            s[middle_row+2*direction*scale, n_columns - 2, :] = keys['middle_aisle']

            direction = direction*-1
            counter += 1

        return s.astype(int)

    def create_orders_from_distribution(self):
        """creates delivery and retrieval orders automatically based on input
        parameters. SKUs are selected randomly and times are created
        with a distribution based on number of storage locations
        """
        self.events.running = []
        sim_time = 0


        def random_sku():
            return int(np.random.randint(1, self.orders.n_SKUs+1))

        # make a copy of SKU_counts because the self variable will be used
        # to keep track of stock during simulation but copy is just used to
        # create orders and make sure retrieval orders aren't created
        # when no stock would be available and delivery orders aren't created
        # when no spaces would be available
        available_source_tiles = [i for i in
                                  range(0, self.warehouse.n_sources)]
        available_sink_tiles = [i for i in range(0, self.warehouse.n_sinks)]
        SKU_counts = self.SKU_counts.copy()
        for i in range(1, self.orders.n_orders):
            total_pallets = sum(SKU_counts.values())
            order_time = self.get_order_time()
            order_type = self.get_order_type(total_pallets)
            if order_type == "delivery":  # delivery order
                source_index = np.random.choice(available_source_tiles)
                self.create_delivery_order(SKU_counts, i, random_sku,
                                           sim_time, source_index)
            elif order_type == "retrieval":  # retrieval order
                sink_index = np.random.choice(available_sink_tiles)
                self.create_retrieval_order(SKU_counts, i,
                                            sim_time, sink_index)
            sim_time += order_time
        self.print_events()

    def print_events(self):
        """just prints first 5 orders when calling reset() and after orders
        are added to heap
        """
        if self.resetting and self.verbose:
            self.print("first 5 events")
            for i in range(0, 5):
                self.print(self.events.running[i])

    def get_order_type(self, total_pallets: int) -> str:
        """determines order type based on how full warehouse is at that point.
        intended to keep number of pallets between a certain range. if it's
        already between that range, then it's a random choice between
        delivery and retrieval
        """
        order_type = np.random.choice(["delivery", "retrieval"])
        # if too many pallets in warehouse,
        # make next order a retrieval order
        if total_pallets > 1.2 * self.orders.average_n_pallets:
            order_type = "retrieval"
        # if too few pallets in warehouse, make next order a delivery order
        if total_pallets < 0.8 * self.orders.average_n_pallets:
            order_type = "delivery"
        return order_type

    def get_order_time(self) -> int:
        """gets arrival time of retrieval and delivery orders. based on a
        normal distribution with set mean and standard deviation. intended to
        create some overlap of orders and travel times but not every single one.
        """
        mean_order_time = self.state.get_n_storage_locations() * 0.4
        std_order_time = self.state.get_n_storage_locations() * 0.1
        order_time = np.random.normal(mean_order_time,
                                      std_order_time, 1)[0]
        return order_time

    def create_delivery_order(self, sku_counts: dict, i: int, random_sku,
                              sim_time: int, source: int):
        """get random sku, create delivery order, push to running event heap,
        and update sku_counts dictionary
        """
        sku = random_sku()
        heap.heappush(self.events.running,
                      Delivery(sim_time, sku, i, self.verbose, source))
        sku_counts[sku] += 1

    def create_retrieval_order(self, sku_counts: dict, i: int, sim_time: int,
                               sink: int):
        """get random feasible sku, create retrieval order, push to running
        event heap, and update sku_counts dictionary
        """
        # if there are no pallets for a specific SKU available,
        # do not make a retrieval order for it
        possible_skus = [sku for sku in range(1, self.orders.n_SKUs + 1)]
        for j in possible_skus:
            if sku_counts[j] == 0:
                possible_skus.remove(j)
        sku = np.random.choice(possible_skus)
        heap.heappush(self.events.running,
                      Retrieval(sim_time, sku, i, self.verbose, sink))
        sku_counts[sku] -= 1

    @staticmethod
    def _init_arrival_time_matrix(n_rows: int,
                                  n_columns: int, n_levels: int) -> np.ndarray:
        """creates np array full of -1 values. represents no pallet on that
        tile
        """
        t = np.full((n_rows, n_columns, n_levels), -1.0)
        return t

    def add_silent_storage_state(self):
        """add storage matrix np array to storage_matrix_history during
        step_no_action()
        """
        self.storage_matrix_history.append(self.state.S)

    def get_state(self) -> np.ndarray:
        """returns most recently added concatenated state"""
        return self.state_stack[0]

    def _get_state(self) -> np.ndarray:
        """concatenates current state, appends it to state_stack and
        storage_state_matrix, returns it as well
        """
        new_state = self.state.concatenate()
        self.state_stack.appendleft(new_state)
        self.add_silent_storage_state()
        return new_state

    def reset(self, refill_warehouse=True):
        """this function should be called to initialize and/or reset the
        slap environment to its initial state. It initializes parameters,
        adds initial pallets, creates orders, logs states. Lastly, it executes
        step_no_action() so that the steps that don't require actions are
        executed and environment is ready to accept an action afterwards"""
        self.print("~" * 150 + "\n" + "reset\n" + "~" * 150)
        previous_storage_matrix = np.copy(self.state.S)
        self.__init__(self.parameters)
        self._assert_orders()
        if not self.orders.generate_orders:
            self.create_orders_from_list()
        else:
            self.create_orders_from_distribution()
        self.state.add_orders(self.events.running)
        if refill_warehouse:
            self._add_initial_pallets()
        else:
            self.state.S = previous_storage_matrix
        self.state.state_cache.calculate_initial_entropy()
        assert self.events.running
        self.logger.log_state()
        self.add_silent_storage_state()
        self.step_no_action()

    def get_state_space_shape(self):
        return self.warehouse.n_rows, self.warehouse.n_columns * 3




    def step_no_action(self):
        """this function is the second most important one in the simulation.
        it executes all of the events in the simulation that do not require an
        action, logs each state, and sets legal actions for the method that
        will be called: step()
        """
        action_needed = False
        # loop through simulation until an action
        # is needed (for delivery second leg event)
        # or until there are no more events or queued orders to process
        while not action_needed and \
                (self.events.running or self.events.n_queued_retrieval_orders()
                 or self.events.queued_delivery_orders):
            self.print("~" * 150 + "\n" + "step no action \n" + "~" * 150)
            next_event = None
            # if there are serviceable queued events, take care of them first.
            if (((self.events.n_queued_retrieval_orders() and
                 self.events.at_least_one_serviceable_retrieval_order(self.state.state_cache))
                 or self.events.queued_delivery_orders and
                 self.state.state_cache.get_open_locations()) and self.free_agv()):
                 # or self.events.queued_delivery_orders and self.state.state_cache.open_storage_locations) and self.free_agv()):
                next_event = self.pop_queued_event()
            else:
                if not self.events.running:
                    return
                next_event = heap.heappop(self.events.running)
            assert next_event is not None
            action_needed = self.handle_event_and_update_env(next_event)

        if not self.events.all_orders_complete(self.state.state_cache):
            # update legal actions for next step()
            self.legal_actions = self.get_legal_actions()
            self.print("legal actions for " + self.current_order +
                       " order: " + str(self.legal_actions))

    def handle_event_and_update_env(self, next_event: Event) -> bool:
        """this function takes care of the bookkeeping, from handling an event -
        (manually) removes any travel events, calculates time between handled
        and previous event, updates state time, adds future events to
        appropriate data structures, logs states, and returns a boolean that
        decides if another event should be handled or if it should move on to
        step()
        """
        if next_event in self.events.current_travel:
            self.events.current_travel.remove(next_event)
        self.print("current time: " + str(next_event.time))
        self.print("popped event: " + str(next_event))
        elapsed_time = round(next_event.time -
                             self.state.time, 2)
        self.state.trackers.update_time(elapsed_time)
        #  travel events need to be updated by time_to_simulate
        self.process_travel_events(elapsed_time)
        self.state.time = next_event.time
        # handle event and see if an action is needed and what data
        # structures should the next (or same) event be added to
        info = next_event.handle(self.state)
        action_needed = info[0]
        event_to_add = info[1]
        travel_event_to_add = info[2]
        queued_retrieval_order_to_add = info[3]
        queued_delivery_order_to_add = info[4]
        self.add_events_to_data_structures(event_to_add,
                                           queued_delivery_order_to_add,
                                           queued_retrieval_order_to_add,
                                           travel_event_to_add)
        self.update_prev_event_and_curr_order(next_event,
                                              queued_retrieval_order_to_add)
        self.print(self.state)
        self.print("SKU counts: " + str(self.SKU_counts))
        if self.verbose:
            self.print_any_events()
        self.state.n_silent_steps += 1
        self.logger.log_state()
        self.add_silent_storage_state()
        return action_needed

    def __print_debug_info(self, action: Tuple[int, int, int]):
        """just prints time and action taken"""
        self.print("~" * 150 + "\n" + "step with action \n" + "~" * 150)
        self.print("time: " + str(self.state.time))
        self.print("given action: " + str(action))

    def track_queued_orders(self):
        self.state.trackers.queued_order_tracking.append([self.state.time,
                                                          self.events.n_queued_retrieval_orders(),
                                                          len(
                                                              self.events.queued_delivery_orders),

                                                         self.events.n_queued_retrieval_orders() + len(self.events.queued_delivery_orders)])
        self.state.trackers.n_free_agvs.append(
            len(self.state.state_cache.free_agv_positions))

    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """key function that takes an action from an agent, randomly, or from
        a storage or retrieval strategy. first action is converted from an int
        to a 3D tuple that fits the warehouse shape, then depending on if the
        current order is a delivery or retrieval order, it creates a
        DeliverySecondLeg or RetrievalFirstLeg event, respectively. Updates
        event data structures, and executes step_no_action(). If the simulation
        is done afterwards, it can be ended here."""
        action_unraveled = np.unravel_index(action, self.warehouse.shape)
        action = action_unraveled
        self.state.n_steps += 1
        self.__print_debug_info(action)
        travel_event = None
        self.track_queued_orders()
        if self.current_order == "delivery":
            travel_event = DeliverySecondLeg(
                state=self.state,
                start_point=self.state.source_positions[
                    self.previous_event.source],
                end_point=action[0:2],
                SKU=self.previous_event.SKU, travel_type="delivery_second_leg",
                order_number=self.previous_event.order_number,
                verbose=self.verbose, level=int(action[2]),
                source=self.previous_event.source,
                order_start_time=self.previous_event.order_start_time)
        elif self.current_order == "retrieval":
            agv_position = self.state.state_cache.get_close_agv(
                action[0:2])
            self.state.state_cache.update_agv_cache(agv_position)
            self.state.update_v_matrix(agv_position, None)
            travel_event = RetrievalFirstLeg(
                state=self.state, start_point=agv_position,
                end_point=action[0:2], SKU=self.previous_event.SKU,
                travel_type="retrieval_first_leg",
                order_number=self.previous_event.order_number,
                verbose=self.verbose, level=int(action[2]),
                sink=self.previous_event.sink,
                order_start_time=self.previous_event.time)
        self.state.add_travel_event(travel_event)
        #  add event to both env_running_events heap and current_travel_events
        #  list at same time (pointer)
        heap.heappush(self.events.running, travel_event)
        self.events.current_travel.append(travel_event)
        self.logger.log_state()
        if not self.events.n_queued_retrieval_orders() and \
                self.events.queued_delivery_orders and \
                not self.state.state_cache.get_open_locations():
            self.print("only deliver orders left but warehouse is full")
            return self._get_state(), True
        self.step_no_action()
        if self.events.all_orders_complete(self.state.state_cache):
            return self._get_state(), True
        elif self.events.running or self.events.queued_delivery_orders or \
                self.events.n_queued_retrieval_orders():
            return self._get_state(), False,  # state, done
        else:
            return self._get_state(), True  # sim done

    def update_prev_event_and_curr_order(self, next_event: Event,
                                         queued_retrieval_order_to_add:
                                         Retrieval):
        """" updates parameters previous_event, current_order, and SKU_counts
        """
        #self.previous_event = next_event # TODO check that previous event isn't getting a different order
        if isinstance(next_event, Delivery) or isinstance(next_event, DeliveryFirstLeg) or isinstance(next_event, DeliverySecondLeg):
            self.current_order = "delivery"
            if isinstance(next_event, DeliveryFirstLeg):
                self.previous_event = next_event
            if isinstance(next_event, DeliverySecondLeg):
                self.SKU_counts[next_event.SKU] += 1
        if isinstance(next_event, Retrieval) or isinstance(next_event, RetrievalFirstLeg) or isinstance(next_event, RetrievalSecondLeg):
            self.current_order = "retrieval"
            if isinstance(next_event, Retrieval) and not queued_retrieval_order_to_add:
                self.previous_event = next_event
                self.SKU_counts[next_event.SKU] -= 1
                self.print("SKU counts: " + str(self.SKU_counts))
        # save delivery first leg travel events so that step() remembers
        # its parameters (path, SKU)

    def pop_queued_event(self) -> Event:
        """this method is only executed if there are queued orders that can be
        handled and there are free AGVs. It is a bit hairy because of many
        specific if statements but these are the three conditions below. Note
        that retrieval orders have higher priority than delivery orders.
        1) if there is at least one serviceable retrieval order, handle
        whichever order one is oldest (i.e. queued first)
        2) if there are no queued retrieval orders, but there are queued
        delivery orders that can be serviced (i.e. there is space in the
        warehouse/ there are legal actions), handle the oldest delivery order
        3) if there are both serviceable queued retrieval orders and queued
        delivery orders, handle whichever one is oldest.
        Finally, since the queued events time are in the past, they are updated
        to be the current state time.
        return the event to be handled
        """
        next_event = None  # to appease inspection
        self.print("picking from queued events")
        # if self.events.queued_retrieval_orders and \
        if self.events.n_queued_retrieval_orders() and \
                not self.events.queued_delivery_orders and \
                self.events.at_least_one_serviceable_retrieval_order(self.state.state_cache):
            next_event = self.events.get_queued_retrieval_order(self.state.state_cache)
            self.events.removed_queued_retrieval_order(next_event)
            # starting from first order, look for retrieval order
            # that can be fulfilled
            # fringe case - if there are available AGVs and queued
            # retrieval orders, but none of them can be fulfilled
            # i.e. no stock left for any desired SKU,
            # then pop self.env_running_events
        elif not self.events.n_queued_retrieval_orders() and \
                self.events.queued_delivery_orders and \
                self.state.state_cache.get_open_locations():
                # self.events.queued_delivery_orders and self.state.state_cache.open_storage_locations:
            next_event = self.events.queued_delivery_orders[0]
            self.events.queued_delivery_orders.remove(next_event)
        # elif self.state.state_cache.open_storage_locations and self.events.queued_delivery_orders:
        elif self.state.state_cache.get_open_locations() and \
                self.events.queued_delivery_orders:
            if self.events.get_first_queued_retrieval_order_time() < \
                    self.events.queued_delivery_orders[0].time and \
                    self.events.at_least_one_serviceable_retrieval_order(self.state.state_cache):
                next_event = self.events.get_queued_retrieval_order(self.state.state_cache)
                self.events.removed_queued_retrieval_order(next_event)

            else:
                next_event = self.events.queued_delivery_orders[0]
                self.events.queued_delivery_orders.remove(next_event)
        # the queued order's time has already passed so it must be
        # updated to current time
        if next_event:
            next_event.time = self.state.time
        return next_event

    def process_travel_events(self, elapsed_time: float):
        """ if time has elapsed, update any currently active travel events"""
        if self.events.current_travel and elapsed_time > 0:
            self.print("simulating travel events by " + str(elapsed_time))
            self.simulate_travel_events(elapsed_time)

    def print_any_events(self):
        """debugging purposes - print any event data structures that aren't
        empty
        """
        if self.events.current_travel:
            self.print("currently active travel events: ")
            for i in self.events.current_travel:
                self.print("active: " + str(i))
        if self.events.n_queued_retrieval_orders():
            self.print("currently queued retrieval orders: ")
            for i in self.events.queued_retrieval_orders_dict.values():
                for j in i:
                    self.print("queued: " + str(j))

        if self.events.queued_delivery_orders:
            self.print("currently queued delivery orders: ")
            for i in self.events.queued_delivery_orders:
                self.print("queued: " + str(i))

        if self.state.state_cache.unserviceable_retrieval_skus:
            self.print("unserviceable SKUs: " + str(self.state.state_cache.unserviceable_retrieval_skus))

    def add_events_to_data_structures(
            self, event_to_add: Event, queued_delivery_order_to_add: Delivery,
            queued_retrieval_order_to_add: Retrieval,
            travel_event_to_add: Travel):
        """this function takes the events that were returned from handling an
        event and adds them to their appropriate data structures. For example,
        if a travel event was created when handling a delivery order, it will
        be added to self.events.current_travel.
        """
        if event_to_add:
            heap.heappush(self.events.running, event_to_add)
        if travel_event_to_add:
            self.events.current_travel.append(travel_event_to_add)
        if queued_retrieval_order_to_add:
            self.print("added retrieval order to queue: " +
                       str(queued_retrieval_order_to_add))
            self.events.add_queued_retrieval_order(queued_retrieval_order_to_add)
        if queued_delivery_order_to_add:
            self.print("added delivery order to queue: " +
                       str(queued_delivery_order_to_add))
            self.events.queued_delivery_orders. \
                append(queued_delivery_order_to_add)

    def _add_initial_pallets(self):
        """ this function adds pallets/SKUs to the initial storage and arrival
        time matrices. It is versatile and can add a random quantity of SKU
        numbers or it can read from a dictionary from self.orders. The location
        of the initial pallets can either be random free locations, or they can
        follow a storage strategy (i.e. closest to source, furthest from source)
        If needed in the future, both SKU number and location can be directly
        given as an input of type numpy array
        """
        n_initial_pallets = 0
        SKU_choices = []
        # first calculate how many initial pallets there are
        # if it is already known how many initial pallets per sku there are, make a list of sku_choices
        if not self.orders.read_sku_from_dict():
            n_initial_pallets = self.orders.average_n_pallets
        elif self.orders.read_sku_from_dict():
            n_initial_pallets = sum(self.orders.initial_pallets_sku_counts.values())
            for sku, count in self.orders.initial_pallets_sku_counts.items():
                for i in range(count): SKU_choices.append(sku)
            self.SKU_counts = self.orders.initial_pallets_sku_counts
        self.print("")

        # loop for each initial pallet, get sku number to add to storage matrix. either random integer or by reading
        # from sku choices.
        # if a storage strategy is being used for the locations, it is done here, otherwise location is random.
        SKU_choices = np.random.choice(SKU_choices, size=len(SKU_choices), replace=False)
        if not self.orders.initial_pallets_as_np_array():
            for i in range(0, n_initial_pallets):
                chosen_sku = 0
                if not self.orders.read_sku_from_dict():
                    chosen_sku = np.random.randint(1, self.orders.n_SKUs+1)
                    self.SKU_counts[chosen_sku] = self.SKU_counts[chosen_sku] + 1
                else:
                    chosen_sku = int(SKU_choices[i])
                    #chosen_sku = int(np.random.choice(SKU_choices))
                    #SKU_choices.remove(chosen_sku)
                self.state.current_sku = chosen_sku
                if self.orders.follow_storage_strategy():
                    index = self.orders.initial_pallets_storage_strategy. \
                        get_action(self.state)
                    index = np.unravel_index(index, self.warehouse.shape)
                else:
                    possible_locations = \
                        self.state.state_cache.get_open_locations()

                    for agv in self.state.state_cache.get_agv_locations():
                        if agv in possible_locations:
                            possible_locations.remove(agv)
                    # try:
                    assert len(possible_locations), 'warehouse full'
                    index = np.random.randint(0, len(possible_locations))
                    # except:
                    #     print()
                    index = possible_locations[index]
                    index = np.unravel_index(index, self.warehouse.shape)
                    # index = np.random.choice(possible_locations)
                self.state.state_cache.update_location_cache(index,
                                                             chosen_sku,
                                                             initial_pallet=True)
                self.state.update_s_t_matrices(index, chosen_sku, 0)
        self.print(str(n_initial_pallets) + " initial pallets added")
        self.print(self.SKU_counts)
        self.print(self.state)
        self.print("")

    def get_legal_actions(self) -> List[int]:
        """returns legal actions depending on what the current order is.
        actions are returned as integers as agents are better at picking 1D
        tuples than 3D
        """
        if self.current_order == "delivery":
            legal_actions = self.get_legal_delivery_actions()
            return legal_actions
        elif self.current_order == "retrieval":
            legal_actions = self.get_legal_retrieval_actions()
            return legal_actions

    def get_legal_retrieval_actions(self) -> List[int]:
        """returns legal actions that are all of the storage locations with
        the desired SKU number
        """
        legal_actions = self.state. \
            state_cache.get_sku_locations(self.previous_event.SKU)
        if self.state.state_cache.there_are_retrievable_out_of_zone_skus():
            legal_actions = \
                self.state.state_cache.get_out_of_zone_sku_locations(
                    self.previous_event.SKU)
        self.state.set_current_order(self.current_order)
        if self.previous_event:
            self.state.set_current_sku(self.previous_event.SKU)
            self.state.set_current_source_sink(self.previous_event.sink)

        self.state.set_legal_actions(legal_actions)
        #legal_actions = [int(np.ravel_multi_index(i, self.warehouse.shape))
        #                 for i in legal_actions]
        return list(legal_actions)


    def get_legal_delivery_actions(self) -> List[int]:
        """returns legal actions that are all empty locations in the warehouse.
        If there are multiple levels, then it only returns the lowest space in
        the stack (of pallets)
        """
        legal_actions = set(self.state
                         .state_cache
                         .get_open_locations())
        # legal_actions = set(self.one_legal_action_per_lane(self.state
        #                  .state_cache
        #                  .get_open_locations()))
        self.state.set_current_order(self.current_order)
        if self.previous_event:
            self.state.set_current_sku(self.previous_event.SKU) # TODO explanation
            self.state.set_current_source_sink(self.previous_event.source)

        self.state.set_legal_actions(legal_actions)
        # converts tuple legal actions to linear index
        #legal_actions = [int(np.ravel_multi_index(i, self.warehouse.shape))
        #                 for i in legal_actions]
        return list(legal_actions)

    def free_agv(self) -> bool:
        """returns true if there are free AGVs, else false"""
        if self.state.state_cache.get_agv_locations():
            return True
        else:
            return False

    def simulate_travel_events(self, elapsed_time: float):  # TODO move to handle?
        """if there are any currently active travel events, they are updated.
        the updated routes are also used for tracking statistics.
        """
        for travel_event in self.events.current_travel:
            prev_total_distance, prev_total_time, previous_first_node = \
                self.handle_route_before_update(travel_event)
            travel_event.route.update_route(elapsed_time)
            self.handle_route_after_update(prev_total_distance,
                                          prev_total_time,
                                          previous_first_node,
                                          travel_event)
        self.print("")

    def handle_route_before_update(self, travel_event: Travel) -> \
            Tuple[int, int, Tuple[int, int]]:
        """this function saves some of the parameters of the route of the travel
        event before it gets updated (total distance, total time, starting node
        of the route) and also removes it from the state's set of current routes
        """
        self.print("before update: " + str(travel_event))
        prev_total_distance = travel_event.route.get_total_distance()
        prev_total_time = travel_event.route.get_duration()
        previous_first_node = travel_event.route.get_first_node()
        self.state.remove_route(travel_event.route.get_indices())
        return prev_total_distance, prev_total_time, previous_first_node

    def handle_route_after_update(self, prev_total_distance: float,
                                 prev_total_time: float,
                                 previous_first_node: Tuple[int, int],
                                 travel_event: Travel):
        """this function updates statistic trackers using the parameters of the
        travel event's route from before it was updated.
        """
        cur_total_distance = travel_event.route.get_total_distance()
        cur_total_time = travel_event.route.get_duration()
        self.state.add_route(travel_event.route.get_indices())
        self.print("after update: " + str(travel_event))
        new_first_node = travel_event.route.get_indices()[0]
        self.state.update_v_matrix(
            previous_first_node, new_first_node)
        dist_traveled = prev_total_distance - cur_total_distance
        time_traveled = prev_total_time - cur_total_time
        self.state.trackers.update_distance(dist_traveled)
        self.state.trackers.update_travel_time(time_traveled)
        travel_event.distance_traveled += dist_traveled

    def create_orders_from_list(self):
        """takes a list of orders as an input and creates events from it
        instead of using a random distribution. An order should be represented
        as a tuple with parameters
        (order_type: str, sku: int, arrival time: int),
        such as ("delivery", 2, 300)
        """
        order_number = 1
        first_order_time = 0
        for order in self.orders.order_list:
            if len(order) == 4:
                order_type, sku, arrival_time, source_sink = order
                batch = 1
            if len(order) == 5: # coming from use case
                order_type, sku, arrival_time, source_sink, batch = order
                source_sink -= 1
            if order_type == "retrieval":
                heap.heappush(self.events.running,
                              Retrieval(arrival_time, sku, order_number,
                                        self.verbose, source_sink, batch))
            elif order_type == "delivery":
                heap.heappush(self.events.running,
                              Delivery(arrival_time, sku, order_number,
                                       self.verbose, source_sink, batch))
            order_number += 1
        self.events.order_times = [i.time for i in
                                   self.events.running]
        self.print("")

    def print(self, string: str):
        """this function can be used instead of the python default print(). It
        allows all print statements to be turned on/off with one parameter:
        verbose
        """
        if self.verbose:
            print(string)

    def _assert_orders(self):
        """asserts that the correct configuration of n_orders, generate_orders,
        and order_list is used"""
        if self.orders.order_list is False:
            assert self.orders.generate_orders
            assert self.orders.n_orders
        if self.orders.generate_orders is False:
            assert self.orders.order_list


class SlapOrders:
    """this class groups together parameters that have to do with skus, orders,
    initial, pallets"""
    def __init__(self, n_skus: int, n_orders: int, generate_orders: bool,
                 desired_fill_level: float,
                 order_list: List[Tuple[str, int, int]],
                 initial_pallets_sku_counts: dict,
                 initial_pallets_storage_strategy,
                 n_storage_locations: int,
                 use_case: UseCase):
        """
        n_SKUs: int
            number of stock keeping units (SKU)/unique part numbers
        n_orders: int
            the number of retrieval and delivery orders (combined) that
            are added to the initial event heap queue (if an order list is not
            given). This is the main factor in the length of a simulation.
        generate_orders: bool
            determines if the user wants the simulation to generate its own
            orders from a distribution. should be set to false if user wants to
            give a list of orders
        desired_fill_level: float
            determines what percent of warehouse the should be fill level should
            fluctuate around. represents a percentage but should be inputted as
            a number between 0.0 and 1.0. If there are 100 storage locations
            and the fill level is 0.5, the simulation will try to keep around
            an average of 50 pallets in the warehouse at all times.
        order_list: list of tuples
            format of tuple: (delivery type: str, sku: int, arrival time: int)
        initial_pallets_sku_counts: dictionary
            user can provide a dictionary to decide how many pallets for each
            sku number should be added as initial pallets instead of generating
            automatically from a distribution. format {sku: quantity}. for
            example, {1: 5, 2: 10, 3: 7} would create five pallets of sku 1,
            ten pallets of sku 2, and seven pallets of sku 3.
        initial_pallets_storage_strategy: StoragePolicy
            user can provide a storage policy to determine how initial pallets
            are placed in the warehouse. if no policy is provided, their
            locations are just selected randomly.
        average_n_pallets: int
            average number of pallets that should be in the warehouse at one
            time. can fluctuate around this number.
        finished_orders: list of orders that were successfully completed.
        """
        self.n_SKUs = n_skus
        self.n_orders = n_orders
        self.generate_orders = generate_orders
        assert 0.0 <= desired_fill_level <= 0.5
        self.desired_fill_level = desired_fill_level
        self.initial_pallets_sku_counts = initial_pallets_sku_counts
        self.initial_pallets_storage_strategy = initial_pallets_storage_strategy
        self.average_n_pallets = int(self.desired_fill_level*n_storage_locations)
        self.order_list = order_list
        if use_case:
            self.generate_orders = False
            self.order_list = use_case.orders
            self.initial_pallets_sku_counts = use_case.initial_fill_level
            self.n_SKUs = use_case.n_skus
            self.n_orders = len(use_case.orders)
        if initial_pallets_storage_strategy:
            try:
                initial_pallets_storage_strategy.orders_to_assign_skus = use_case.orders_to_assign_first_skus
            except:
                pass


    def read_sku_from_dict(self) -> bool:
        """returns true if a dictionary was passed into the
        initial_pallets_sku_counts parameter
        """
        if isinstance(self.initial_pallets_sku_counts, dict):
            return True
        else:
            return False

    def follow_storage_strategy(self) -> bool:
        """returns true if a storage policy was passed into the
        initial_pallets_storage_strategy parameter"""
        if self.initial_pallets_storage_strategy is not None:
            return True
        else:
            return False

    def initial_pallets_as_np_array(self) -> bool:
        return False

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
