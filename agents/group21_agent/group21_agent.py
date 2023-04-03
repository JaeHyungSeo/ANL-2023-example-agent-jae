import logging
import random
import time
from typing import cast, Dict, List, Union

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.LearningDone import LearningDone
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Value import Value
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.Profile import Profile
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import ProfileConnectionFactory
from geniusweb.profileconnection.ProfileInterface import ProfileInterface
from geniusweb.progress.Progress import Progress
from geniusweb.progress.ProgressRounds import ProgressRounds
from tudelft.utilities.immutablelist.ImmutableList import ImmutableList
from tudelft.utilities.immutablelist.Outer import Outer

import numpy as np
from uri.uri import URI
from geniusweb.progress.ProgressRounds import ProgressRounds
from tudelft_utilities_logging.Reporter import Reporter


class Group21Agent(DefaultParty):
    def __init__(self, reporter: Reporter = None):
        super().__init__(reporter)

        # TODO

        self.getReporter().log(logging.INFO, "Agent initialized")

    # Informs the GeniusWeb system what protocols the agent supports and what type of profile it has.
    def getCapabilities(self) -> Capabilities:
        return Capabilities({"SAOP"}, {'geniusweb.profile.utilityspace.LinearAdditive'})

    # Gives a description of the agent.
    def getDescription(self) -> str:
        return 'Custom agent created by CAI group 21 (2023).'

    # Handles all interaction between the agent and the session.
    def notifyChange(self, info: Inform):
        # TODO: modify this function

        # First message sent in the negotiation - informs the agent about details of the negotiation session.
        if isinstance(info, Settings):
            self._session_settings = cast(Settings, info)

            self._id = self._session_settings.getID()
            self._session_progress = self._session_settings.getProgress()
            self._uri: str = str(self._session_settings.getProtocol().getURI())

            if "Learn" == str(self._session_settings.getProtocol().getURI()):
                self.getConnection().send(LearningDone(self._id))

            else:
                self._profile = ProfileConnectionFactory.create(info.getProfile().getURI(), self.getReporter())
                profile = self._profile.getProfile()

                # Finds all possible bids the agent can make and their corresponding utilities.
                if isinstance(profile, UtilitySpace):
                    issues = list(profile.getDomain().getIssues())
                    values: List[ImmutableList[Value]] = [profile.getDomain().getValues(issue) for issue in issues]
                    all_bids: Outer = Outer[Value](values)

                    for i in range(all_bids.size()):
                        bid = {}
                        for j in range(all_bids.get(i).size()):
                            bid[issues[j]] = all_bids.get(i).get(j)

                        utility = float(profile.getUtility(Bid(bid)))
                        self._all_bids.append((utility, bid))

                # Sorts by highest utility first.
                self._all_bids = sorted(self._all_bids, key=lambda x: x[0], reverse=True)

                self._max_utility = self._all_bids[0][0]
                self._min_utility = self._all_bids[len(self._all_bids) - 1][0]

        # Indicates that the opponent has ended their turn by accepting your last bid or offering a new one.
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()

            if isinstance(action, Offer):
                self._last_received_bid = cast(Offer, action).getBid()

        # Indicates that it is the agent's turn. The agent can accept the opponent's last bid or offer a new one.
        elif isinstance(info, YourTurn):
            action = self._execute_turn()

            if isinstance(self._session_progress, ProgressRounds):
                self._session_progress = self._session_progress.advance()
            
            self.getConnection().send(action)

        # Indicates that the session is complete - either the time has expired or a bid has been agreed on.
        elif isinstance(info, Finished):
            finished = cast(Finished, info)
            self.terminate()

        # Indicates that the information received was of an unknown type.
        else:
            self.getReporter().log(logging.WARNING, "Ignoring unknown info: " + str(info))

    # Terminates the agent and its connections.
    def terminate(self):
        self.getReporter().log(logging.INFO, "Agent is terminating...")

        super().terminate()

        if self._profile is not None:
            self._profile.close()
            self._profile = None

    ###############################################################
    # Functions below determine the agent's negotiation strategy. #
    ###############################################################

    # Processes the opponent's last bid and offers a new one if it isn't satisfactory.
    def _execute_turn(self):
        # TODO: modify this function
        if self._last_received_bid is not None:
            # Updates opponent preference profile model by incrementing issue values which appeared in the bid.
            issues = self._last_received_bid.getIssues()
            for issue in issues:
                value = self._last_received_bid.getValue(issue)

                if issue in self._opponent_model and value in self._opponent_model[issue]:
                    self._opponent_model[issue][value] += 1
                else:
                    if issue not in self._opponent_model:
                        self._opponent_model[issue] = {}

                    self._opponent_model[issue][value] = 1

            # Creates normalized opponent profile with updated values
            opponent_normalized_model: Dict[str, dict[Value, float]] = {}
            for issue, value in self._opponent_model.items():
                opponent_normalized_model[issue] = {}

                if len(self._opponent_model.get(issue).values()) > 0:
                    max_count = max(self._opponent_model.get(issue).values())

                    for discrete_value, count in self._opponent_model.get(issue).items():
                        opponent_normalized_model[issue][discrete_value] = count / max_count

            # Calculates the predicted utility that the opponent gains from their last proposed bid.
            opponent_utility = 0
            for issue in self._last_received_bid.getIssues():
                if issue in opponent_normalized_model:
                    value = self._last_received_bid.getValue(issue)
                    if value in opponent_normalized_model.get(issue):
                        opponent_utility += opponent_normalized_model.get(issue).get(value)
            opponent_utility = opponent_utility / len(self._last_received_bid.getIssues())
            self._opponent_bid_utilities.append(opponent_utility)

            # Predicts how much the opponent is conceding based on best-fit line gradient of previous proposed bids.
            opponent_concession_estimate = -1.0
            self._session_settings.getProgress().getTerminationTime()
            if self._session_progress.get(int(time.time())) > 0.1:
                variables = np.polyfit(
                    [x for x in range(0, 20)],
                    self._opponent_bid_utilities[
                        len(self._opponent_bid_utilities) - 21:
                        len(self._opponent_bid_utilities) - 1
                    ],
                    1
                )
                opponent_concession_estimate = variables[0] * 10

            # Checks if opponent is hard-lining and adjusts strategy.
            if abs(opponent_concession_estimate) < 0.001:
                self._accept_concession_param = self._accept_concession_param * 0.9
                self._offer_concession_param = self._offer_concession_param * 0.9

        if self._accept_bid(self._last_received_bid):
            action = Accept(self._id, self._last_received_bid)

        else:
            bid = self._create_bid()
            action = Offer(self._id, bid)

        return action

    # Checks if a bid should be accepted.
    def _accept_bid(self, bid: Bid) -> bool:
        # TODO

    # Creates a new bid to offer.
    def _create_bid(self) -> Bid:
        # TODOg