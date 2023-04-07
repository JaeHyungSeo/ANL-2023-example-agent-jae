import logging
from random import randint
from time import time
from typing import cast
import numpy as np

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.DiscreteValue import DiscreteValue
from geniusweb.issuevalue.NumberValue import NumberValue
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import (
    LinearAdditiveUtilitySpace,
)
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel


def extract_number_issues(bid: Bid):
    """
    Extracts numeric issue values from the given bid.
    """
    numeric_vals = []
    for iss_val in bid.getIssueValues().values():
        if iss_val is not NumberValue:
            continue
        numeric_vals.append(float(iss_val.getValue()))
    return np.array(numeric_vals)


def simulate_bid(bid: Bid, issue_deltas, steps, discount_factor=0.8):
    """
    Predicts the opponents bid X steps into the future.
    """
    future_issue_vals = extract_number_issues(bid)
    for i in range(steps):
        future_issue_vals += issue_deltas * (discount_factor ** (i + 1))

    # construct predicted bid
    predicted_issues_map = bid.getIssueValues()
    i = 0
    for issue in predicted_issues_map.keys():
        iss_val = predicted_issues_map[issue]
        if iss_val is not NumberValue:
            continue
        predicted_issues_map[issue] = NumberValue(future_issue_vals[i])
        i += 1
    return Bid(predicted_issues_map)


def has_numeric_issues(bid: Bid):
    """
    Checks whether the given bid has any numeric issues.
    """
    return len(extract_number_issues(bid) > 0)


class Group21Agent(DefaultParty):
    """
    Negotiation agent by Group 21.
    """

    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.logger.log(logging.INFO, "party is initialized")

        # store history of our agent's actions (offers). 
        self._my_actions = []

    # get progress time normalized from 0 to 1
    def _get_progress(self):    
        elapsed    = (time() - self.progress.getStart().timestamp()) * 1000
        time_limit = self.progress.getDuration()
        return elapsed / time_limit

    def notifyChange(self, data: Inform):
        """MUST BE IMPLEMENTED
        This is the entry point of all interaction with your agent after is has been initialised.
        How to handle the received data is based on its class type.

        Args:
            info (Inform): Contains either a request for action or information.
        """

        # a Settings message is the first message that will be send to your
        # agent containing all the information about the negotiation session.
        if isinstance(data, Settings):
            self.settings = cast(Settings, data)
            self.me = self.settings.getID()

            # progress towards the deadline has to be tracked manually through the use of the Progress object
            self.progress = self.settings.getProgress()

            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            # the profile contains the preferences of the agent over the domain
            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter()
            )
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

        # ActionDone informs you of an action (an offer or an accept)
        # that is performed by one of the agents (including yourself).
        elif isinstance(data, ActionDone):
            action = cast(ActionDone, data).getAction()
            actor = action.getActor()

            # ignore action if it is our action
            if actor != self.me:
                # obtain the name of the opponent, cutting of the position ID.
                self.other = str(actor).rsplit("_", 1)[0]

                # process action done by opponent
                self.opponent_action(action)
        # YourTurn notifies you that it is your turn to act
        elif isinstance(data, YourTurn):
            # execute a turn
            self.my_turn()

        # Finished will be send if the negotiation has ended (through agreement or deadline)
        elif isinstance(data, Finished):
            self.save_data()
            # terminate the agent MUST BE CALLED
            self.logger.log(logging.INFO, "party is terminating:")
            super().terminate()
        else:
            self.logger.log(logging.WARNING, "Ignoring unknown info " + str(data))

    def getCapabilities(self) -> Capabilities:
        """
        Method to indicate to the protocol what the capabilities of this agent are.
        Leave it as is for the ANL 2022 competition

        Returns:
            Capabilities: Capabilities representation class
        """
        return Capabilities(
            set(["SAOP"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    def send_action(self, action: Action):
        """Sends an action to the opponent(s)

        Args:
            action (Action): action of this agent
        """
        # store action
        self._my_actions.append(action)
        # send action
        self.getConnection().send(action)

    # give a description of your agent
    def getDescription(self) -> str:
        """MUST BE IMPLEMENTED
        Returns a description of your agent. 1 or 2 sentences.

        Returns:
            str: Agent description
        """
        return "Negotiation agent implemented by Group 21"

    def opponent_action(self, action):
        """Process an action that was received from the opponent.

        Args:
            action (Action): action of opponent
        """
        # if it is an offer, set the last received bid
        if isinstance(action, Offer):
            # create opponent model if it was not yet initialised
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = cast(Offer, action).getBid()

            # update opponent model with bid
            self.opponent_model.update(bid)
            # set bid as last received
            self.last_received_bid = bid

    def my_turn(self):
        """This method is called when it is our turn. It should decide upon an action
        to perform and send this action to the opponent.
        """
        # check if the last received offer is good enough
        if self.accept_condition(self.last_received_bid):
            # if so, accept the offer
            action = Accept(self.me, self.last_received_bid)
        else:
            # if not, find a bid to propose as counter offer
            bid = self.find_bid()
            action = Offer(self.me, bid)

        # send the action
        self.send_action(action)

    def save_data(self):
        """This method is called after the negotiation is finished. It can be used to store data
        for learning capabilities. Note that no extensive calculations can be done within this method.
        Taking too much time might result in your agent being killed, so use it for storage only.
        """
        # TODO: implement learning maybe
        # data = "Data for learning (see README.md)"
        # with open(f"{self.storage_dir}/data.md", "w") as f:
        #     f.write(data)
        return

    ###########################################################################################
    #################################### Our methods below ####################################
    ###########################################################################################

    def accept_condition(self, bid: Bid) -> bool:
        """
        Decides whether the given bid should be accepted or rejected.
        """
        if bid is None:
            return False

        conditions = [
            self.accept_condition_derivative(bid),
            self.accept_condition_time(bid),
        ]
        return all(conditions)

    def accept_condition_derivative(self, bid: Bid):
        """
        Checks whether the given bid should be accepted based on the derivative condition.

        """
        # ensure that the bid has any numeric issues
        if not has_numeric_issues(bid):
            return True

        # ensure there are at last 2 past bids
        # ! the current bid is already in the list of past bids
        opp_model = self.opponent_model
        if len(opp_model.offers) < 2:
            return False

        # estimate issue deltas
        issue_deltas = []
        last_issue_vals = None
        for past_bid in opp_model.offers:
            # get numeric issue values from bid
            print(f"Issues: {past_bid.getIssueValues()}")
            issue_values = extract_number_issues(past_bid)
            print(f"Issue vals: {issue_values}")

            # skip if first in the list
            if last_issue_vals is None:
                last_issue_vals = issue_values
                continue

            # calculate deltas
            deltas = issue_values - last_issue_vals
            issue_deltas.append(deltas)
            last_issue_vals = issue_values

        # simulate next bid
        last_deltas = issue_deltas[len(issue_deltas) - 1]

        # construct predicted bid
        predicted_bid = simulate_bid(bid, last_deltas, 5)

        # get utility of current bid
        curr_utility = self.profile.getUtility(bid)
        pred_utility = self.profile.getUtility(predicted_bid)

        return curr_utility > pred_utility

    def accept_condition_time(self, bid: Bid):
        """
        Checks whether the given bid should be accepted based on the time condition.
        """
        # progress (normalized from 0 to 1 with 1 is deadline)
        progress = self._get_progress()
        # oppononent's bid utility value
        bid_util = float(self.profile.getUtility(bid))
        # init default threshold 
        acc_thresh = 0.9
        # compute the minimum utility value mapped from reservation to the first utility based on time 
        if len(self._my_actions) > 0:
            # best utility value (first offer)
            max_util_val = float(self.profile.getUtility(self._my_actions[0].getBid()))
            # min acceptable utility
            if self.profile.getReservationBid() is not None:
                # use reservation value if there is one 
                min_util_val = float(self.profile.getUtility(self.profile.getReservationBid()))
            else: 
                # use the average between the latest bid offer from the opponent (assumed to be the best bid they can offer)
                # and our last bid 
                min_util_val = float(
                    self.profile.getUtility(self.last_received_bid) + self.profile.getUtility(self._my_actions[-1].getBid())
                ) / 2
            # determine threshold for acceptance 
            acc_thresh = min_util_val + (1 - progress) * (max_util_val - min_util_val)
        # result
        return bid_util >= acc_thresh

    def find_bid(self) -> Bid:
        # compose a list of all possible bids
        domain = self.profile.getDomain()
        all_bids = AllBidsList(domain)

        best_bid_score = 0.0
        best_bid = None

        # take 500 attempts to find a bid according to a heuristic score
        for _ in range(500):
            bid = all_bids.get(randint(0, all_bids.size() - 1))
            bid_score = self.score_bid(bid)
            if bid_score > best_bid_score:
                best_bid_score, best_bid = bid_score, bid

        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.95, eps: float = 0.1) -> float:
        """Calculate heuristic score for a bid

        Args:
            bid (Bid): Bid to score
            alpha (float, optional): Trade-off factor between self interested and
                altruistic behaviour. Defaults to 0.95.
            eps (float, optional): Time pressure factor, balances between conceding
                and Boulware behaviour over time. Defaults to 0.1.

        Returns:
            float: score
        """
        progress = self.progress.get(time() * 1000)

        our_utility = float(self.profile.getUtility(bid))

        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            opponent_score = (1.0 - alpha * time_pressure) * opponent_utility
            score += opponent_score

        return score
