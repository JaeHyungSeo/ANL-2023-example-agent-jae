import logging, time 
from random import randint
import traceback
from typing import cast, Dict, List, Set, Collection

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.LearningDone import LearningDone
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.actions.Vote import Vote
from geniusweb.actions.Votes import Votes
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.OptIn import OptIn
from geniusweb.inform.Settings import Settings
from geniusweb.inform.Voting import Voting
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.issuevalue.Value import Value
from geniusweb.issuevalue.ValueSet import ValueSet
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.UtilitySpace import UtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import (
    ProfileConnectionFactory,
)
from geniusweb.progress.ProgressRounds import ProgressRounds
from geniusweb.utils import val


class Group21Agent(DefaultParty):
    """
    Offers random bids until a bid with sufficient utility is offered.
    """

    def __init__(self):
        super().__init__()
        self.getReporter().log(logging.INFO, "party is initialized")
        self._profile = None
        self._lastReceivedBid: Bid = None

        # store history of our agent's actions (offers). 
        self._my_actions = []

    # get progress time normalized from 0 to 1
    def _get_progress(self):    
        elapsed    = (time.time() - self._progress.getStart().timestamp()) * 1000
        time_limit = self._progress.getDuration()
        return elapsed / time_limit

    # Override
    def notifyChange(self, info: Inform):
        # self.getReporter().log(logging.INFO,"received info:"+str(info))
        if isinstance(info, Settings):
            self._settings: Settings = cast(Settings, info)
            self._me = self._settings.getID()
            self._protocol: str = str(self._settings.getProtocol().getURI())
            self._progress = self._settings.getProgress()
            if "Learn" == self._protocol:
                self.getConnection().send(LearningDone(self._me))  # type:ignore
            else:
                self._profile = ProfileConnectionFactory.create(
                    info.getProfile().getURI(), self.getReporter()
                )
        elif isinstance(info, ActionDone):
            action: Action = cast(ActionDone, info).getAction()
            if isinstance(action, Offer):
                self._lastReceivedBid = cast(Offer, action).getBid()
        elif isinstance(info, YourTurn):
            self._myTurn()
            if isinstance(self._progress, ProgressRounds):
                self._progress = self._progress.advance()
        elif isinstance(info, Finished):
            self.terminate()
        elif isinstance(info, Voting):
            # MOPAC protocol
            self._lastvotes = self._vote(cast(Voting, info))
            val(self.getConnection()).send(self._lastvotes)
        elif isinstance(info, OptIn):
            val(self.getConnection()).send(self._lastvotes)
        else:
            self.getReporter().log(
                logging.WARNING, "Ignoring unknown info " + str(info)
            )

    # Override
    def getCapabilities(self) -> Capabilities:
        return Capabilities(
            set(["SAOP", "Learn", "MOPAC"]),
            set(["geniusweb.profile.utilityspace.LinearAdditive"]),
        )

    # Override
    def getDescription(self) -> str:
        return "Offers random bids until a bid with sufficient utility is offered. Parameters minPower and maxPower can be used to control voting behaviour."

    # Override
    def terminate(self):
        self.getReporter().log(logging.INFO, "party is terminating:")
        super().terminate()
        if self._profile != None:
            self._profile.close()
            self._profile = None

    def _myTurn(self):
        if self._isGood(self._lastReceivedBid):
            action = Accept(self._me, self._lastReceivedBid)
        else:
            # TODO: bidding stratgy
            for _attempt in range(20):
                bid = self._getRandomBid(self._profile.getProfile().getDomain())
                if self._isGood(bid):
                    break
            action = Offer(self._me, bid)
            # store action. Do not remove (tan wrote this)
            self._my_actions.append(action)
        self.getConnection().send(action)

    # TODO: acceptance strategy
    def _isGood(self, bid: Bid) -> bool:
        # base case 
        if bid == None:
            return False

        # get our agent profile
        profile = self._profile.getProfile()

        # progress (normalized from 0 to 1)
        progress = self._get_progress()

        # oppononent's bid utility value
        bid_util = float(profile.getUtility(bid))

        # init default threshold 
        acc_thresh = 0.9
        # compute the minimum utility value mapped from reservation to the first utility based on time 
        if len(self._my_actions) > 0:
            # best utility value (first offer)
            max_util_val = float(profile.getUtility(self._my_actions[0].getBid()))
            # min acceptable utility
            if profile.getReservationBid() is not None:
                # use reservation value if there is one 
                min_util_val = float(profile.getUtility(profile.getReservationBid()))
            else: 
                # use the latest bid offer 
                min_util_val = float(profile.getUtility(self._my_actions[-1].getBid()))
            # determine threshold for acceptance 
            acc_thresh = min_util_val + (1 - progress) * (max_util_val - min_util_val)

        # only accept the offer when ALL conditions are satisfied
        try:
            return all(
                [bid_util >= acc_thresh]
            )
        except:
            raise Exception("Can not handle this type of profile")

    def _getRandomBid(self, domain: Domain) -> Bid:
        allBids = AllBidsList(domain)
        return allBids.get(randint(0, allBids.size() - 1))

    def _vote(self, voting: Voting) -> Votes:
        """
        @param voting the {@link Voting} object containing the options

        @return our next Votes.
        """
        val = self._settings.getParameters().get("minPower")
        minpower: int = val if isinstance(val, int) else 2
        val = self._settings.getParameters().get("maxPower")
        maxpower: int = val if isinstance(val, int) else 9999999

        votes: Set[Vote] = set(
            [
                Vote(self._me, offer.getBid(), minpower, maxpower)
                for offer in voting.getOffers()
                if self._isGood(offer.getBid())
            ]
        )
        return Votes(self._me, votes)
