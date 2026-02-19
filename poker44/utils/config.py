"""Bittensor configuration helpers vendored for the Poker44 subnet."""

from __future__ import annotations

import argparse

import bittensor as bt
import os
import traceback

traceback.format_exc()


def add_args(cls, parser: argparse.ArgumentParser) -> None:
    if parser is None:
        parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.Axon.add_args(parser)
    
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)
    
    parser.add_argument(
        "--neuron.device",
        type=str,
        default="cpu",
        help="Torch device to execute forwards on (cpu, cuda:0, ...).",
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        default=50,
        help="Blocks between mandatory syncs.",
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Skip setting weights on-chain.",
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        default=0.05,
        help="Exponential moving average smoothing factor for scores.",
    )
    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        default=1,
        help="Concurrent forward coroutines to execute per step.",
    )
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=60,
        help="Default delay between validator ingestion cycles.",
    )
    parser.add_argument(
        "--neuron.axon_off",
        action="store_true",
        help="Disable serving the axon endpoint.",
    )
    parser.add_argument(
    "--blacklist.force_validator_permit",
    action="store_true",
    default=True,
    help="Only allow requests from validators with permits.",
    )
    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_false",
        default=False,
        help="Allow requests from non-registered entities.",
    )

def add_validator_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validator.manual_players",
        nargs="*",
        default=[],
        help="Player descriptors to track manually (player_uid[:label]).",
    )


def add_miner_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--miner.mock",
        action="store_true",
        help="Placeholder flag retained for compatibility.",
    )



def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    # if not config.neuron.dont_save_events:
    #     # Add custom event logger for the events.
    #     events_logger = setup_events_logger(
    #         config.neuron.full_path, config.neuron.events_retention_size
    #     )
    #     bt.logging.register_primary_logger(events_logger.name)


def config(cls) -> bt.Config:
    parser = argparse.ArgumentParser()
    cls.add_args(parser)
    return bt.Config(parser=parser)
