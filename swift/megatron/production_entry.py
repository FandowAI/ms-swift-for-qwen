"""Stable module entrypoint for the packing-v2 Megatron SFT recipe."""

from __future__ import annotations

import os


def main():
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from swift.megatron import megatron_sft_main

    return megatron_sft_main()


if __name__ == '__main__':
    main()
