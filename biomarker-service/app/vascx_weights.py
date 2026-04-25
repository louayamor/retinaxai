"""Pre-download VascX weights for Docker builds."""

from __future__ import annotations

from loguru import logger


def main() -> None:
    try:
        from rtnls_inference import VascX
    except Exception as exc:
        raise RuntimeError(f"Failed to import rtnls_inference: {exc}") from exc

    logger.info("preloading vascx weights")
    VascX()
    logger.info("vascx weights ready")


if __name__ == "__main__":
    main()
