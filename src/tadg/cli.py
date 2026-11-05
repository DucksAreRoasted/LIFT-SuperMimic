"""uv-managed TADG training entry point."""


def main():
    import tyro
    from tadg.trainer import TADGConfig, TADGTrainer

    cfg = tyro.cli(TADGConfig)
    trainer = TADGTrainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()
        if trainer.writer is not None:
            trainer.writer.close()
