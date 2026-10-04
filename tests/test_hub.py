from types import SimpleNamespace

from vla import hub


def test_save_last_and_auto_resume(tmp_path):
    run = tmp_path / "runs" / "x"
    (run / "step_10").mkdir(parents=True)
    (run / "step_10" / "trainer.pt").write_text("a")
    cfg = SimpleNamespace(run=str(run), resume="auto", hub_repo="")
    assert hub.resolve_resume(cfg) == ""              # nothing saved as `last` yet
    hub.save_last(run, "step_10", "")
    assert (run / "last" / "trainer.pt").read_text() == "a"
    assert hub.resolve_resume(cfg) == str(run / "last")
    (run / "step_20").mkdir()
    (run / "step_20" / "trainer.pt").write_text("b")
    hub.save_last(run, "step_20", "")                 # replaces the previous `last`
    assert (run / "last" / "trainer.pt").read_text() == "b"
    cfg.resume = str(run / "step_10")
    assert hub.resolve_resume(cfg) == str(run / "step_10")
