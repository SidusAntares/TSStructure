import torch
from test_segment_smoke import batch, cfg
from test_segment_smoke_launcher import entry, args, setup_run, run_script, TASKS


def test_joint_configuration_and_no_source_weight_loading(tmp_path, monkeypatch):
    from utils.segment_smoke import initialize_source_model
    config = entry().build_config([*args(tmp_path), '--source_init','scratch','--smoke_epochs','10'])
    assert config.lr == .001 and config.selector_lr == .0005 and config.scheduler_epochs == 100
    monkeypatch.setattr(torch, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not load weights')))
    assert initialize_source_model(torch.nn.Linear(2,2), config) is False


def test_joint_ten_epochs_exports_all_epoch_selection_statistics(tmp_path):
    from models.stclassifier import PseLTae
    from models.segment_selector import SegmentSelector
    from utils.segment_smoke import train_smoke_steps
    config = cfg(tmp_path)
    config.source_init, config.smoke_epochs, config.steps = 'scratch', 10, None
    config.scheduler_epochs, config.lr, config.selector_lr = 100, .001, .0005
    config.variant, config.sparse_weight = 'inshape', 3.
    config.selector_protocol = 'inshape_coverage_v1'
    result = train_smoke_steps(PseLTae(num_classes=3, with_extra=False),
        SegmentSelector(width=32, heads=4, dropout=.1, protocol=config.selector_protocol), [batch()], [batch()], config)
    assert result['completed_source_epochs'] == 10 and result['source_checkpoint_loaded'] is False
    assert len(result['epoch_training']) == 10 and result['epoch_training'][-1]['samples'] == 4
    assert result['scheduler_total_steps'] == 100


def test_joint_four_gpu_command_mapping(tmp_path):
    import shlex
    setup_run(tmp_path)
    assert run_script(tmp_path, SOURCE_INIT='scratch', INSHAPE_ONLY='1', SMOKE_EPOCHS='10', GPUS='0 1 2 3').returncode == 0
    for task in TASKS:
        tokens = shlex.split((tmp_path/f'logs/test_smoke/{task}_inshape.log').read_text())
        config = entry().build_config(tokens[2:])
        assert config.source_init == 'scratch' and config.smoke_epochs == 10
